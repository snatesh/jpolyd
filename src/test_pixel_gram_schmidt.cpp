#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>

#include <jbasis.hh>   // your Jacobi basis evaluator

// Discrete inner product on R^P: <a,b> = sum_p a[p] * b[p]
static double dotP(const std::vector<double>& a,
                   const std::vector<double>& b,
                   int P)
{
  double s = 0.0;
  for (int p = 0; p < P; ++p)
    s += a[p] * b[p];
  return s;
}

// Euclidean norm of vector in R^P
static double normP(const std::vector<double>& a, int P)
{
  return std::sqrt(dotP(a,a,P));
}

// Modified Gram–Schmidt on columns of V (P x M) -> Q (P x M)
static int gram_schmidt_orthonormalize(const std::vector<double>& V,
                                       int P, int M,
                                       std::vector<double>& Q,
                                       double tol = 1e-12)
{
  Q.assign(P * M, 0.0);
  int rank = 0;

  std::vector<double> qm(P);

  for (int m = 0; m < M; ++m)
  {
    // Start with column m of V
    for (int p = 0; p < P; ++p)
      qm[p] = V[p + m*P];

    // Subtract projections onto previous q_j
    for (int j = 0; j < m; ++j)
    {
      // q_j is column j of Q
      // If that column is zeroed (dependent), skip
      double norm_qj = 0.0;
      for (int p = 0; p < P; ++p)
        norm_qj += Q[p + j*P] * Q[p + j*P];
      if (norm_qj < tol) continue;

      double r_jm = 0.0;
      for (int p = 0; p < P; ++p)
        r_jm += Q[p + j*P] * qm[p];

      for (int p = 0; p < P; ++p)
        qm[p] -= r_jm * Q[p + j*P];
    }

    double nrm = normP(qm, P);
    if (nrm < tol)
    {
      // Nearly dependent column; leave Q column as zeros.
      std::cerr << "Column " << m << " is nearly dependent; skipping.\n";
      continue;
    }

    // Normalize and store in Q
    for (int p = 0; p < P; ++p)
      Q[p + m*P] = qm[p] / nrm;

    ++rank;
  }

  return rank;
}

// Simple MSE/PSNR on a pixel vector f (P) vs f_rec (P)
static void mse_psnr(const std::vector<double>& f,
                     const std::vector<double>& f_rec,
                     int P,
                     double& mse,
                     double& psnr,
                     double max_val = 255.0)
{
  mse = 0.0;
  for (int p = 0; p < P; ++p)
  {
    double diff = f[p] - f_rec[p];
    mse += diff * diff;
  }
  mse /= (double)P;

  if (mse <= 0.0)
  {
    psnr = std::numeric_limits<double>::infinity();
  }
  else
  {
    psnr = 10.0 * std::log10((max_val*max_val) / mse);
  }
}

int main()
{
  using Basis2 = jsimplex::Basis<2,double>;

  // ---------------------------------------------------------------
  // Pick polynomial degree
  // ---------------------------------------------------------------
  int p_max = 6;  // or try 5 if you want a more compact basis
  int M = Basis2::dim_Pi(p_max);
  std::cout << "Using total-degree p_max=" << p_max
            << ", M=" << M << " modes\n";

  // ---------------------------------------------------------------
  // Setup Jacobi meta
  // ---------------------------------------------------------------
  double kappa[3] = {0.5,0.5,0.5};
  std::vector<int>    alpha_table(M*2);
  std::vector<int>    tail_deg   (M*2);
  std::vector<double> inv_h      (M);

  Basis2::build_structures(
    kappa, p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data()
  );

  // ---------------------------------------------------------------
  // Build pixel nodes in triangle 0
  // ---------------------------------------------------------------
  std::vector<double> X_pix;    // flattened (xi,eta) pairs
  X_pix.reserve(64*2);

  std::vector<int> pix_idx;     // store pixel indices (i*8 + j)
  pix_idx.reserve(64);

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      double x = (i + 0.5)/8.0;
      double y = (j + 0.5)/8.0;

      if (x + y <= 1.0)   // in triangle 0
      {
        X_pix.push_back(x);   // xi
        X_pix.push_back(y);   // eta
        pix_idx.push_back(i*8 + j);
      }
    }
  }

  int P = (int)pix_idx.size();
  std::cout << "Triangle 0: P=" << P << " pixel nodes\n";

  // ---------------------------------------------------------------
  // Build Vandermonde V (P x M)
  // V[p + m*P] = phi_m(x_p)
  // ---------------------------------------------------------------
  std::vector<double> V(P * M);

  Basis2::eval_all(
    X_pix.data(),
    /*ld_point=*/2,
    /*ld_dim=*/1,
    /*npts=*/P,
    kappa,
    p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data(),
    V.data(),
    /*ld_V=*/P,
    nullptr
  );

  // ---------------------------------------------------------------
  // Run Gram–Schmidt on columns of V to get Q (P x M)
  // ---------------------------------------------------------------
  std::vector<double> Q;
  int rank = gram_schmidt_orthonormalize(V, P, M, Q, 1e-12);

  std::cout << "Gram–Schmidt rank: " << rank << " (out of M=" << M << ")\n";

  // ---------------------------------------------------------------
  // Check Q^T Q ~ I
  // ---------------------------------------------------------------
  std::vector<double> G(M * M, 0.0);
  for (int m = 0; m < M; ++m)
  {
    for (int n = 0; n < M; ++n)
    {
      double sum = 0.0;
      for (int p = 0; p < P; ++p)
        sum += Q[p + m*P] * Q[p + n*P];
      G[m*M + n] = sum;
    }
  }

  double max_abs_off = 0.0;
  double frob = 0.0;
  double min_diag = 1e300;
  double max_diag = -1e300;

  for (int m = 0; m < M; ++m)
  {
    for (int n = 0; n < M; ++n)
    {
      double target = (m == n ? 1.0 : 0.0);
      double diff   = G[m*M + n] - target;
      frob += diff * diff;
      if (m == n)
      {
        min_diag = std::min(min_diag, G[m*M + n]);
        max_diag = std::max(max_diag, G[m*M + n]);
      }
      else
      {
        max_abs_off = std::max(max_abs_off, std::fabs(diff));
      }
    }
  }
  frob = std::sqrt(frob);

  std::cout << "\nQ^T Q CHECK:\n";
  std::cout << "  Frobenius norm ||Q^T Q - I||_F = " << frob << "\n";
  std::cout << "  max |offdiag(Q^T Q - I)|        = " << max_abs_off << "\n";
  std::cout << "  diag range                      = ["
            << min_diag << ", " << max_diag << "]\n";

  // ---------------------------------------------------------------
  // Test a roundtrip encode/decode on a synthetic pixel block
  // ---------------------------------------------------------------
  std::vector<double> f(P), c(M), f_rec(P);

  // Synthetic data: e.g. f[p] = i + 0.5*j from global pixel coords
  for (int k = 0; k < P; ++k)
  {
    int ij = pix_idx[k];
    int i = ij / 8;
    int j = ij % 8;
    f[k] = (double)i + 0.5 * (double)j;
  }

  // Encode: c[m] = sum_p f[p] * Q[p + m*P]
  for (int m = 0; m < M; ++m)
  {
    double cm = 0.0;
    for (int p = 0; p < P; ++p)
      cm += f[p] * Q[p + m*P];
    c[m] = cm;
  }

  // Decode: f_rec[p] = sum_m c[m] * Q[p + m*P]
  for (int p = 0; p < P; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M; ++m)
      val += c[m] * Q[p + m*P];
    f_rec[p] = val;
  }

  double mse, psnr;
  mse_psnr(f, f_rec, P, mse, psnr, /*max_val=*/255.0);

  std::cout << "\nRoundtrip on synthetic f:\n";
  std::cout << "  MSE  = " << mse  << "\n";
  std::cout << "  PSNR = " << psnr << " dB\n";

  return 0;
}
