#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <fstream>
#include <iomanip>
#include <jbasis.hh>   // your Jacobi basis evaluator

// ===============================================================
//  Simple vector helpers
// ===============================================================

static double dotP(const std::vector<double>& a,
                   const std::vector<double>& b,
                   int P)
{
  double s = 0.0;
  for (int p = 0; p < P; ++p)
    s += a[p] * b[p];
  return s;
}

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
      // check if column j is non-trivial
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

// Affine map helper for triangle 1
struct TriangleMap
{
  double x0[2];
  double x1[2];
  double x2[2];
  double Ainv[2][2];

  void build(const double v0[2],
             const double v1[2],
             const double v2[2])
  {
    x0[0] = v0[0]; x0[1] = v0[1];
    x1[0] = v1[0]; x1[1] = v1[1];
    x2[0] = v2[0]; x2[1] = v2[1];

    double a11 = x1[0] - x0[0];
    double a21 = x1[1] - x0[1];
    double a12 = x2[0] - x0[0];
    double a22 = x2[1] - x0[1];

    double det = a11*a22 - a12*a21;
    assert(std::fabs(det) > 1e-14);
    double invdet = 1.0/det;

    Ainv[0][0] =  a22 * invdet;
    Ainv[0][1] = -a12 * invdet;
    Ainv[1][0] = -a21 * invdet;
    Ainv[1][1] =  a11 * invdet;
  }

  // physical -> reference
  inline void phys_to_ref(double x, double y,
                          double &xi, double &eta) const
  {
    double dx = x - x0[0];
    double dy = y - x0[1];
    xi  = Ainv[0][0]*dx + Ainv[0][1]*dy;
    eta = Ainv[1][0]*dx + Ainv[1][1]*dy;
  }
};

// ===============================================================
//  Gram check
// ===============================================================

static void gram_check(const std::vector<double>& Q,
                       int P, int M,
                       const std::string& label)
{
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

  std::cout << "\n" << label << " Q^T Q CHECK:\n";
  std::cout << "  Frobenius norm ||Q^T Q - I||_F = " << frob << "\n";
  std::cout << "  max |offdiag(Q^T Q - I)|        = " << max_abs_off << "\n";
  std::cout << "  diag range                      = ["
            << min_diag << ", " << max_diag << "]\n";
}

// ===============================================================
//  Save Q to disk (row-major for text): each line p: Q[p,0..M-1]
// ===============================================================

static bool save_Q_to_file(const std::string& filename,
                           int p_max, int M, int P,
                           const std::vector<int>& pix_idx,
                           const std::vector<double>& Q)
{
  std::ofstream out(filename.c_str());
  if (!out)
  {
    std::cerr << "ERROR: could not open " << filename << " for writing.\n";
    return false;
  }

  // FULL double precision, scientific format
  out << std::setprecision(17) << std::scientific;

  // header
  out << p_max << " " << M << " " << P << "\n";

  // pixel indices
  for (int p = 0; p < P; ++p)
  {
    out << pix_idx[p];
    if (p + 1 < P) out << " ";
  }
  out << "\n";

  // Q matrix in row-major (each row is pixel p, all M entries)
  for (int p = 0; p < P; ++p)
  {
    for (int m = 0; m < M; ++m)
    {
      double v = Q[p + m * P];
      out << v;
      if (m + 1 < M) out << " ";
    }
    out << "\n";
  }

  std::cout << "Saved full-precision Q to " << filename << "\n";
  return true;
}

// ===============================================================
//  Main: precompute Q for both triangles and save to disk
// ===============================================================

int main()
{
  using Basis2 = jsimplex::Basis<2,double>;

  // -------------------- degree setup ----------------------------
  int p_max = 6;  // total degree
  int M = Basis2::dim_Pi(p_max);
  std::cout << "Using total-degree p_max=" << p_max
            << ", M=" << M << " modes\n";

  // -------------------- Jacobi meta -----------------------------
  double kappa[3] = {0.5, 0.5, 0.5};
  std::vector<int>    alpha_table(M * 2);
  std::vector<int>    tail_deg   (M * 2);
  std::vector<double> inv_h      (M);

  Basis2::build_structures(
    kappa, p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data()
  );

  // ---------------------------------------------------------------
  // Triangle 0: ref triangle (0,0),(1,0),(0,1)
  // ---------------------------------------------------------------
  std::vector<double> X_pix0;   // (xi,eta)
  std::vector<int>    pix_idx0;

  X_pix0.reserve(64 * 2);
  pix_idx0.reserve(64);

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      double x = (i + 0.5) / 8.0;
      double y = (j + 0.5) / 8.0;

      if (x + y <= 1.0)
      {
        X_pix0.push_back(x);   // xi
        X_pix0.push_back(y);   // eta
        pix_idx0.push_back(i*8 + j);
      }
    }
  }

  int P0 = (int)pix_idx0.size();
  std::cout << "Triangle 0: P0 = " << P0 << " pixel nodes\n";

  std::vector<double> V0(P0 * M);
  Basis2::eval_all(
    X_pix0.data(),
    /*ld_point=*/2,
    /*ld_dim=*/1,
    /*npts=*/P0,
    kappa,
    p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data(),
    V0.data(),
    /*ld_V=*/P0,
    nullptr
  );

  std::vector<double> Q0;
  int rank0 = gram_schmidt_orthonormalize(V0, P0, M, Q0, 1e-12);
  std::cout << "Triangle 0: Gram–Schmidt rank = " << rank0
            << " (out of M=" << M << ")\n";

  gram_check(Q0, P0, M, "Triangle 0");

  // ---------------------------------------------------------------
  // Triangle 1: physical tri (1,1),(0,1),(1,0) mapped to ref tri
  // ---------------------------------------------------------------
  TriangleMap tri1;
  {
    double v0[2] = {1.0, 1.0};
    double v1[2] = {0.0, 1.0};
    double v2[2] = {1.0, 0.0};
    tri1.build(v0, v1, v2);
  }

  std::vector<double> X_pix1;
  std::vector<int>    pix_idx1;

  X_pix1.reserve(64 * 2);
  pix_idx1.reserve(64);

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      double x = (i + 0.5) / 8.0;
      double y = (j + 0.5) / 8.0;

      if (x + y > 1.0)
      {
        double xi, eta;
        tri1.phys_to_ref(x, y, xi, eta);
        X_pix1.push_back(xi);
        X_pix1.push_back(eta);
        pix_idx1.push_back(i*8 + j);
      }
    }
  }

  int P1 = (int)pix_idx1.size();
  std::cout << "Triangle 1: P1 = " << P1 << " pixel nodes\n";

  std::vector<double> V1(P1 * M);
  Basis2::eval_all(
    X_pix1.data(),
    /*ld_point=*/2,
    /*ld_dim=*/1,
    /*npts=*/P1,
    kappa,
    p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data(),
    V1.data(),
    /*ld_V=*/P1,
    nullptr
  );

  std::vector<double> Q1;
  int rank1 = gram_schmidt_orthonormalize(V1, P1, M, Q1, 1e-12);
  std::cout << "Triangle 1: Gram–Schmidt rank = " << rank1
            << " (out of M=" << M << ")\n";

  gram_check(Q1, P1, M, "Triangle 1");

  // ---------------------------------------------------------------
  // Save Q0 and Q1 to disk
  // ---------------------------------------------------------------
  std::string file0 = "jacobi_Q_tri0_p6.txt";
  std::string file1 = "jacobi_Q_tri1_p6.txt";

  bool ok0 = save_Q_to_file(file0, p_max, M, P0, pix_idx0, Q0);
  bool ok1 = save_Q_to_file(file1, p_max, M, P1, pix_idx1, Q1);

  if (!ok0 || !ok1)
  {
    std::cerr << "ERROR: failed to save one or both Q files.\n";
    return 1;
  }

  std::cout << "Done precomputing Q for both triangles.\n";
  return 0;
}
