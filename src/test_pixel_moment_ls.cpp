#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include <jbasis.hh>   // your Jacobi basis evaluator

// ===============================================================
//  Build pixel nodes for a triangle in the 8x8 square
// ===============================================================

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
// Solve SPD system A x = rhs via naive Cholesky.  A is size n*n.
// ===============================================================

static bool solve_cholesky(std::vector<double> &A,
                           std::vector<double> &x,
                           const std::vector<double> &rhs,
                           int n)
{
  // A overwritten with L in lower triangle.

  // Factorization
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      double sum = A[i*n + j];
      for (int k = 0; k < j; ++k)
        sum -= A[i*n + k] * A[j*n + k];
      if (i == j)
      {
        if (sum <= 0.0) return false;
        A[i*n + j] = std::sqrt(sum);
      }
      else
      {
        A[i*n + j] = sum / A[j*n + j];
      }
    }
    // zero the upper
    for (int j = i+1; j < n; ++j)
      A[i*n + j] = 0.0;
  }

  // Forward solve L y = rhs
  std::vector<double> y(n);
  for (int i = 0; i < n; ++i)
  {
    double sum = rhs[i];
    for (int k = 0; k < i; ++k)
      sum -= A[i*n + k] * y[k];
    y[i] = sum / A[i*n + i];
  }

  // Back solve L^T x = y
  for (int i = n-1; i >= 0; --i)
  {
    double sum = y[i];
    for (int k = i+1; k < n; ++k)
      sum -= A[k*n + i] * x[k];
    x[i] = sum / A[i*n + i];
  }

  return true;
}


// ===============================================================
// Main: solve discrete moment fitting on pixel nodes of a triangle
// ===============================================================

int main()
{
  using Basis2 = jsimplex::Basis<2,double>;

  // ---------------------------------------------------------------
  // Pick polynomial degree
  // ---------------------------------------------------------------
  int p_max = 4;  // try 6 or 7; 6 is safer vs number of pixels
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
  // Build pixel nodes on triangle 0
  // Triangle 0 = (0,0),(1,0),(0,1). Pixel centers already in this.
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
        X_pix.push_back(x);   // xi = x
        X_pix.push_back(y);   // eta= y
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
  // Build LS system:  K w = b, with K in R^{M^2 x P}, b = vec(I)
  // K_{(m,n),p} = V(p,m)*V(p,n)
  // ---------------------------------------------------------------

  int Krows = M*M;
  int Kcols = P;

  std::vector<double> K(Krows * Kcols);
  std::vector<double> b(Krows, 0.0);

  for (int m = 0; m < M; ++m)
  for (int n = 0; n < M; ++n)
  {
    int row = m + n*M;
    double target = (m == n ? 1.0 : 0.0);
    b[row] = target;

    for (int p = 0; p < P; ++p)
    {
      double Vm = V[p + m*P];
      double Vn = V[p + n*P];
      K[row*Kcols + p] = Vm * Vn;
    }
  }

  // ---------------------------------------------------------------
  // Solve min ||K w - b||^2   via normal equations:
  //    (K^T K) w = K^T b
  // ---------------------------------------------------------------
  std::vector<double> KT_K(P * P, 0.0);
  std::vector<double> KT_b(P,     0.0);

  // Compute K^T K
  for (int i = 0; i < P; ++i)
  {
    for (int j = 0; j < P; ++j)
    {
      double sum = 0.0;
      for (int r = 0; r < Krows; ++r)
        sum += K[r*Kcols + i] * K[r*Kcols + j];
      KT_K[i*P + j] = sum;
    }
  }

  // Compute K^T b
  for (int i = 0; i < P; ++i)
  {
    double sum = 0.0;
    for (int r = 0; r < Krows; ++r)
      sum += K[r*Kcols + i] * b[r];
    KT_b[i] = sum;
  }

  // Solve (K^T K) w = K^T b
  std::vector<double> w(P, 0.0);
  bool ok = solve_cholesky(KT_K, w, KT_b, P);

  if (!ok)
  {
    std::cerr << "ERROR: Cholesky failed, system not SPD.\n";
    return 1;
  }

  // ---------------------------------------------------------------
  // Print weights
  // ---------------------------------------------------------------
  std::cout << "\nSolved weights w_p:\n";
  for (int p = 0; p < P; ++p)
    std::cout << "  w[" << p << "] = " << w[p] << "\n";

  // ---------------------------------------------------------------
  // GRAM CHECK: G = V^T W V  vs  I_M
  // ---------------------------------------------------------------

  std::vector<double> G(M * M, 0.0);

  for (int m = 0; m < M; ++m)
  {
    for (int n = 0; n < M; ++n)
    {
      double sum = 0.0;
      for (int p = 0; p < P; ++p)
      {
        double Vm = V[p + m*P];
        double Vn = V[p + n*P];
        sum += w[p] * Vm * Vn;
      }
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

  std::cout << "\nGRAM CHECK (G = V^T W V):\n";
  std::cout << "  Frobenius norm ||G - I||_F = " << frob << "\n";
  std::cout << "  max |offdiag(G - I)|        = " << max_abs_off << "\n";
  std::cout << "  diag range                  = ["
            << min_diag << ", " << max_diag << "]\n";

  return 0;
}

