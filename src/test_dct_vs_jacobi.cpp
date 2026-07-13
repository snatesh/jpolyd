#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <limits>

#include <jbasis.hh>
#include <jdetail.hh>

using namespace jsimplex;

// --------------------- DCT (8x8, orthonormal) ---------------------

static void dct2_8x8(const double in[8][8], double out[8][8])
{
  const int N = 8;
  const double PI = 3.14159265358979323846;

  auto alpha = [](int k) -> double {
    return (k == 0) ? std::sqrt(0.5) : 1.0;
  };

  for (int u = 0; u < N; ++u)
  {
    for (int v = 0; v < N; ++v)
    {
      double sum = 0.0;
      for (int x = 0; x < N; ++x)
      {
        for (int y = 0; y < N; ++y)
        {
          double cx = std::cos(PI * (2.0 * x + 1.0) * u / (2.0 * N));
          double cy = std::cos(PI * (2.0 * y + 1.0) * v / (2.0 * N));
          sum += in[x][y] * cx * cy;
        }
      }
      out[u][v] = 0.25 * alpha(u) * alpha(v) * sum;
    }
  }
}

// Inverse DCT (with same orthonormal scaling)
static void idct2_8x8(const double in[8][8], double out[8][8])
{
  const int N = 8;
  const double PI = 3.14159265358979323846;

  auto alpha = [](int k) -> double {
    return (k == 0) ? std::sqrt(0.5) : 1.0;
  };

  for (int x = 0; x < N; ++x)
  {
    for (int y = 0; y < N; ++y)
    {
      double sum = 0.0;
      for (int u = 0; u < N; ++u)
      {
        for (int v = 0; v < N; ++v)
        {
          double cx = std::cos(PI * (2.0 * x + 1.0) * u / (2.0 * N));
          double cy = std::cos(PI * (2.0 * y + 1.0) * v / (2.0 * N));
          sum += alpha(u) * alpha(v) * in[u][v] * cx * cy;
        }
      }
      out[x][y] = 0.25 * sum;
    }
  }
}

static double dct_energy_raw(const double coeff[8][8])
{
  double E = 0.0;
  for (int u = 0; u < 8; ++u)
    for (int v = 0; v < 8; ++v)
      E += coeff[u][v] * coeff[u][v];
  return E;
}

// ----------------- Piecewise-constant sampling ---------------------
//
// We represent the pixel block as a piecewise-constant function on
// an 8x8 grid of cells covering [0,1]^2.
// Cell (i,j) = [i/8,(i+1)/8) x [j/8,(j+1)/8), value = pix[i][j].
//
// For a quadrature point (x,y), we just pick the pixel whose cell
// contains that point.

static double piecewise_constant_8x8(const double pix[8][8],
                                     double x, double y)
{
  if (x < 0.0) x = 0.0;
  if (y < 0.0) y = 0.0;
  if (x >= 1.0) x = std::nextafter(1.0, 0.0);
  if (y >= 1.0) y = std::nextafter(1.0, 0.0);

  double gx = x * 8.0;
  double gy = y * 8.0;

  int i = static_cast<int>(std::floor(gx)); // 0..7
  int j = static_cast<int>(std::floor(gy)); // 0..7

  if (i < 0) i = 0;
  if (j < 0) j = 0;
  if (i > 7) i = 7;
  if (j > 7) j = 7;

  return pix[i][j];
}

// ----------------- Triangle mapping helpers -------------------------
//
// Reference triangle T_hat: (0,0), (1,0), (0,1).
// Square [0,1]^2 split along anti-diagonal (0,1) -- (1,0).
//
// Tri 0: (0,0), (1,0), (0,1)  -- matches reference triangle.
// Tri 1: (1,1), (0,1), (1,0).

struct TriangleMap
{
  double x0[2];
  double x1[2];
  double x2[2];
  double Ainv[2][2]; // inverse of [v1-v0 v2-v0]

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

    double det = a11 * a22 - a12 * a21;
    assert(std::fabs(det) > 1e-14);
    double invdet = 1.0 / det;

    Ainv[0][0] =  a22 * invdet;
    Ainv[0][1] = -a12 * invdet;
    Ainv[1][0] = -a21 * invdet;
    Ainv[1][1] =  a11 * invdet;
  }

  inline void ref_to_phys(double xi, double eta, double& x, double& y) const
  {
    x = x0[0] + xi * (x1[0] - x0[0]) + eta * (x2[0] - x0[0]);
    y = x0[1] + xi * (x1[1] - x0[1]) + eta * (x2[1] - x0[1]);
  }

  inline void phys_to_ref(double x, double y, double& xi, double& eta) const
  {
    double dx = x - x0[0];
    double dy = y - x0[1];
    xi  = Ainv[0][0] * dx + Ainv[0][1] * dy;
    eta = Ainv[1][0] * dx + Ainv[1][1] * dy;
  }
};

// -------------- Loader for flattened quad file ----------------------
//
// Python saved: np.savetxt(fname, np.concatenate([X.reshape(-1), w]))
// For dimension D, file is:
//   [ x0_0, ..., x0_{D-1},
//     x1_0, ..., x1_{D-1},
//     ...
//     x_{N-1,0},...,x_{N-1,D-1},
//     w0, ..., w_{N-1} ]

bool load_quad_rule_flat(const std::string& filename,
                         int D,
                         std::vector<double>& X,   // size N*D
                         std::vector<double>& w)   // size N
{
  std::ifstream in(filename);
  if (!in) return false;

  std::vector<double> z;
  double val;
  while (in >> val)
    z.push_back(val);

  if (z.empty()) return false;

  int total = static_cast<int>(z.size());
  if (total % (D + 1) != 0)
  {
    std::cerr << "Quadrature file malformed: total=" << total
              << " not divisible by D+1=" << (D+1) << "\n";
    return false;
  }

  int N = total / (D + 1);
  X.resize(N * D);
  w.resize(N);

  for (int i = 0; i < N * D; ++i)
    X[i] = z[i];

  for (int i = 0; i < N; ++i)
    w[i] = z[N * D + i];

  return true;
}

// -------- Jacobi forward transform on a single triangle -------------
//
// V layout from eval_all:
//   V[p + m*ld_V] = φ_m(x_p),  p = 0..N-1,  m = 0..Km-1,
//   ld_V is leading dimension in p (we set ld_V=N).
//
// So V is "column-major" in mode index m.

static void triangle_jacobi_coeffs(
    const double pix[8][8],
    const std::vector<double>& Xref,   // size N*2
    const std::vector<double>& wref,   // size N
    const std::vector<double>& V,      // size N*Km, col-major in m
    int p_deg,
    const TriangleMap& triMap,
    std::vector<double>& coeffs)
{
  const int D = 2;
  const int N = static_cast<int>(wref.size());
  const int Km = jsimplex::Basis<D,double>::dim_Pi(p_deg);

  coeffs.assign(Km, 0.0);

  for (int p = 0; p < N; ++p)
  {
    double xi  = Xref[D*p + 0];
    double eta = Xref[D*p + 1];

    double x, y;
    triMap.ref_to_phys(xi, eta, x, y);

    double fval = piecewise_constant_8x8(pix, x, y);
    double wi   = wref[p];

    for (int m = 0; m < Km; ++m)
    {
      double phi = V[p + m * N];  // ld_V = N
      coeffs[m] += wi * fval * phi;
    }
  }
}

// ----------------- PSNR helper -------------------------------------

static void compute_mse_psnr(const double orig[8][8],
                             const double recon[8][8],
                             double& mse,
                             double& psnr,
                             double max_val = 255.0)
{
  mse = 0.0;
  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      double diff = orig[i][j] - recon[i][j];
      mse += diff * diff;
    }
  }
  mse /= 64.0;

  if (mse <= 0.0)
  {
    psnr = std::numeric_limits<double>::infinity();
  }
  else
  {
    psnr = 10.0 * std::log10((max_val * max_val) / mse);
  }
}

// --------------------------- main test ------------------------------

int main()
{
  // 1) Simple 8x8 test block (replace with real image block if you like)
  double pix[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      pix[i][j] = static_cast<double>(i) + 0.5 * static_cast<double>(j);

  // 2) Forward DCT
  double dct[8][8];
  dct2_8x8(pix, dct);
  double E_dct_raw  = dct_energy_raw(dct);
  double E_dct_cont = E_dct_raw / 64.0; // piecewise-constant L2^2 over [0,1]^2

  std::cout << "DCT energy (sum c^2)............: " << E_dct_raw  << "\n";
  std::cout << "DCT L2^2 over square (pc model).: " << E_dct_cont << "\n";

  // 3) Inverse DCT and PSNR
  double pix_dct_rec[8][8];
  idct2_8x8(dct, pix_dct_rec);

  double mse_dct, psnr_dct;
  compute_mse_psnr(pix, pix_dct_rec, mse_dct, psnr_dct);
  std::cout << "DCT recon MSE....................: " << mse_dct  << "\n";
  std::cout << "DCT recon PSNR (max=255).........: " << psnr_dct << " dB\n";

  // 4) Load quadrature rule on reference triangle
  std::string quad_file = "t2_kappa0.5-0.5-0.5_N45_M120_n8_m14.txt";
  std::vector<double> Xref;  // (ξ,η)
  std::vector<double> wref;

  if (!load_quad_rule_flat(quad_file, 2, Xref, wref))
  {
    std::cerr << "ERROR: could not load quadrature rule from "
              << quad_file << "\n";
    return 1;
  }

  int N = static_cast<int>(wref.size());
  std::cout << "Loaded quadrature with N=" << N << " points.\n";

  double sum_w = 0.0;
  for (double wi : wref) sum_w += wi;
  std::cout << "Sum of quadrature weights.........: " << sum_w << "\n";

  // 5) Build Jacobi basis metadata for degree p=7
  const int D = 2;
  int p_deg = 7;
  int Km = jsimplex::Basis<D,double>::dim_Pi(p_deg);

  std::vector<int>    alpha_table(Km * D);
  std::vector<int>    tail_deg   (Km * D);
  std::vector<double> inv_h      (Km);

  double kappa[3] = {0.5, 0.5, 0.5};

  jsimplex::Basis<D,double>::build_structures(
    kappa,
    p_deg,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data()
  );

  // 6) Precompute V at reference quad nodes: V_quad[p + m*ld_V], ld_V = N
  std::vector<double> V_quad(N * Km);
  {
    std::vector<double> X(N * D);
    for (int p = 0; p < N; ++p)
    {
      X[D*p + 0] = Xref[D*p + 0];
      X[D*p + 1] = Xref[D*p + 1];
    }

    jsimplex::Basis<D,double>::eval_all(
      X.data(),
      D,    // ld_point
      1,    // ld_dim
      N,
      kappa,
      p_deg,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data(),
      V_quad.data(),
      N,    // ld_V = N
      nullptr
    );
  }

  // 7) Triangle mappings for square [0,1]^2
  TriangleMap tri0;
  {
    double v0[2] = {0.0, 0.0};
    double v1[2] = {1.0, 0.0};
    double v2[2] = {0.0, 1.0};
    tri0.build(v0, v1, v2);
  }

  TriangleMap tri1;
  {
    double v0[2] = {1.0, 1.0};
    double v1[2] = {0.0, 1.0};
    double v2[2] = {1.0, 0.0};
    tri1.build(v0, v1, v2);
  }

  // 8) Jacobi coefficients per triangle (piecewise-constant sampling)
  std::vector<double> c0, c1;
  triangle_jacobi_coeffs(pix, Xref, wref, V_quad, p_deg, tri0, c0);
  triangle_jacobi_coeffs(pix, Xref, wref, V_quad, p_deg, tri1, c1);

  // 9) Energy in Jacobi coefficients (two triangles).
  double E_tri = 0.0;
  for (int m = 0; m < Km; ++m)
  {
    E_tri += c0[m] * c0[m];
    E_tri += c1[m] * c1[m];
  }

  // Divide by 2 to get L2^2 over the square (since each triangle's
  // quadrature measure has mass ~1 instead of 1/2).
  double E_tri_square = 0.5 * E_tri;

  std::cout << "Jacobi coeff L2^2 (two tris, raw) : " << E_tri << "\n";
  std::cout << "Jacobi L2^2 over square (correct): " << E_tri_square << "\n";

  // 10) Reconstruction from Jacobi to pixels using V_pix (basis at pixel locations)
  //
  // Here is the important part for your question:
  // We reconstruct AT THE PIXELS (same 8x8 indices) by evaluating the
  // Jacobi basis at some chosen (x,y) for each pixel. We choose the
  // cell center (i+0.5)/8, (j+0.5)/8, but that's just a consistent
  // embedding; the comparison is pixel-by-pixel orig vs recon.

  // Precompute reference coords for pixels belonging to each triangle
  std::vector<int> pix_idx_tri0;
  std::vector<int> pix_idx_tri1;
  std::vector<double> X_pix_tri0; // (xi,eta) pairs
  std::vector<double> X_pix_tri1;

  X_pix_tri0.reserve(2 * 64);
  X_pix_tri1.reserve(2 * 64);

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      double x = (i + 0.5) / 8.0;
      double y = (j + 0.5) / 8.0;

      double xi, eta;
      if (x + y <= 1.0)
      {
        tri0.phys_to_ref(x, y, xi, eta);
        pix_idx_tri0.push_back(i*8 + j);
        X_pix_tri0.push_back(xi);
        X_pix_tri0.push_back(eta);
      }
      else
      {
        tri1.phys_to_ref(x, y, xi, eta);
        pix_idx_tri1.push_back(i*8 + j);
        X_pix_tri1.push_back(xi);
        X_pix_tri1.push_back(eta);
      }
    }
  }

  int Npix0 = static_cast<int>(pix_idx_tri0.size());
  int Npix1 = static_cast<int>(pix_idx_tri1.size());

  // Evaluate basis at pixel reference coords for each triangle
  std::vector<double> V_pix0(Npix0 * Km);
  std::vector<double> V_pix1(Npix1 * Km);

  if (Npix0 > 0)
  {
    jsimplex::Basis<D,double>::eval_all(
      X_pix_tri0.data(),
      /*ld_point=*/2,
      /*ld_dim=*/1,
      /*npts=*/Npix0,
      kappa,
      p_deg,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data(),
      V_pix0.data(),
      /*ld_V=*/Npix0,
      nullptr
    );
  }

  if (Npix1 > 0)
  {
    jsimplex::Basis<D,double>::eval_all(
      X_pix_tri1.data(),
      /*ld_point=*/2,
      /*ld_dim=*/1,
      /*npts=*/Npix1,
      kappa,
      p_deg,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data(),
      V_pix1.data(),
      /*ld_V=*/Npix1,
      nullptr
    );
  }

  // Reconstruct pixel values from coefficients:
  double pix_jacobi_rec[8][8];
  // Initialize
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      pix_jacobi_rec[i][j] = 0.0;

  // Triangle 0 pixels
  for (int p = 0; p < Npix0; ++p)
  {
    int idx = pix_idx_tri0[p];
    int i = idx / 8;
    int j = idx % 8;

    double val = 0.0;
    for (int m = 0; m < Km; ++m)
    {
      double phi = V_pix0[p + m * Npix0]; // ld_V = Npix0
      val += c0[m] * phi;
    }
    pix_jacobi_rec[i][j] = val;
  }

  // Triangle 1 pixels
  for (int p = 0; p < Npix1; ++p)
  {
    int idx = pix_idx_tri1[p];
    int i = idx / 8;
    int j = idx % 8;

    double val = 0.0;
    for (int m = 0; m < Km; ++m)
    {
      double phi = V_pix1[p + m * Npix1]; // ld_V = Npix1
      val += c1[m] * phi;
    }
    pix_jacobi_rec[i][j] = val;
  }

  double mse_jacobi, psnr_jacobi;
  compute_mse_psnr(pix, pix_jacobi_rec, mse_jacobi, psnr_jacobi);

  std::cout << "Jacobi recon MSE..................: " << mse_jacobi  << "\n";
  std::cout << "Jacobi recon PSNR (max=255).......: " << psnr_jacobi << " dB\n";

  return 0;
}
