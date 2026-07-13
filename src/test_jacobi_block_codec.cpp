#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <limits>

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

// Truncate DCT by keeping only coefficients with u+v <= max_freq_sum
static void dct_truncate(const double in[8][8],
                         double out[8][8],
                         int max_freq_sum,
                         int &num_kept)
{
  num_kept = 0;
  for (int u = 0; u < 8; ++u)
  {
    for (int v = 0; v < 8; ++v)
    {
      if (u + v <= max_freq_sum)
      {
        out[u][v] = in[u][v];
        ++num_kept;
      }
      else
      {
        out[u][v] = 0.0;
      }
    }
  }
}

// --------------------- MSE / PSNR ---------------------

static void compute_mse_psnr(const double orig[8][8],
                             const double recon[8][8],
                             double &mse,
                             double &psnr,
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

// --------------------- Q loader ---------------------

struct QTriangle
{
  int p_max;
  int M;      // number of modes
  int P;      // number of pixels in this triangle
  std::vector<int> pix_idx; // global pixel indices (i*8 + j)
  std::vector<double> Q;    // size P*M, column-major: Q[p + m*P]
};

static bool load_Q_triangle(const std::string &filename, QTriangle &qt)
{
  std::ifstream in(filename.c_str());
  if (!in)
  {
    std::cerr << "ERROR: could not open " << filename << " for reading.\n";
    return false;
  }

  int p_max, M, P;
  if (!(in >> p_max >> M >> P))
  {
    std::cerr << "ERROR: failed to read header from " << filename << "\n";
    return false;
  }

  qt.p_max = p_max;
  qt.M = M;
  qt.P = P;

  qt.pix_idx.resize(P);
  for (int p = 0; p < P; ++p)
  {
    if (!(in >> qt.pix_idx[p]))
    {
      std::cerr << "ERROR: failed to read pix_idx from " << filename << "\n";
      return false;
    }
  }

  qt.Q.assign(P * M, 0.0);

  // Each of next P lines has M doubles: row-major Q[p,0..M-1]
  for (int p = 0; p < P; ++p)
  {
    for (int m = 0; m < M; ++m)
    {
      double val;
      if (!(in >> val))
      {
        std::cerr << "ERROR: failed to read Q row data from " << filename << "\n";
        return false;
      }
      // store in column-major
      qt.Q[p + m * P] = val;
    }
  }

  return true;
}

// --------------------- main ---------------------

int main()
{
  // 1) Build a test 8x8 pixel block
  double pix[8][8];
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      // simple synthetic block: gradient
      //pix[i][j] = static_cast<double>(i) + 0.5 * static_cast<double>(j);
      // diagonal edge
      //pix[i][j] = (i + j < 8 ? 50 : 200);
      //high frequency
      pix[i][j] = 128 + 50*sin(6*pi*i/8)*sin(6*pi*j/8);

    }
  }

  // 2) DCT forward / inverse (no truncation)
  double dct[8][8];
  dct2_8x8(pix, dct);

  double pix_dct_rec[8][8];
  idct2_8x8(dct, pix_dct_rec);

  double mse_dct, psnr_dct;
  compute_mse_psnr(pix, pix_dct_rec, mse_dct, psnr_dct);

  double E_dct = 0.0;
  for (int u = 0; u < 8; ++u)
    for (int v = 0; v < 8; ++v)
      E_dct += dct[u][v] * dct[u][v];

  std::cout << "DCT energy (sum c^2).............: " << E_dct << "\n";
  std::cout << "DCT recon MSE.....................: " << mse_dct << "\n";
  std::cout << "DCT recon PSNR (max=255)..........: " << psnr_dct << " dB\n";

  // 3) Load Q for both triangles
  QTriangle tri0, tri1;
  std::string file0 = "jacobi_Q_tri0_p6.txt";
  std::string file1 = "jacobi_Q_tri1_p6.txt";

  if (!load_Q_triangle(file0, tri0)) return 1;
  if (!load_Q_triangle(file1, tri1)) return 1;

  std::cout << "\nLoaded triangle 0 Q from " << file0 << "\n";
  std::cout << "  p_max=" << tri0.p_max
            << " M=" << tri0.M
            << " P=" << tri0.P << "\n";

  std::cout << "Loaded triangle 1 Q from " << file1 << "\n";
  std::cout << "  p_max=" << tri1.p_max
            << " M=" << tri1.M
            << " P=" << tri1.P << "\n";

  if (tri0.p_max != tri1.p_max || tri0.M != tri1.M)
  {
    std::cerr << "ERROR: triangle 0 and 1 Q have mismatched p_max or M.\n";
    return 1;
  }

  int p_max = tri0.p_max;
  int M = tri0.M;
  int P0 = tri0.P;
  int P1 = tri1.P;

  std::cout << "Using Jacobi p_max=" << p_max
            << ", M=" << M
            << ", P0=" << P0
            << ", P1=" << P1 << "\n";

  // 4) Jacobi encode/decode using Q0 and Q1 (no truncation)

  std::vector<double> f0(P0), f1(P1);
  std::vector<double> c0(M),  c1(M);
  std::vector<double> f0_rec(P0), f1_rec(P1);

  // Build f0, f1 from the 8x8 pixel block
  for (int k = 0; k < P0; ++k)
  {
    int idx = tri0.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    f0[k] = pix[i][j];
  }
  for (int k = 0; k < P1; ++k)
  {
    int idx = tri1.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    f1[k] = pix[i][j];
  }

  // Encode full Jacobi coeffs
  for (int m = 0; m < M; ++m)
  {
    double sum0 = 0.0;
    for (int p = 0; p < P0; ++p)
      sum0 += f0[p] * tri0.Q[p + m * P0];
    c0[m] = sum0;
  }
  for (int m = 0; m < M; ++m)
  {
    double sum1 = 0.0;
    for (int p = 0; p < P1; ++p)
      sum1 += f1[p] * tri1.Q[p + m * P1];
    c1[m] = sum1;
  }

  // Decode full Jacobi coeffs
  for (int p = 0; p < P0; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M; ++m)
      val += c0[m] * tri0.Q[p + m * P0];
    f0_rec[p] = val;
  }
  for (int p = 0; p < P1; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M; ++m)
      val += c1[m] * tri1.Q[p + m * P1];
    f1_rec[p] = val;
  }

  // Scatter Jacobi reconstruction back to 8x8 grid
  double pix_jacobi_rec[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      pix_jacobi_rec[i][j] = 0.0;

  for (int k = 0; k < P0; ++k)
  {
    int idx = tri0.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    pix_jacobi_rec[i][j] = f0_rec[k];
  }
  for (int k = 0; k < P1; ++k)
  {
    int idx = tri1.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    pix_jacobi_rec[i][j] = f1_rec[k];
  }

  double mse_jacobi, psnr_jacobi;
  compute_mse_psnr(pix, pix_jacobi_rec, mse_jacobi, psnr_jacobi);

  double E_jacobi = 0.0;
  for (int m = 0; m < M; ++m)
  {
    E_jacobi += c0[m] * c0[m];
    E_jacobi += c1[m] * c1[m];
  }

  std::cout << "\nJacobi coeff energy (sum over both tris): " << E_jacobi << "\n";
  std::cout << "Jacobi recon MSE (no truncation)........: " << mse_jacobi << "\n";
  std::cout << "Jacobi recon PSNR (max=255).............: " << psnr_jacobi << " dB\n";

  // ==============================================================
  // 5) Truncation experiments
  // ==============================================================

  std::cout << "\n=== DCT truncation (u+v <= K) ===\n";
  for (int K = 0; K <= 10; ++K)
  {
    double dct_trunc[8][8];
    int num_kept = 0;
    dct_truncate(dct, dct_trunc, K, num_kept);

    double pix_dct_trunc_rec[8][8];
    idct2_8x8(dct_trunc, pix_dct_trunc_rec);

    double mse, psnr;
    compute_mse_psnr(pix, pix_dct_trunc_rec, mse, psnr);

    std::cout << "K = " << K
              << "  kept = " << num_kept
              << "  MSE = " << mse
              << "  PSNR = " << psnr << " dB\n";
  }

  std::cout << "\n=== Jacobi truncation (keep first M_keep modes per triangle) ===\n";

  for (int M_keep = 1; M_keep <= M; ++M_keep)
  {
    // reconstruct with only first M_keep modes
    std::vector<double> f0_tr(P0), f1_tr(P1);

    for (int p = 0; p < P0; ++p)
    {
      double val = 0.0;
      for (int m = 0; m < M_keep; ++m)
        val += c0[m] * tri0.Q[p + m * P0];
      f0_tr[p] = val;
    }
    for (int p = 0; p < P1; ++p)
    {
      double val = 0.0;
      for (int m = 0; m < M_keep; ++m)
        val += c1[m] * tri1.Q[p + m * P1];
      f1_tr[p] = val;
    }

    double pix_jacobi_tr[8][8];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        pix_jacobi_tr[i][j] = 0.0;

    for (int k = 0; k < P0; ++k)
    {
      int idx = tri0.pix_idx[k];
      int i = idx / 8;
      int j = idx % 8;
      pix_jacobi_tr[i][j] = f0_tr[k];
    }
    for (int k = 0; k < P1; ++k)
    {
      int idx = tri1.pix_idx[k];
      int i = idx / 8;
      int j = idx % 8;
      pix_jacobi_tr[i][j] = f1_tr[k];
    }

    double mse, psnr;
    compute_mse_psnr(pix, pix_jacobi_tr, mse, psnr);

    std::cout << "M_keep = " << M_keep
              << "  total coeffs (2*M_keep) = " << 2*M_keep
              << "  MSE = " << mse
              << "  PSNR = " << psnr << " dB\n";
  }

  return 0;
}
