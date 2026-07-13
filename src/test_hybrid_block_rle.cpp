#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <limits>
#include <cstdint>
#include <zlib.h>
#include <random>

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

// --------------------- Q loader for Jacobi ---------------------

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

// --------------------- Quantization helpers ---------------------

static inline int quantize_scalar(double c, double Qstep)
{
  return (int) std::llround(c / Qstep);
}

static inline double dequantize_scalar(int q, double Qstep)
{
  return (double)q * Qstep;
}

// --------------------- JPEG luminance quant table ---------------------

static const int kJpegLumaBase[8][8] = {
  {16, 11, 10, 16, 24, 40, 51, 61},
  {12, 12, 14, 19, 26, 58, 60, 55},
  {14, 13, 16, 24, 40, 57, 69, 56},
  {14, 17, 22, 29, 51, 87, 80, 62},
  {18, 22, 37, 56, 68,109,103, 77},
  {24, 35, 55, 64, 81,104,113, 92},
  {49, 64, 78, 87,103,121,120,101},
  {72, 92, 95, 98,112,100,103, 99}
};

static void build_jpeg_luma_quant_table(int quality, double Q[8][8])
{
  if (quality < 1) quality = 1;
  if (quality > 100) quality = 100;

  int scale;
  if (quality < 50)
    scale = 5000 / quality;
  else
    scale = 200 - quality * 2;

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      int q = kJpegLumaBase[i][j];
      q = (q * scale + 50) / 100;
      if (q < 1) q = 1;
      if (q > 255) q = 255;
      Q[i][j] = (double)q;
    }
  }
}

// --------------------- Jacobi quant table (graded lex) ---------------------

static int degree_from_index(int m, int p_max)
{
  for (int d = 0; d <= p_max; ++d)
  {
    int dim_d = (d + 1) * (d + 2) / 2;
    if (m < dim_d)
      return d;
  }
  return p_max;
}

static void build_jacobi_quant_table(int p_max,
                                     double base_step,
                                     double alpha,
                                     std::vector<double>& Q_jac)
{
  int M = (p_max + 1) * (p_max + 2) / 2;
  Q_jac.resize(M);

  for (int m = 0; m < M; ++m)
  {
    int deg = degree_from_index(m, p_max);
    double factor = 1.0 + alpha * (double)(deg * deg);
    Q_jac[m] = base_step * factor;
  }
}

// --------------------- zlib compression helper ---------------------

static size_t zlib_compress_int16(const std::vector<int16_t>& data)
{
  if (data.empty()) return 0;

  uLong src_len = (uLong)(data.size() * sizeof(int16_t));
  uLongf dst_len = compressBound(src_len);

  std::vector<unsigned char> dst(dst_len);

  int zres = compress2(dst.data(), &dst_len,
                       reinterpret_cast<const Bytef*>(data.data()),
                       src_len,
                       Z_BEST_COMPRESSION);

  if (zres != Z_OK)
  {
    std::cerr << "zlib compress2 error: " << zres << "\n";
    return src_len;
  }

  return (size_t)dst_len;
}

// --------------------- Simple zero-run RLE ---------------------

// Encode coeffs as pairs (run, value) in int16:
// for each nonzero value v, we push run_zeros, v.
// Trailing zeros are implied.
static void rle_encode_zeros(const std::vector<int16_t>& coeffs,
                             std::vector<int16_t>& out)
{
  out.clear();
  int run = 0;
  for (size_t i = 0; i < coeffs.size(); ++i)
  {
    int16_t c = coeffs[i];
    if (c == 0)
    {
      ++run;
    }
    else
    {
      out.push_back((int16_t)run);
      out.push_back(c);
      run = 0;
    }
  }
  // trailing zeros not stored
}

// --------------------- 8x8 zigzag order ---------------------

static const int kZigZag8x8[64] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63
};

// --------------------- Option A classifier: low-mode Jacobi MSE -----

static bool classifier_use_jacobi_optionA(const double pix[8][8],
                                          const QTriangle& tri0,
                                          const QTriangle& tri1,
                                          int M_low,
                                          double mse_threshold)
{
  int P0 = tri0.P;
  int P1 = tri1.P;
  int M  = tri0.M;
  if (M_low > M) M_low = M;

  std::vector<double> f0(P0), f1(P1);
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

  std::vector<double> c0_low(M_low), c1_low(M_low);

  for (int m = 0; m < M_low; ++m)
  {
    double sum0 = 0.0;
    for (int p = 0; p < P0; ++p)
      sum0 += f0[p] * tri0.Q[p + m * P0];
    c0_low[m] = sum0;

    double sum1 = 0.0;
    for (int p = 0; p < P1; ++p)
      sum1 += f1[p] * tri1.Q[p + m * P1];
    c1_low[m] = sum1;
  }

  std::vector<double> f0_low(P0), f1_low(P1);

  for (int p = 0; p < P0; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M_low; ++m)
      val += c0_low[m] * tri0.Q[p + m * P0];
    f0_low[p] = val;
  }
  for (int p = 0; p < P1; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M_low; ++m)
      val += c1_low[m] * tri1.Q[p + m * P1];
    f1_low[p] = val;
  }

  double recon[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      recon[i][j] = 0.0;

  for (int k = 0; k < P0; ++k)
  {
    int idx = tri0.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    recon[i][j] = f0_low[k];
  }
  for (int k = 0; k < P1; ++k)
  {
    int idx = tri1.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    recon[i][j] = f1_low[k];
  }

  double mse, psnr_dummy;
  compute_mse_psnr(pix, recon, mse, psnr_dummy);
  return (mse < mse_threshold);
}

// --------------------- Block type label ---------------------

static const char* block_type_name(int block_type)
{
  switch (block_type)
  {
    case 0: return "gradient";
    case 1: return "diagonal step";
    case 2: return "high-frequency sinusoid";
    case 3: return "horizontal step edge";
    case 4: return "vertical step edge";
    case 5: return "corner (quadrant split)";
    case 6: return "slow cosine variation";
    case 7: return "medium texture";
    case 8: return "random noise";
    case 9: return "checkerboard";
    default: return "unknown";
  }
}

// --------------------- main ---------------------

int main()
{
  // Load Jacobi Qs once
  QTriangle tri0, tri1;
  std::string file0 = "jacobi_Q_tri0_p6.txt";
  std::string file1 = "jacobi_Q_tri1_p6.txt";

  if (!load_Q_triangle(file0, tri0)) return 1;
  if (!load_Q_triangle(file1, tri1)) return 1;

  if (tri0.p_max != tri1.p_max || tri0.M != tri1.M)
  {
    std::cerr << "ERROR: triangle 0 and 1 Q mismatch.\n";
    return 1;
  }

  int p_max = tri0.p_max;
  int M = tri0.M;
  int P0 = tri0.P;
  int P1 = tri1.P;

  std::cout << "Loaded Jacobi: p_max=" << p_max
            << " M=" << M << " P0=" << P0 << " P1=" << P1 << "\n\n";

  // DCT quant table (JPEG-like)
  int jpeg_quality = 50;
  double Q_luma[8][8];
  build_jpeg_luma_quant_table(jpeg_quality, Q_luma);

  // Jacobi base steps to sweep
  std::vector<double> jac_base_steps = {1.0, 2.0, 4.0, 8.0, 16.0};

  // Classifier parameters
  int    M_low       = 2;     // use first 2 modes (deg 0,1) per triangle
  double mse_thresh  = 20.0;  // Jacobi if low-order approx is good

  // RNG for noise block
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> noise_dist(0.0, 255.0);

  const double PI = 3.14159265358979323846;

  // Outer loop over block types
  for (int block_type = 0; block_type <= 9; ++block_type)
  {
    double pix[8][8];

    // Build block
    if (block_type == 0)
    {
      // gradient
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = (double)i + 0.5 * (double)j;
    }
    else if (block_type == 1)
    {
      // diagonal step
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = (i + j < 8 ? 50.0 : 200.0);
    }
    else if (block_type == 2)
    {
      // high-frequency sinusoid
      for (int i = 0; i < 8; ++i)
      {
        for (int j = 0; j < 8; ++j)
        {
          double val = 128.0
            + 50.0 * std::sin(6.0 * PI * (double)i / 8.0)
                   * std::sin(6.0 * PI * (double)j / 8.0);
          pix[i][j] = val;
        }
      }
    }
    else if (block_type == 3)
    {
      // horizontal step edge (top vs bottom)
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = (i < 4 ? 50.0 : 200.0);
    }
    else if (block_type == 4)
    {
      // vertical step edge (left vs right)
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = (j < 4 ? 50.0 : 200.0);
    }
    else if (block_type == 5)
    {
      // corner (quadrant split): top-left vs others
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = (i < 4 && j < 4 ? 50.0 : 200.0);
    }
    else if (block_type == 6)
    {
      // slow cosine variation
      for (int i = 0; i < 8; ++i)
      {
        for (int j = 0; j < 8; ++j)
        {
          double val = 128.0
            + 40.0 * std::cos(PI * (double)i / 8.0)
                   * std::cos(PI * (double)j / 8.0);
          pix[i][j] = val;
        }
      }
    }
    else if (block_type == 7)
    {
      // medium texture (lower frequency sinusoid)
      for (int i = 0; i < 8; ++i)
      {
        for (int j = 0; j < 8; ++j)
        {
          double val = 128.0
            + 30.0 * std::sin(2.0 * PI * (double)i / 8.0)
                   * std::sin(3.0 * PI * (double)j / 8.0);
          pix[i][j] = val;
        }
      }
    }
    else if (block_type == 8)
    {
      // random noise
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix[i][j] = noise_dist(rng);
    }
    else if (block_type == 9)
    {
      // checkerboard
      for (int i = 0; i < 8; ++i)
      {
        for (int j = 0; j < 8; ++j)
        {
          bool black = ((i + j) & 1) != 0;
          pix[i][j] = black ? 30.0 : 220.0;
        }
      }
    }

    std::cout << "============================================================\n";
    std::cout << "Block type " << block_type << " (" << block_type_name(block_type) << ")\n";

    // Full DCT
    double dct[8][8];
    dct2_8x8(pix, dct);

    // Full Jacobi coeffs
    std::vector<double> f0(P0), f1(P1);
    std::vector<double> c0(M),  c1(M);

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

    for (int m = 0; m < M; ++m)
    {
      double sum0 = 0.0;
      for (int p = 0; p < P0; ++p)
        sum0 += f0[p] * tri0.Q[p + m * P0];
      c0[m] = sum0;

      double sum1 = 0.0;
      for (int p = 0; p < P1; ++p)
        sum1 += f1[p] * tri1.Q[p + m * P1];
      c1[m] = sum1;
    }

    // Classifier
    bool use_jacobi = classifier_use_jacobi_optionA(pix, tri0, tri1,
                                                    M_low, mse_thresh);
    std::cout << "Classifier decision (Option A, M_low=" << M_low
              << ", mse_thresh=" << mse_thresh << "): "
              << (use_jacobi ? "Jacobi" : "DCT") << "\n\n";

    std::cout << "=== Hybrid block experiment with RLE + zlib ===\n";
    std::cout << "Columns: base_step, DCT_bpp, DCT_PSNR, Jac_bpp, Jac_PSNR\n\n";

    for (double jac_base_step : jac_base_steps)
    {
      // ---------- DCT path ----------
      double dct_q[8][8];
      for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v)
        {
          double step = Q_luma[u][v];
          int q = quantize_scalar(dct[u][v], step);
          dct_q[u][v] = dequantize_scalar(q, step);
        }

      double pix_dct_rec[8][8];
      idct2_8x8(dct_q, pix_dct_rec);

      double mse_dct, psnr_dct;
      compute_mse_psnr(pix, pix_dct_rec, mse_dct, psnr_dct);

      std::vector<int16_t> dct_coeffs_zz(64);
      for (int k = 0; k < 64; ++k)
      {
        int idx = kZigZag8x8[k];
        int u = idx / 8;
        int v = idx % 8;

        double step = Q_luma[u][v];
        int q = quantize_scalar(dct[u][v], step);
        dct_coeffs_zz[k] = (int16_t)q;
      }

      std::vector<int16_t> dct_rle;
      rle_encode_zeros(dct_coeffs_zz, dct_rle);

      size_t dct_bytes = zlib_compress_int16(dct_rle);
      double dct_bpp = (double)(dct_bytes * 8) / 64.0;

      // ---------- Jacobi path ----------
      std::vector<double> Q_jac;
      build_jacobi_quant_table(p_max, jac_base_step, 0.3, Q_jac);

      std::vector<int16_t> q_jac_vec(2 * M);
      std::vector<double> c0_q(M), c1_q(M);

      for (int m = 0; m < M; ++m)
      {
        double step_m = Q_jac[m];

        int q0 = quantize_scalar(c0[m], step_m);
        int q1 = quantize_scalar(c1[m], step_m);

        q_jac_vec[2*m + 0] = (int16_t)q0;
        q_jac_vec[2*m + 1] = (int16_t)q1;

        c0_q[m] = dequantize_scalar(q0, step_m);
        c1_q[m] = dequantize_scalar(q1, step_m);
      }

      std::vector<double> f0_q(P0), f1_q(P1);
      for (int p = 0; p < P0; ++p)
      {
        double val = 0.0;
        for (int m = 0; m < M; ++m)
          val += c0_q[m] * tri0.Q[p + m * P0];
        f0_q[p] = val;
      }
      for (int p = 0; p < P1; ++p)
      {
        double val = 0.0;
        for (int m = 0; m < M; ++m)
          val += c1_q[m] * tri1.Q[p + m * P1];
        f1_q[p] = val;
      }

      double pix_jac_rec[8][8];
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pix_jac_rec[i][j] = 0.0;

      for (int k = 0; k < P0; ++k)
      {
        int idxp = tri0.pix_idx[k];
        int i = idxp / 8;
        int j = idxp % 8;
        pix_jac_rec[i][j] = f0_q[k];
      }
      for (int k = 0; k < P1; ++k)
      {
        int idxp = tri1.pix_idx[k];
        int i = idxp / 8;
        int j = idxp % 8;
        pix_jac_rec[i][j] = f1_q[k];
      }

      double mse_jac, psnr_jac;
      compute_mse_psnr(pix, pix_jac_rec, mse_jac, psnr_jac);

      std::vector<int16_t> jac_rle;
      rle_encode_zeros(q_jac_vec, jac_rle);

      size_t jac_bytes = zlib_compress_int16(jac_rle);
      double jac_bpp = (double)(jac_bytes * 8) / 64.0;

      std::cout << "base_step=" << jac_base_step
                << "  DCT: bpp=" << dct_bpp
                << " PSNR=" << psnr_dct
                << "   Jac: bpp=" << jac_bpp
                << " PSNR=" << psnr_jac << " dB\n";
    }

    std::cout << "\n";
  }

  return 0;
}
