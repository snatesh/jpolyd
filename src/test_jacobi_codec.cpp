#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <limits>
#include <cstdint>
#include <zlib.h>
#include <random>

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
                                     std::vector<double> &Q_jac)
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

static size_t zlib_compress_int16(const std::vector<int16_t> &data)
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
//
// Encode coeffs as pairs (run, value) in int16:
// for each nonzero value v, we push run_zeros, v.
// Trailing zeros are implied.
//
static void rle_encode_zeros(const std::vector<int16_t> &coeffs,
                             std::vector<int16_t> &out)
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

// --------------------- Block type label ---------------------

static const char *block_type_name(int block_type)
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

// --------------------- Synthetic block generator ---------------------

static void make_block(int block_type, double pix[8][8],
                       std::mt19937 &rng)
{
  const double PI = 3.14159265358979323846;
  std::uniform_real_distribution<double> noise_dist(0.0, 255.0);

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
        pix[i][j] = 128.0
          + 50.0 * std::sin(6.0 * PI * i / 8.0)
                  * std::sin(6.0 * PI * j / 8.0);
      }
    }
  }
  else if (block_type == 3)
  {
    // horizontal step edge
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        pix[i][j] = (i < 4 ? 50.0 : 200.0);
  }
  else if (block_type == 4)
  {
    // vertical step edge
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        pix[i][j] = (j < 4 ? 50.0 : 200.0);
  }
  else if (block_type == 5)
  {
    // corner (quadrant split)
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
        pix[i][j] = 128.0
          + 40.0 * std::cos(PI * i / 8.0)
                  * std::cos(PI * j / 8.0);
      }
    }
  }
  else if (block_type == 7)
  {
    // medium texture
    for (int i = 0; i < 8; ++i)
    {
      for (int j = 0; j < 8; ++j)
      {
        pix[i][j] = 128.0
          + 30.0 * std::sin(2.0 * PI * i / 8.0)
                  * std::sin(3.0 * PI * j / 8.0);
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
}

// --------------------- main: pure Jacobi encoder experiment ---------------------

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

  // Jacobi "quality" knobs
  std::vector<double> jac_base_steps = {1.0, 2.0, 4.0, 8.0, 16.0};
  std::vector<int> jac_deg_list = {0, 1, 2, 3, 6};
  double alpha = 0.3;  // curvature in quant table

  std::mt19937 rng(12345);

  // Loop over block types
  for (int block_type = 0; block_type <= 9; ++block_type)
  {
    double pix[8][8];
    make_block(block_type, pix, rng);

    std::cout << "============================================================\n";
    std::cout << "Block type " << block_type
              << " (" << block_type_name(block_type) << ")\n\n";

    // Precompute triangle-wise samples for this block
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

    // Compute full Jacobi coefficients c0, c1
    std::vector<double> c0(M), c1(M);
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

    // Loop over truncation degrees
    for (int jac_deg_keep : jac_deg_list)
    {
      std::cout << "=== Jacobi degree truncation: deg <= " << jac_deg_keep << " ===\n";
      std::cout << "Columns: base_step, Jac_bpp, Jac_PSNR\n";

      // Loop over quality-like base_step
      for (double jac_base_step : jac_base_steps)
      {
        // Build quant table
        std::vector<double> Q_jac;
        build_jacobi_quant_table(p_max, jac_base_step, alpha, Q_jac);

        // Quantize with degree truncation
        std::vector<int16_t> q_jac_vec(2 * M);
        std::vector<double> c0_q(M), c1_q(M);

        for (int m = 0; m < M; ++m)
        {
          int deg = degree_from_index(m, p_max);
          if (deg > jac_deg_keep)
          {
            q_jac_vec[2*m + 0] = 0;
            q_jac_vec[2*m + 1] = 0;
            c0_q[m] = 0.0;
            c1_q[m] = 0.0;
          }
          else
          {
            double step_m = Q_jac[m];
            int q0 = quantize_scalar(c0[m], step_m);
            int q1 = quantize_scalar(c1[m], step_m);

            q_jac_vec[2*m + 0] = (int16_t)q0;
            q_jac_vec[2*m + 1] = (int16_t)q1;

            c0_q[m] = dequantize_scalar(q0, step_m);
            c1_q[m] = dequantize_scalar(q1, step_m);
          }
        }

        // Reconstruct block from truncated / quantized coeffs
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

        // RLE + zlib on quantized coeffs
        std::vector<int16_t> jac_rle;
        rle_encode_zeros(q_jac_vec, jac_rle);
        size_t jac_bytes = zlib_compress_int16(jac_rle);
        double jac_bpp = (double)(jac_bytes * 8) / 64.0;

        std::cout << "base_step=" << jac_base_step
                  << "  Jac: bpp=" << jac_bpp
                  << " PSNR=" << psnr_jac << " dB\n";
      }

      std::cout << "\n";
    }

    std::cout << "\n";
  }

  return 0;
}

