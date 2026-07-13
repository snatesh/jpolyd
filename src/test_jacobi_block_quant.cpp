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

// --------------------- Quantization helpers ---------------------

// Return total degree for mode index m in graded lex ordering
static int degree_from_index(int m, int p_max)
{
  // dim Pi_d^2 = (d+1)(d+2)/2
  for (int d = 0; d <= p_max; ++d)
  {
    int dim_d = (d + 1) * (d + 2) / 2;
    if (m < dim_d)
      return d;
  }
  // Should not happen if m < dim Pi_p_max
  return p_max;
}

// Build per-mode Jacobi quantization steps Q_jac[m]
// p_max: total-degree max (e.g. 6)
// base_step: overall scale (like JPEG "quality" scaling)
// alpha: how fast to grow with degree (tune by hand if you like)
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

    // Example heuristic:
    //   Q(m) = base_step * (1 + alpha * deg^2)
    // so deg=0 -> base_step, deg=6 -> base_step*(1+36*alpha)
    double factor = 1.0 + alpha * (double)(deg * deg);
    Q_jac[m] = base_step * factor;
  }
}


// Naive scalar quantization: q = round(c / Qstep)
static inline int quantize_scalar(double c, double Qstep)
{
  return (int) std::llround(c / Qstep);
}

static inline double dequantize_scalar(int q, double Qstep)
{
  return (double)q * Qstep;
}

// Approximate bit cost for an array of quantized coefficients
// Model: for each nonzero q, bits ~ 1 (sign) + ceil(log2(|q|+1))
static double approximate_bit_cost(const std::vector<int> &qcoeffs)
{
  double bits = 0.0;
  for (int v : qcoeffs)
  {
    if (v == 0) continue;
    int mag = std::abs(v);
    int mag_plus = mag + 1;
    int mag_bits = 0;
    while ((1 << mag_bits) < mag_plus) ++mag_bits;
    bits += 1 + mag_bits;  // sign + magnitude bits
  }
  return bits;
}

// --------------------- main ---------------------

int main()
{
  // -----------------------------------------------------------
  // 1) Build a test 8x8 pixel block
  // -----------------------------------------------------------
  double pix[8][8];

  // Choose test type: 0=gradient, 1=diagonal step, 2=high freq
  //int block_type = 1;

  for (int block_type = 0; block_type < 3; ++block_type)
  {

  std::cout << block_type << std::endl;
  if (block_type == 0)
  {
    
    // simple gradient
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        pix[i][j] = (double)i + 0.5 * (double)j;
  }
  else if (block_type == 1)
  {
    // diagonal step: low vs high on i + j < 8 vs >= 8
    for (int i = 0; i < 8; ++i)
    {
      for (int j = 0; j < 8; ++j)
      {
        if (i + j < 8)
          pix[i][j] = 50.0;
        else
          pix[i][j] = 200.0;
      }
    }
  }
  else
  {
    // high-frequency-ish pattern
    const double PI = 3.14159265358979323846;
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

  // -----------------------------------------------------------
  // 2) DCT forward (no quant yet)
  // -----------------------------------------------------------
  double dct[8][8];
  dct2_8x8(pix, dct);

  // Baseline: perfect recon (no quant)
  double pix_dct_rec_full[8][8];
  idct2_8x8(dct, pix_dct_rec_full);

  double mse_dct_full, psnr_dct_full;
  compute_mse_psnr(pix, pix_dct_rec_full, mse_dct_full, psnr_dct_full);

  std::cout << "DCT full recon MSE................: " << mse_dct_full << "\n";
  std::cout << "DCT full recon PSNR (max=255).....: " << psnr_dct_full << " dB\n";

  // -----------------------------------------------------------
  // 3) Load Q for both triangles (Jacobi)
  // -----------------------------------------------------------
  QTriangle tri0, tri1;
  std::string file0 = "jacobi_Q_tri0_p6.txt";
  std::string file1 = "jacobi_Q_tri1_p6.txt";

  if (!load_Q_triangle(file0, tri0)) return 1;
  if (!load_Q_triangle(file1, tri1)) return 1;

  if (tri0.p_max != tri1.p_max || tri0.M != tri1.M)
  {
    std::cerr << "ERROR: triangle 0 and 1 Q have mismatched p_max or M.\n";
    return 1;
  }

  int p_max = tri0.p_max;
  int M = tri0.M;
  int P0 = tri0.P;
  int P1 = tri1.P;

  std::cout << "Loaded Jacobi transforms: p_max=" << p_max
            << ", M=" << M << ", P0=" << P0 << ", P1=" << P1 << "\n";

  // -----------------------------------------------------------
  // 4) Jacobi encode (full, no quant yet)
  // -----------------------------------------------------------
  std::vector<double> f0(P0), f1(P1);
  std::vector<double> c0(M),  c1(M);

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

  // Baseline perfect Jacobi recon (no quantization)
  std::vector<double> f0_rec_full(P0), f1_rec_full(P1);

  for (int p = 0; p < P0; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M; ++m)
      val += c0[m] * tri0.Q[p + m * P0];
    f0_rec_full[p] = val;
  }
  for (int p = 0; p < P1; ++p)
  {
    double val = 0.0;
    for (int m = 0; m < M; ++m)
      val += c1[m] * tri1.Q[p + m * P1];
    f1_rec_full[p] = val;
  }

  double pix_jacobi_full[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      pix_jacobi_full[i][j] = 0.0;

  for (int k = 0; k < P0; ++k)
  {
    int idx = tri0.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    pix_jacobi_full[i][j] = f0_rec_full[k];
  }
  for (int k = 0; k < P1; ++k)
  {
    int idx = tri1.pix_idx[k];
    int i = idx / 8;
    int j = idx % 8;
    pix_jacobi_full[i][j] = f1_rec_full[k];
  }

  double mse_jac_full, psnr_jac_full;
  compute_mse_psnr(pix, pix_jacobi_full, mse_jac_full, psnr_jac_full);

  std::cout << "Jacobi full recon MSE.............: " << mse_jac_full << "\n";
  std::cout << "Jacobi full recon PSNR (max=255)..: " << psnr_jac_full << " dB\n";

  // -----------------------------------------------------------
  // 5) Quantization experiments: scalar Q steps
  // -----------------------------------------------------------
  std::vector<double> Qvals = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};

  std::cout << "\n=== Scalar quantization experiments ===\n";
  std::cout << "Columns: Q, bits/pixel (DCT), PSNR(DCT), bits/pixel (Jac), PSNR(Jac)\n";

  for (double Qstep : Qvals)
  {
    // ---- DCT quant/dequant ----
    std::vector<int> q_dct(64);
    double dct_q[8][8];

    int idx = 0;
    for (int u = 0; u < 8; ++u)
    {
      for (int v = 0; v < 8; ++v)
      {
        int q = quantize_scalar(dct[u][v], Qstep);
        q_dct[idx++] = q;
        dct_q[u][v] = dequantize_scalar(q, Qstep);
      }
    }

    double dct_bits = approximate_bit_cost(q_dct);
    double dct_bpp  = dct_bits / 64.0;

    double pix_dct_q_rec[8][8];
    idct2_8x8(dct_q, pix_dct_q_rec);

    double mse_dct_q, psnr_dct_q;
    compute_mse_psnr(pix, pix_dct_q_rec, mse_dct_q, psnr_dct_q);

    // ---- Jacobi quant/dequant with per-mode table ----
    std::vector<double> Q_jac; 
    // base_step = Qstep, alpha ~ 0.3 is a decent starting guess
    build_jacobi_quant_table(p_max, Qstep, 0.3, Q_jac);
    
    std::vector<int> q_jac(2 * M);
    std::vector<double> c0_q(M), c1_q(M);
    
    for (int m = 0; m < M; ++m)
    {
      double step_m = Q_jac[m];
    
      int q0 = quantize_scalar(c0[m], step_m);
      int q1 = quantize_scalar(c1[m], step_m);
    
      q_jac[2*m + 0] = q0;
      q_jac[2*m + 1] = q1;
    
      c0_q[m] = dequantize_scalar(q0, step_m);
      c1_q[m] = dequantize_scalar(q1, step_m);
    }

    double jac_bits = approximate_bit_cost(q_jac);
    double jac_bpp  = jac_bits / 64.0;

    // reconstruct from c0_q, c1_q
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

    double pix_jac_q[8][8];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        pix_jac_q[i][j] = 0.0;

    for (int k = 0; k < P0; ++k)
    {
      int idxp = tri0.pix_idx[k];
      int i = idxp / 8;
      int j = idxp % 8;
      pix_jac_q[i][j] = f0_q[k];
    }
    for (int k = 0; k < P1; ++k)
    {
      int idxp = tri1.pix_idx[k];
      int i = idxp / 8;
      int j = idxp % 8;
      pix_jac_q[i][j] = f1_q[k];
    }

    double mse_jac_q, psnr_jac_q;
    compute_mse_psnr(pix, pix_jac_q, mse_jac_q, psnr_jac_q);

    std::cout << "Q=" << Qstep
              << "   DCT:  bpp=" << dct_bpp
              << "  PSNR=" << psnr_dct_q
              << "   Jac:  bpp=" << jac_bpp
              << "  PSNR=" << psnr_jac_q
              << " dB\n";
  }
  }
  return 0;
}
