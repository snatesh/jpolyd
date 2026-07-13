// brick_tet_codec.cpp
// Prototype 3D tet-based video encoder/decoder for a single brick.
//
// Build on Ubuntu (after `sudo apt install libopencv-dev`):
//   mkdir build && cd build
//   cmake ..   (with a CMakeLists.txt that finds OpenCV)
//   make
//
// Or directly:
//   g++ brick_tet_codec.cpp -o brick_tet_codec `pkg-config --cflags --libs opencv4`
//
// Adjust paths for jbasis.hh, quadrature file, and input video.

#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <string>

#include <opencv2/opencv.hpp>

// Adjust to correct path for your basis header
#include "jbasis.hh"

using jsimplex::Basis;

struct Vec3 {
  double x, y, z;
};

struct Tet {
  int v[4];  // indices into the brick-vertex array
};

// Unit cube vertices in (x,y,t) ∈ [0,1]^3
static const std::array<Vec3, 8> cubeVerts = {{
  {0.0, 0.0, 0.0}, // v0
  {1.0, 0.0, 0.0}, // v1
  {1.0, 1.0, 0.0}, // v2
  {0.0, 1.0, 0.0}, // v3
  {0.0, 0.0, 1.0}, // v4
  {1.0, 0.0, 1.0}, // v5
  {1.0, 1.0, 1.0}, // v6
  {0.0, 1.0, 1.0}  // v7
}};

// 6-tet decomposition of the cube using body diagonal v0->v6
static const std::array<Tet, 6> cubeTets = {{
  {{0, 1, 2, 6}},
  {{0, 2, 3, 6}},
  {{0, 3, 7, 6}},
  {{0, 7, 4, 6}},
  {{0, 4, 5, 6}},
  {{0, 5, 1, 6}}
}};

// Simple vector ops
inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator*(double s, const Vec3 &a) {
  return {s * a.x, s * a.y, s * a.z};
}

// Map reference tet point r=(r,s,t) to physical cube tet K
Vec3 map_ref_to_tet(const Vec3 &r,
                    const std::array<Vec3,8> &verts,
                    const Tet &K) {
  const Vec3 &v0 = verts[K.v[0]];
  const Vec3 &v1 = verts[K.v[1]];
  const Vec3 &v2 = verts[K.v[2]];
  const Vec3 &v3 = verts[K.v[3]];

  Vec3 e1 = v1 - v0;
  Vec3 e2 = v2 - v0;
  Vec3 e3 = v3 - v0;

  return v0 + r.x * e1 + r.y * e2 + r.z * e3;
}

// |det J_K| for tet K in the cube
double tet_jacobian_det(const std::array<Vec3,8> &verts,
                        const Tet &K) {
  const Vec3 &v0 = verts[K.v[0]];
  const Vec3 &v1 = verts[K.v[1]];
  const Vec3 &v2 = verts[K.v[2]];
  const Vec3 &v3 = verts[K.v[3]];

  Vec3 e1 = v1 - v0;
  Vec3 e2 = v2 - v0;
  Vec3 e3 = v3 - v0;

  double det =
      e1.x * (e2.y * e3.z - e2.z * e3.y)
    - e1.y * (e2.x * e3.z - e2.z * e3.x)
    + e1.z * (e2.x * e3.y - e2.y * e3.x);

  return std::fabs(det);
}

// Compute inverse of B=[e1 e2 e3] (columns), store in B_inv as row-major 3x3
void compute_B_inv(const Vec3 &e1, const Vec3 &e2, const Vec3 &e3,
                   double B_inv[9]) {
  double det =
      e1.x * (e2.y * e3.z - e2.z * e3.y)
    - e1.y * (e2.x * e3.z - e2.z * e3.x)
    + e1.z * (e2.x * e3.y - e2.y * e3.x);

  double invDet = 1.0 / det;

  // Cofactor matrix (then transpose for adjugate)
  double c00 =  (e2.y * e3.z - e2.z * e3.y);
  double c01 = -(e2.x * e3.z - e2.z * e3.x);
  double c02 =  (e2.x * e3.y - e2.y * e3.x);

  double c10 = -(e1.y * e3.z - e1.z * e3.y);
  double c11 =  (e1.x * e3.z - e1.z * e3.x);
  double c12 = -(e1.x * e3.y - e1.y * e3.x);

  double c20 =  (e1.y * e2.z - e1.z * e2.y);
  double c21 = -(e1.x * e2.z - e1.z * e2.x);
  double c22 =  (e1.x * e2.y - e1.y * e2.x);

  // B_inv = adj(B)/det = C^T / det
  B_inv[0] = c00 * invDet;
  B_inv[1] = c10 * invDet;
  B_inv[2] = c20 * invDet;

  B_inv[3] = c01 * invDet;
  B_inv[4] = c11 * invDet;
  B_inv[5] = c21 * invDet;

  B_inv[6] = c02 * invDet;
  B_inv[7] = c12 * invDet;
  B_inv[8] = c22 * invDet;
}

// Load quadrature: (D+1)*N doubles, first N*D coords, then N weights
bool load_tet_quadrature(const std::string &filename,
                         std::vector<Vec3> &Xref,
                         std::vector<double> &W) {
  std::ifstream in(filename);
  if (!in) {
    std::cerr << "Could not open quadrature file: " << filename << "\n";
    return false;
  }

  std::vector<double> data;
  double val;
  while (in >> val) {
    data.push_back(val);
  }

  const int D = 3;
  if (data.size() % (D + 1) != 0) {
    std::cerr << "Quadrature data length not divisible by D+1.\n";
    return false;
  }

  int N = static_cast<int>(data.size()) / (D + 1);
  Xref.resize(N);
  W.resize(N);

  for (int q = 0; q < N; ++q) {
    Xref[q].x = data[q * D + 0];
    Xref[q].y = data[q * D + 1];
    Xref[q].z = data[q * D + 2];
  }
  for (int q = 0; q < N; ++q) {
    W[q] = data[N * D + q];
  }

  return true;
}

// Basis wrapper
struct TetBasis3D {
  int p;         // total degree
  int M;         // dim_Pi(p)
  std::vector<int> alpha_table;
  std::vector<int> tail_deg;
  std::vector<double> inv_h;

  // Optional cached values of basis at reference quadrature points
  std::vector<double> Vref;
  int ld_V;      // leading dimension (Nq)

  void init_with_quadrature(int degree,
                            const double *kappa,
                            const std::vector<Vec3> &Xref) {
    p = degree;
    M = Basis<3,double>::dim_Pi(p);

    alpha_table.resize(M * 3);
    tail_deg.resize(M * 3);
    inv_h.resize(M);

    Basis<3,double>::build_structures(
      kappa, p,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data()
    );

    int Nq = static_cast<int>(Xref.size());
    ld_V = Nq;
    Vref.assign(M * ld_V, 0.0);

    std::vector<double> Xraw(Nq * 3);
    for (int q = 0; q < Nq; ++q) {
      Xraw[q * 3 + 0] = Xref[q].x;
      Xraw[q * 3 + 1] = Xref[q].y;
      Xraw[q * 3 + 2] = Xref[q].z;
    }

    Basis<3,double>::eval_all(
      Xraw.data(),
      3,  // ld_point
      1,  // ld_dim
      Nq,
      kappa,
      p,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data(),
      Vref.data(),
      ld_V,
      nullptr
    );
  }

  void init_structures_only(int degree, const double *kappa) {
    p = degree;
    M = Basis<3,double>::dim_Pi(p);
    alpha_table.resize(M * 3);
    tail_deg.resize(M * 3);
    inv_h.resize(M);
    Basis<3,double>::build_structures(
      kappa, p,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data()
    );
    ld_V = 0;
    Vref.clear();
  }

  inline double phi_ref(int m, int q) const {
    return Vref[q + m * ld_V];
  }
};

// Piecewise-constant sampler for a single space-time brick
struct VideoBrickSampler {
  int imgW, imgH;    // full frame size
  int Bx, By, Bt;    // brick size in pixels/frames
  int bx0, by0;      // brick origin (top-left) in pixels
  int t0;            // starting frame index in the video
  std::vector<cv::Mat> frames;  // Bt grayscale frames [0..Bt-1]

  // uvw ∈ [0,1]^3 -> sample Y ∈ [0,255] as piecewise-constant per cell
  double sample_local(const Vec3 &uvw) const {
    auto clamp01m = [](double s) {
      return std::min(std::max(s, 0.0), std::nextafter(1.0, 0.0));
    };
    double u = clamp01m(uvw.x);
    double v = clamp01m(uvw.y);
    double w = clamp01m(uvw.z);

    int lx = static_cast<int>(std::floor(u * Bx));
    int ly = static_cast<int>(std::floor(v * By));
    int lt = static_cast<int>(std::floor(w * Bt));

    lx = std::min(std::max(lx, 0), Bx - 1);
    ly = std::min(std::max(ly, 0), By - 1);
    lt = std::min(std::max(lt, 0), Bt - 1);

    int x = bx0 + lx;
    int y = by0 + ly;
    int k = lt;

    x = std::min(std::max(x, 0), imgW - 1);
    y = std::min(std::max(y, 0), imgH - 1);

    const uchar *row = frames[k].ptr<uchar>(y);
    return static_cast<double>(row[x]);
  }
};

// Load Bt frames starting at t0 into sampler.frames, as grayscale
bool load_brick_frames(const std::string &filename,
                       int t0, int Bt,
                       VideoBrickSampler &sampler) {
  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cerr << "Could not open video: " << filename << "\n";
    return false;
  }

  int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  sampler.imgW = W;
  sampler.imgH = H;

  if (t0 >= frameCount) {
    std::cerr << "t0 beyond end of video.\n";
    return false;
  }

  cap.set(cv::CAP_PROP_POS_FRAMES, t0);

  sampler.frames.clear();
  sampler.frames.reserve(Bt);

  for (int k = 0; k < Bt; ++k) {
    cv::Mat frameBGR;
    if (!cap.read(frameBGR)) {
      break;
    }
    cv::Mat gray;
    cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
    sampler.frames.push_back(gray);
  }

  if (static_cast<int>(sampler.frames.size()) < Bt) {
    std::cerr << "Reached end of video before Bt frames.\n";
    return false;
  }

  return true;
}

// ---------------- ENCODE ----------------

// Simple text coeff format:
//
// First line: p Bx By Bt numTets M
// Then for each tet k = 0..numTets-1:
//   One line with M doubles: c_0 c_1 ... c_{M-1}
//
// No quantization, raw doubles.
bool encode_brick(const std::string &videoFile,
                  const std::string &quadFile,
                  const std::string &coeffFile,
                  int Bx, int By, int Bt,
                  int bx0, int by0, int t0,
                  int degree) {
  // 1) Load quadrature
  std::vector<Vec3> Xref;
  std::vector<double> W;
  if (!load_tet_quadrature(quadFile, Xref, W)) {
    return false;
  }
  int Nq = static_cast<int>(Xref.size());

  // 2) Basis
  double kappa[4] = {0.5, 0.5, 0.5, 0.5};
  TetBasis3D basis;
  basis.init_with_quadrature(degree, kappa, Xref);
  int M = basis.M;
  int numTets = static_cast<int>(cubeTets.size());

  // 3) Video brick sampler
  VideoBrickSampler sampler;
  sampler.Bx  = Bx;
  sampler.By  = By;
  sampler.Bt  = Bt;
  sampler.bx0 = bx0;
  sampler.by0 = by0;
  sampler.t0  = t0;

  if (!load_brick_frames(videoFile, sampler.t0, sampler.Bt, sampler)) {
    return false;
  }

  std::cout << "Encoding brick: "
            << "Bx=" << Bx << " By=" << By << " Bt=" << Bt
            << ", degree=" << degree
            << ", M=" << M
            << "\n";

  // 4) Compute coefficients per tet
  std::vector<std::vector<double>> coeffs(numTets, std::vector<double>(M, 0.0));

  for (int kt = 0; kt < numTets; ++kt) {
    const Tet &K = cubeTets[kt];
    auto &ck = coeffs[kt];

    for (int q = 0; q < Nq; ++q) {
      Vec3 r   = Xref[q];
      Vec3 uvw = map_ref_to_tet(r, cubeVerts, K);

      double u_q = sampler.sample_local(uvw);
      double w_q = W[q];

      for (int m = 0; m < M; ++m) {
        ck[m] += w_q * u_q * basis.phi_ref(m, q);
      }
    }
  }

  // 5) Write coefficients to file
  std::ofstream out(coeffFile);
  if (!out) {
    std::cerr << "Could not open coeff output file: " << coeffFile << "\n";
    return false;
  }

  out << degree << " " << Bx << " " << By << " " << Bt
      << " " << numTets << " " << M << "\n";

  for (int kt = 0; kt < numTets; ++kt) {
    for (int m = 0; m < M; ++m) {
      out << coeffs[kt][m];
      if (m + 1 < M) out << " ";
    }
    out << "\n";
  }

  std::cout << "Wrote coefficients to " << coeffFile << "\n";
  return true;
}

// ---------------- DECODE ----------------

struct CoeffData {
  int degree;
  int Bx, By, Bt;
  int numTets;
  int M;
  std::vector<std::vector<double>> coeffs; // [numTets][M]
};

bool read_coeffs(const std::string &coeffFile, CoeffData &cd) {
  std::ifstream in(coeffFile);
  if (!in) {
    std::cerr << "Could not open coeff file: " << coeffFile << "\n";
    return false;
  }

  in >> cd.degree >> cd.Bx >> cd.By >> cd.Bt >> cd.numTets >> cd.M;
  if (!in) {
    std::cerr << "Failed to read header from coeff file.\n";
    return false;
  }

  cd.coeffs.assign(cd.numTets, std::vector<double>(cd.M));
  for (int kt = 0; kt < cd.numTets; ++kt) {
    for (int m = 0; m < cd.M; ++m) {
      in >> cd.coeffs[kt][m];
      if (!in) {
        std::cerr << "Error reading coeffs.\n";
        return false;
      }
    }
  }
  return true;
}

// Decode to grayscale frames of size Nx x Ny x Nt
bool decode_brick(const std::string &coeffFile,
                  int Nx, int Ny, int Nt,
                  std::vector<cv::Mat> &framesOut) {
  CoeffData cd;
  if (!read_coeffs(coeffFile, cd)) {
    return false;
  }

  int degree = cd.degree;
  int numTets = cd.numTets;
  int M = cd.M;

  // Basis structures only
  double kappa[4] = {0.5, 0.5, 0.5, 0.5};
  TetBasis3D basis;
  basis.init_structures_only(degree, kappa);

  // Precompute affine inverses for each tet
  struct TetAffine {
    Vec3 v0;
    double B_inv[9]; // row-major
  };
  std::vector<TetAffine> taff(numTets);

  for (int kt = 0; kt < numTets; ++kt) {
    const Tet &K = cubeTets[kt];
    Vec3 v0 = cubeVerts[K.v[0]];
    Vec3 v1 = cubeVerts[K.v[1]];
    Vec3 v2 = cubeVerts[K.v[2]];
    Vec3 v3 = cubeVerts[K.v[3]];

    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 e3 = v3 - v0;

    taff[kt].v0 = v0;
    compute_B_inv(e1, e2, e3, taff[kt].B_inv);
  }

  int totalSamples = Nx * Ny * Nt;
  std::vector<double> uhat(totalSamples, 0.0);

  // For each sample grid point, determine which tet it lies in and its ref coords
  struct SampleRef {
    int sampleIdx;
    Vec3 r; // reference tet coordinates
  };
  std::vector<std::vector<SampleRef>> samplesPerTet(numTets);

  int idx = 0;
  const double eps = 1e-12;

  for (int kt = 0; kt < numTets; ++kt) {
    samplesPerTet[kt].clear();
  }

  for (int it = 0; it < Nt; ++it) {
    double w = (it + 0.5) / Nt;
    for (int iy = 0; iy < Ny; ++iy) {
      double v = (iy + 0.5) / Ny;
      for (int ix = 0; ix < Nx; ++ix) {
        double u = (ix + 0.5) / Nx;
        Vec3 x = {u, v, w};

        bool assigned = false;
        for (int kt = 0; kt < numTets && !assigned; ++kt) {
          const TetAffine &TA = taff[kt];
          Vec3 dx = x - TA.v0;
          const double *B = TA.B_inv;
          Vec3 r;
          r.x = B[0] * dx.x + B[1] * dx.y + B[2] * dx.z;
          r.y = B[3] * dx.x + B[4] * dx.y + B[5] * dx.z;
          r.z = B[6] * dx.x + B[7] * dx.y + B[8] * dx.z;

          double s = r.x + r.y + r.z;
          if (r.x >= -eps && r.y >= -eps && r.z >= -eps && s <= 1.0 + eps) {
            samplesPerTet[kt].push_back({idx, r});
            assigned = true;
          }
        }

        if (!assigned) {
          // Shouldn't happen, but if it does, leave uhat[idx]=0
        }
        ++idx;
      }
    }
  }

  // For each tet: evaluate basis at its sample ref points and reconstruct
  for (int kt = 0; kt < numTets; ++kt) {
    auto &lst = samplesPerTet[kt];
    int npts = static_cast<int>(lst.size());
    if (npts == 0) continue;

    // Build Xraw for these reference points
    std::vector<double> Xraw(npts * 3);
    for (int i = 0; i < npts; ++i) {
      Xraw[i * 3 + 0] = lst[i].r.x;
      Xraw[i * 3 + 1] = lst[i].r.y;
      Xraw[i * 3 + 2] = lst[i].r.z;
    }

    // Evaluate basis at these points
    std::vector<double> V_eval(M * npts);
    int ld_V = npts;

    Basis<3,double>::eval_all(
      Xraw.data(),
      3,  // ld_point
      1,  // ld_dim
      npts,
      kappa,
      degree,
      basis.alpha_table.data(),
      basis.tail_deg.data(),
      basis.inv_h.data(),
      V_eval.data(),
      ld_V,
      nullptr
    );

    // Reconstruct
    const auto &ck = cd.coeffs[kt];

    for (int i = 0; i < npts; ++i) {
      int sidx = lst[i].sampleIdx;
      double val = 0.0;
      for (int m = 0; m < M; ++m) {
        val += ck[m] * V_eval[i + m * ld_V];
      }
      uhat[sidx] = val;
    }
  }

  // Convert uhat to frames (clamp to [0,255])
  framesOut.clear();
  framesOut.resize(Nt);
  idx = 0;
  for (int it = 0; it < Nt; ++it) {
    framesOut[it] = cv::Mat(Ny, Nx, CV_8UC1);
    for (int iy = 0; iy < Ny; ++iy) {
      uchar *row = framesOut[it].ptr<uchar>(iy);
      for (int ix = 0; ix < Nx; ++ix) {
        double v = uhat[idx++];
        if (v < 0.0) v = 0.0;
        if (v > 255.0) v = 255.0;
        row[ix] = static_cast<uchar>(std::round(v));
      }
    }
  }

  std::cout << "Decoded brick to " << Nt
            << " frames of size " << Nx << "x" << Ny << "\n";
  return true;
}

// ---------------- WRITE VIDEO ----------------

bool write_video(const std::string &outFile,
                 const std::vector<cv::Mat> &frames,
                 double fps) {
  if (frames.empty()) {
    std::cerr << "No frames to write.\n";
    return false;
  }
  int W = frames[0].cols;
  int H = frames[0].rows;

  int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
  cv::VideoWriter writer(outFile, fourcc, fps, cv::Size(W, H), false);
  if (!writer.isOpened()) {
    std::cerr << "Could not open VideoWriter for " << outFile << "\n";
    return false;
  }

  for (const auto &f : frames) {
    writer.write(f);
  }
  std::cout << "Wrote video to " << outFile << "\n";
  return true;
}

// ---------------- MAIN (demo) ----------------

int main(int argc, char* argv[]) {
  // Hard-coded demo params; you can wire these to CLI args later
  std::string quadFile  = "t3_kappa0.5-0.5-0.5-0.5_N364_M1330_n11_m18.txt";
  std::string videoFile(argv[1]);
  std::string coeffFile = "brick_coeffs.txt";
  std::string reconFile = "recon.mp4";

  int Bx = 16, By = 16, Bt = 8;
  int bx0 = 0, by0 = 0, t0 = 0;
  int degree = 4;

  // ENCODE
  if (!encode_brick(videoFile, quadFile, coeffFile,
                    Bx, By, Bt, bx0, by0, t0, degree)) {
    std::cerr << "Encode failed.\n";
    return 1;
  }

  // DECODE to same resolution as brick for now
  int Nx = Bx;
  int Ny = By;
  int Nt = 10*Bt;

  std::vector<cv::Mat> reconFrames;
  if (!decode_brick(coeffFile, Nx, Ny, Nt, reconFrames)) {
    std::cerr << "Decode failed.\n";
    return 1;
  }

  // WRITE reconstructed video
  double fps = 30.0; // or read from original video if you like
  if (!write_video(reconFile, reconFrames, fps)) {
    return 1;
  }

  std::cout << "Done.\n";
  return 0;
}

