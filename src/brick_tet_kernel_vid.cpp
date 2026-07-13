// brick_tet_video_l2.cpp
// Compile with something like:
//   g++ brick_tet_video_l2.cpp -o brick_tet_video_l2 `pkg-config --cflags --libs opencv4`
//
// Adjust includes, quadFile and videoFile paths as needed.

#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

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

// Convenience wrapper: basis on 3D tet
struct TetBasis3D {
  int p;         // total degree
  int M;         // dim_Pi(p)
  std::vector<int> alpha_table;
  std::vector<int> tail_deg;
  std::vector<double> inv_h;
  std::vector<double> Vref; // size M * Nq (layout: V[q + m*ld_V])
  int ld_V;      // leading dimension (Nq)

  void init(int degree,
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

    // pack reference nodes
    std::vector<double> Xraw(Nq * 3);
    for (int q = 0; q < Nq; ++q) {
      Xraw[q * 3 + 0] = Xref[q].x;
      Xraw[q * 3 + 1] = Xref[q].y;
      Xraw[q * 3 + 2] = Xref[q].z;
    }

    Basis<3,double>::eval_all(
      Xraw.data(),
      /*ld_point*/ 3,
      /*ld_dim*/   1,
      /*npts*/     Nq,
      kappa,
      p,
      alpha_table.data(),
      tail_deg.data(),
      inv_h.data(),
      Vref.data(),
      ld_V,
      /*dV*/ nullptr
    );
  }

  inline double phi(int m, int q) const {
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
    // clamp to [0,1)
    auto clamp01m = [](double s) {
      return std::min(std::max(s, 0.0), std::nextafter(1.0, 0.0));
    };
    double u = clamp01m(uvw.x);
    double v = clamp01m(uvw.y);
    double w = clamp01m(uvw.z);

    // map to cell indices in this brick
    int lx = static_cast<int>(std::floor(u * Bx));
    int ly = static_cast<int>(std::floor(v * By));
    int lt = static_cast<int>(std::floor(w * Bt));

    lx = std::min(std::max(lx, 0), Bx - 1);
    ly = std::min(std::max(ly, 0), By - 1);
    lt = std::min(std::max(lt, 0), Bt - 1);

    int x = bx0 + lx;
    int y = by0 + ly;
    int k = lt;  // frame index within this slab

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

int main(int argc, char* argv[]) {
  // 1) Load quadrature on reference tet
  std::vector<Vec3> Xref;
  std::vector<double> W;

  // TODO: set this to your actual quadrature file path
  std::string quadFile =
    "t3_kappa0.5-0.5-0.5-0.5_N364_M1330_n11_m18.txt";

  if (!load_tet_quadrature(quadFile, Xref, W)) {
    return 1;
  }

  int Nq = static_cast<int>(Xref.size());
  std::cout << "Loaded " << Nq << " quadrature points.\n";

  // 2) Build 3D Jacobi basis (kappa = 0.5 for uniform weight)
  double kappa[4] = {0.5, 0.5, 0.5, 0.5};
  int degree = 9;  // total degree p (adjust as you like)

  TetBasis3D basis;
  basis.init(degree, kappa, Xref);
  std::cout << "3D basis dim M = " << basis.M << "\n";

  // 3) Prepare video brick sampler
  VideoBrickSampler sampler;
  sampler.Bx  = 16;         // brick size in pixels
  sampler.By  = 16;
  sampler.Bt  = 8;          // frames per brick (time slab)
  sampler.bx0 = 0;          // brick origin (top-left) in pixels
  sampler.by0 = 0;
  sampler.t0  = 0;          // starting frame index for this slab

  // TODO: set this to your video file path
  std::string videoFile(argv[1]);

  if (!load_brick_frames(videoFile, sampler.t0, sampler.Bt, sampler)) {
    return 1;
  }

  std::cout << "Video frames loaded: "
            << sampler.frames.size()
            << " (" << sampler.imgW << "x" << sampler.imgH << ")\n";

  double total_err2 = 0.0;
  double total_u2   = 0.0;

  // 4) For each tet: project video brick and accumulate L2 error
  for (int kt = 0; kt < 6; ++kt) {
    const Tet &K = cubeTets[kt];
    double Jdet  = tet_jacobian_det(cubeVerts, K);

    std::vector<double> coeffs(basis.M, 0.0);

    // a) compute coefficients c_m^K
    for (int q = 0; q < Nq; ++q) {
      Vec3 r   = Xref[q];                         // ref node
      Vec3 uvw = map_ref_to_tet(r, cubeVerts, K); // in [0,1]^3

      double u_q = sampler.sample_local(uvw);     // piecewise-constant video
      double w_q = W[q];

      for (int m = 0; m < basis.M; ++m) {
        coeffs[m] += w_q * u_q * basis.phi(m, q);
      }
    }

    // b) reconstruct and accumulate L2 error on this tet
    for (int q = 0; q < Nq; ++q) {
      Vec3 r   = Xref[q];
      Vec3 uvw = map_ref_to_tet(r, cubeVerts, K);

      double u_q  = sampler.sample_local(uvw);
      double uh_q = 0.0;

      for (int m = 0; m < basis.M; ++m) {
        uh_q += coeffs[m] * basis.phi(m, q);
      }

      double diff = u_q - uh_q;

      total_err2 += Jdet * W[q] * diff * diff;
      total_u2   += Jdet * W[q] * u_q  * u_q;
    }
  }

  double L2_err = std::sqrt(total_err2);
  double L2_u   = std::sqrt(total_u2);
  double rel_err = L2_err / (L2_u + 1e-15);

  std::cout << "L2(video brick)       ≈ " << L2_u   << "\n";
  std::cout << "L2(u - uh) over brick ≈ " << L2_err << "\n";
  std::cout << "Relative L2 error     ≈ " << rel_err << "\n";

  return 0;
}

