#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <jbasis.hh>  // jsimplex::Basis<2,double>, like in your existing code

struct Vec2 { double x,y; };

struct TriCode {
  uint8_t root;  // 0 or 1
  int     level; // 0..max_level
  uint64_t code; // 2 bits per level, child index 0..3

  TriCode(uint8_t r=0, int L=0, uint64_t c=0) : root(r), level(L), code(c) {}
};

inline TriCode child_code(const TriCode &parent, uint8_t child_idx) {
  TriCode c = parent;
  c.level += 1;
  c.code  = (c.code << 2) | (uint64_t(child_idx) & 0x3u);
  return c;
}

// Same decode_triangle we used in the adaptive prototype:
static void decode_triangle(const TriCode &tc, Vec2 &A, Vec2 &B, Vec2 &C)
{
  if (tc.root == 0) {
    A={0.0,0.0}; B={1.0,0.0}; C={0.0,1.0};
  } else {
    A={1.0,1.0}; B={1.0,0.0}; C={0.0,1.0};
  }

  uint64_t code = tc.code;
  int L = tc.level;
  for (int k=L-1; k>=0; --k) {
    uint8_t child = uint8_t((code >> (2*k)) & 0x3u);

    Vec2 m_ab{0.5*(A.x+B.x), 0.5*(A.y+B.y)};
    Vec2 m_bc{0.5*(B.x+C.x), 0.5*(B.y+C.y)};
    Vec2 m_ca{0.5*(C.x+A.x), 0.5*(C.y+A.y)};

    switch(child) {
      case 0: B=m_ab; C=m_ca; break;          // (A,m_ab,m_ca)
      case 1: A=m_ab; C=m_bc; break;          // (m_ab,B,m_bc)
      case 2: A=m_ca; B=m_bc; break;          // (m_ca,m_bc,C)
      case 3: A=m_ab; B=m_bc; C=m_ca; break;  // (m_ab,m_bc,m_ca)
    }
  }
}

// Compute barycentric coordinates of P w.r.t triangle ABC.
// Returns (u,v,w) where w = 1 - u - v.
// Returns true if strictly inside or on boundary.
//
// NOTE: We keep the core routine symmetric.
// Asymmetric splitting is handled outside for level==0.
static inline bool barycentric(const Vec2 &A,
                               const Vec2 &B,
                               const Vec2 &C,
                               const Vec2 &P,
                               double &u,
                               double &v)
{
    // Solve:
    //   P = A + u (B-A) + v (C-A)
    double x = P.x - A.x;
    double y = P.y - A.y;

    double b_x = B.x - A.x;
    double b_y = B.y - A.y;
    double c_x = C.x - A.x;
    double c_y = C.y - A.y;

    double det = b_x * c_y - b_y * c_x;

    // Degenerate triangle (should never happen)
    if (std::fabs(det) < 1e-15) {
        u = v = -1;
        return false;
    }

    double idet = 1.0 / det;

    // Solve 2×2 system
    u =  ( x * c_y - y * c_x) * idet;
    v = (-x * b_y + y * b_x) * idet;

    double w = 1.0 - u - v;

    // Standard closed triangle rule:
    // inside if u,v,w >= -eps
    const double eps = 1e-12;
    return (u >= -eps && v >= -eps && w >= -eps);
}

// ------- your GS code from precompute_jacobi_Q_triangles, slightly renamed -----

static int gram_schmidt_orthonormalize(const std::vector<double>& V,
                                       int P,int M,
                                       std::vector<double>& Q,
                                       double tol=1e-12)
{
  Q.assign(P*M,0.0);
  int rank=0;
  std::vector<double> qm(P);

  for (int m=0;m<M;++m) {
    for (int p=0;p<P;++p) qm[p] = V[p + m*P];

    for (int j=0;j<m;++j) {
      double norm_qj=0.0;
      for (int p=0;p<P;++p) norm_qj += Q[p + j*P]*Q[p + j*P];
      if (norm_qj < tol) continue;
      double r_jm=0.0;
      for (int p=0;p<P;++p) r_jm += Q[p + j*P]*qm[p];
      for (int p=0;p<P;++p) qm[p] -= r_jm*Q[p + j*P];
    }

    double nrm2=0.0;
    for (int p=0;p<P;++p) nrm2 += qm[p]*qm[p];
    if (nrm2 < tol) continue;
    double invnrm = 1.0/std::sqrt(nrm2);
    for (int p=0;p<P;++p) Q[p + m*P] = qm[p]*invnrm;
    rank++;
  }
  return rank;
}

// ------- precompute for all triangles up to max_level -------

int main()
{
  using Basis2 = jsimplex::Basis<2,double>;

  int p_max = 6;
  int M     = Basis2::dim_Pi(p_max);

  double kappa[3] = {0.5,0.5,0.5};
  std::vector<int>    alpha_table(M*2);
  std::vector<int>    tail_deg(M*2);
  std::vector<double> inv_h(M);

  Basis2::build_structures(
    kappa,p_max,
    alpha_table.data(),
    tail_deg.data(),
    inv_h.data()
  );

  // canonical 8x8 pixel centers in [0,1]^2
  const int Npix_side = 8;
  std::vector<Vec2> pix_xy(64);
  for (int i=0;i<8;++i)
    for (int j=0;j<8;++j) {
      double x=(i+0.5)/8.0;
      double y=(j+0.5)/8.0;
      pix_xy[i*8 + j] = {x,y};
    }

  int max_level = 3; // or 4; tune

  // We'll write everything in one text file
  std::ofstream out("jacobi_Q_tree_p6_L3.txt");
  if (!out) {
    std::cerr << "Cannot open output file.\n";
    return 1;
  }
  out << std::setprecision(17) << std::scientific;

  // global header
  out << "# p_max M max_level Npix_side\n";
  out << p_max << " " << M << " " << max_level << " " << Npix_side << "\n";

  // DFS over triangle tree
  std::vector<TriCode> stack;
  stack.emplace_back(0,0,0); // root 0
  stack.emplace_back(1,0,0); // root 1

  while (!stack.empty()) {
    TriCode tc = stack.back();
    stack.pop_back();

    Vec2 A,B,C;
    decode_triangle(tc,A,B,C);

    // collect pixels inside this triangle
    std::vector<int> pix_idx;
    std::vector<double> X_ref; // (xi,eta) pairs
    pix_idx.reserve(64);
    X_ref.reserve(64*2);

    for (int k = 0; k < 64; ++k)
    {
      const Vec2 &P = pix_xy[k];
      double x = P.x;
      double y = P.y;

      bool inside = false;
      double u = 0.0, v = 0.0;

      if (tc.level == 0)
      {
        // --- Asymmetric split at the top level ---
        // x + y < 1       -> root 0
        // x + y >= 1      -> root 1
        double s = x + y;

        if (tc.root == 0) {
          if (s < 1.0 - 1e-12) {
            inside = true;
          }
        } else { // tc.root == 1
          if (s >= 1.0 - 1e-12) {
            inside = true;
          }
        }

        if (inside) {
          // Map to reference coords for this root triangle
          barycentric(A, B, C, P, u, v);
        }
      }
      else
      {
        // --- Deeper levels: standard barycentric test ---
        if (barycentric(A, B, C, P, u, v)) {
          inside = true;
        }
      }

      if (inside) {
        pix_idx.push_back(k);
        X_ref.push_back(u);
        X_ref.push_back(v);
      }
    }
    int P = (int)pix_idx.size();
    if (P == 0) continue;

    // build V (P x M) using Basis2::eval_all
    std::vector<double> V(P*M);
    Basis2::eval_all(
      X_ref.data(),
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

    // orthonormalize columns -> Q (P x M)
    std::vector<double> Q;
    int rank = gram_schmidt_orthonormalize(V,P,M,Q,1e-12);
    std::cout << "Tri(root=" << (int)tc.root
              << ",level=" << tc.level
              << ",code=" << tc.code
              << "): P=" << P
              << " rank=" << rank << "\n";

    // ---------- write this triangle's block ----------
    // header for this triangle:
    // root level code P
    out << tc.root << " " << tc.level << " " << tc.code << " " << P << "\n";

    // pixel indices (0..63)
    for (int p=0;p<P;++p) {
      out << pix_idx[p];
      if (p+1<P) out << " ";
    }
    out << "\n";

    // Q matrix: P lines, each line M entries
    for (int p=0;p<P;++p) {
      for (int m=0;m<M;++m) {
        double v = Q[p + m*P];
        out << v;
        if (m+1<M) out << " ";
      }
      out << "\n";
    }

    // push children if not at max_level
    if (tc.level < max_level) {
      for (int c=0;c<4;++c)
        stack.push_back(child_code(tc,(uint8_t)c));
    }
  }

  std::cout << "Wrote jacobi_Q_tree_p6_L3.txt\n";
  return 0;
}
