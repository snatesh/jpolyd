// brick_tet_kernel.cpp

#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>

// adjust this to your actual header name / include path
#include <jbasis.hh>

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
                    const Tet &K)
{
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
                        const Tet &K)
{
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


bool load_tet_quadrature(const std::string &filename,
                         std::vector<Vec3> &Xref,
                         std::vector<double> &W)
{
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

struct TetBasis3D {
  int p;         // total degree
  int M;         // dim_Pi(p)
  std::vector<int> alpha_table;
  std::vector<int> tail_deg;
  std::vector<double> inv_h;
  std::vector<double> Vref; // size M * Nq (layout: V[p + m*ld_V])

  int ld_V;      // leading dimension (Nq)

  void init(int degree,
            const double *kappa,
            const std::vector<Vec3> &Xref)
  {
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

    // pack reference quadrature nodes into raw array for eval_all
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

  // φ_m( Xref[q] )
  inline double phi(int m, int q) const {
    return Vref[q + m * ld_V];
  }
};

//double test_function(const Vec3 &x)
//{
//  return std::sin(M_PI * x.x) * std::cos(M_PI * x.y) * x.z;
//}
double test_function(const Vec3 &x)
{
  return std::pow(x.x,9) + std::pow(x.y,9) + std::pow(x.z,9);
}

int main()
{
  // 1) Load quadrature on reference tet
  std::vector<Vec3> Xref;
  std::vector<double> W;

  // TODO: adjust filename to your actual quadrature file
  std::string quadFile =
    "t3_kappa0.5-0.5-0.5-0.5_N364_M1330_n11_m18.txt";

  if (!load_tet_quadrature(quadFile, Xref, W)) {
    return 1;
  }

  int Nq = static_cast<int>(Xref.size());
  std::cout << "Loaded " << Nq << " quadrature points.\n";

  // 2) Build 3D Jacobi basis (kappa = 0.5 for uniform weight)
  double kappa[4] = {0.5, 0.5, 0.5, 0.5};
  int degree = 9;  // e.g. total degree p=4

  TetBasis3D basis;
  basis.init(degree, kappa, Xref);

  std::cout << "3D basis dim M = " << basis.M << "\n";

  double total_err2 = 0.0;
  double total_u2   = 0.0;

  // 3) For each tet: project and accumulate L2 error and L2 norm
  for (int kt = 0; kt < 6; ++kt) {
    const Tet &K = cubeTets[kt];

    // |J_K| for physical L2 integral on this tet
    double Jdet = tet_jacobian_det(cubeVerts, K);

    // a) compute coefficients c_m^K
    std::vector<double> coeffs(basis.M, 0.0);

    for (int q = 0; q < Nq; ++q) {
      Vec3 r = Xref[q];
      Vec3 x_phys = map_ref_to_tet(r, cubeVerts, K);

      double u_q = test_function(x_phys);
      double w_q = W[q];

      for (int m = 0; m < basis.M; ++m) {
        coeffs[m] += w_q * u_q * basis.phi(m, q);
      }
    }

    // b) evaluate approximation at quadrature nodes and accumulate error
    for (int q = 0; q < Nq; ++q) {
      Vec3 r = Xref[q];
      Vec3 x_phys = map_ref_to_tet(r, cubeVerts, K);

      double u_q = test_function(x_phys);

      double uh_q = 0.0;
      for (int m = 0; m < basis.M; ++m) {
        uh_q += coeffs[m] * basis.phi(m, q);
      }

      double diff = u_q - uh_q;

      total_err2 += Jdet * W[q] * diff * diff;
      total_u2   += Jdet * W[q] * u_q * u_q;
    }
  }

  double L2_err = std::sqrt(total_err2);
  double L2_u   = std::sqrt(total_u2);
  double rel_err = L2_err / (L2_u + 1e-15);

  std::cout << "L2(u)      ≈ " << L2_u   << "\n";
  std::cout << "L2(u - uh) ≈ " << L2_err << "\n";
  std::cout << "Relative L2 error ≈ " << rel_err << "\n";

  return 0;
}

