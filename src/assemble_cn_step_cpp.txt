#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include "jbasis.hh"      // Jacobi simplex basis
#include <lapacke.h>      // LAPACKE_dgeqp3, dpotrf, dpotrs
#include <umfpack.h>      // UMFPACK direct solver
#include <cholmod.h>      // CHOLMOD for SPQR
#include <SuiteSparseQR.hpp>  // SuiteSparseQR

// ------------------------------- Basic types ---------------------------------

struct Mesh {
  int num_nodes = 0;
  int num_tets  = 0;
  std::vector<double> pts;   // 3 * num_nodes
  std::vector<int>    tets;  // 4 * num_tets

  int num_bfaces = 0;
  std::vector<int> bfaces_nodes; // 3 * num_bfaces
  std::vector<int> bfaces_tags;  // num_bfaces

  // per-tet affine data
  std::vector<double> tet_v0;    // 3 * num_tets
  std::vector<double> tet_A;     // 9 * num_tets
  std::vector<double> tet_detJ;  // num_tets
  std::vector<double> tet_invA;  // 9 * num_tets
  std::vector<double> tet_invAT; // 9 * num_tets
};

struct DofMap {
  int M;
  int num_tets;
  int num_dofs;
  DofMap(int M_, int num_tets_) : M(M_), num_tets(num_tets_) {
    num_dofs = M * num_tets;
  }
  inline int dof_index(int K, int m) const { return K * M + m; }
};

// ---------------------------- Quadrature loader ------------------------------

struct QuadRule {
  int dim = 0;
  int npts = 0;
  std::vector<double> x;  // npts * dim
  std::vector<double> w;  // npts
};

QuadRule load_quad(const std::string& fname, int dim) {
  std::ifstream ifs(fname.c_str());
  if (!ifs) throw std::runtime_error("Could not open quad file: " + fname);
  std::vector<double> data;
  double v;
  while (ifs >> v) data.push_back(v);
  if (data.empty()) throw std::runtime_error("Empty quad file: " + fname);
  int D = dim;
  int total = (int)data.size();
  if (total % (D+1) != 0)
    throw std::runtime_error("Quad file size not divisible by dim+1: " + fname);
  int N = total / (D+1);

  QuadRule q;
  q.dim  = D;
  q.npts = N;
  q.x.resize(N*D);
  q.w.resize(N);
  for (int i=0;i<N;++i)
    for (int j=0;j<D;++j)
      q.x[i*D + j] = data[i*D + j];
  for (int i=0;i<N;++i)
    q.w[i] = data[N*D + i];
  return q;
}

// ---------------------------- Reference basis --------------------------------

struct ReferenceBasis3D {
  int degree;
  int M;
  std::vector<int> alpha;
  std::vector<int> tail;
  std::vector<double> kappa;
  std::vector<double> inv_h;
};

ReferenceBasis3D make_ref_basis_3d(int degree) {
  ReferenceBasis3D rb;
  rb.degree = degree;
  rb.M = jsimplex::Basis<3,double>::dim_Pi(degree);
  rb.alpha.resize(rb.M * 3);
  rb.tail.resize(rb.M * 3);
  rb.kappa.assign(4, 0.5);        // Legendre-type kappa = 1/2
  rb.inv_h.resize(rb.M);

  jsimplex::Basis<3,double>::build_alpha_table(degree, rb.alpha.data());
  jsimplex::Basis<3,double>::build_tail_deg(
    degree, rb.alpha.data(), rb.tail.data());
  for (int m=0;m<rb.M;++m) {
    const int* arow = rb.alpha.data() + 3*m;
    rb.inv_h[m] = jsimplex::Basis<3,double>::inv_h_alpha(arow, rb.kappa.data());
  }
  return rb;
}

struct RefTetData {
  int M;
  int nq;
  std::vector<double> Vref;   // M * nq
  std::vector<double> dVref;  // M * nq * 3
};

RefTetData make_ref_tet_data(const ReferenceBasis3D& rb,
                             const QuadRule& tet_q) {
  RefTetData r;
  r.M  = rb.M;
  r.nq = tet_q.npts;
  const int D = 3;

  std::vector<double> Xref(r.nq * D);
  for (int q=0;q<r.nq;++q)
    for (int j=0;j<D;++j)
      Xref[q*D + j] = tet_q.x[q*D + j];

  r.Vref.resize(r.M * r.nq);
  r.dVref.resize(r.M * r.nq * D);

  jsimplex::Basis<3,double>::eval_all(
    Xref.data(), D, 1, r.nq,
    rb.kappa.data(), rb.degree,
    rb.alpha.data(), rb.tail.data(),
    rb.inv_h.data(),
    r.Vref.data(), r.nq,
    r.dVref.data()
  );
  return r;
}

// -----------------------------------------------------------------------------
// Write u evaluated at tetrahedral quadrature points to a VTK legacy file.
//
// We build a point cloud: each quad point in each tet becomes a VERTEX cell.
// Scalar value at each point is u_h(x_q) = sum_i u_{K,i} phi_i(r_q).
// -----------------------------------------------------------------------------
static void precompute_tet_affine(Mesh& mesh);
static void map_tet_point(const Mesh& mesh, int K,
                          const double r[3], double x[3]);
void write_vtk_tet_quad_solution(const std::string& filename,
                                 Mesh& mesh,
                                 const QuadRule& tet_q,
                                 const ReferenceBasis3D& rb,
                                 const std::vector<double>& u)
{
  // Ensure we have tet affine data for map_tet_point
  precompute_tet_affine(mesh);

  RefTetData rt = make_ref_tet_data(rb, tet_q);
  const int M   = rt.M;
  const int nq  = rt.nq;
  const int NT  = mesh.num_tets;

  if ((int)u.size() != NT * M) {
    std::cerr << "write_vtk_tet_quad_solution: size mismatch, "
              << "NT*M = " << NT * M << ", u.size() = " << u.size() << "\n";
    throw std::runtime_error("write_vtk_tet_quad_solution: size mismatch");
  }

  const int Npts = NT * nq;

  std::ofstream out(filename.c_str());
  if (!out) {
    throw std::runtime_error("Could not open VTK output file: " + filename);
  }

  out << "# vtk DataFile Version 2.0\n";
  out << "u at tet quadrature points\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  // ---------------- POINTS ----------------
  out << "POINTS " << Npts << " double\n";
  for (int K = 0; K < NT; ++K) {
    for (int q = 0; q < nq; ++q) {
      double r[3] = {
        tet_q.x[3*q + 0],
        tet_q.x[3*q + 1],
        tet_q.x[3*q + 2]
      };
      double x[3];
      map_tet_point(mesh, K, r, x); // uses mesh.tet_v0, tet_A

      out << x[0] << " " << x[1] << " " << x[2] << "\n";
    }
  }

  // ---------------- CELLS (VTK_VERTEX per quad point) ----------------
  out << "CELLS " << Npts << " " << 2 * Npts << "\n";
  for (int i = 0; i < Npts; ++i) {
    out << 1 << " " << i << "\n";
  }

  // Cell types: VTK_VERTEX = 1
  out << "CELL_TYPES " << Npts << "\n";
  for (int i = 0; i < Npts; ++i) {
    out << 1 << "\n";
  }

  // ---------------- SCALARS at each quad point ----------------
  out << "CELL_DATA " << Npts << "\n";
  out << "SCALARS u double 1\n";
  out << "LOOKUP_TABLE default\n";

  for (int K = 0; K < NT; ++K) {
    const double* uK = &u[K * M];
    for (int q = 0; q < nq; ++q) {
      double val = 0.0;
      // Vref is stored as phi_i(q) = Vref[q + i*nq]
      for (int i = 0; i < M; ++i) {
        double phi_i_q = rt.Vref[q + i * nq];
        val += uK[i] * phi_i_q;
      }
      out << val << "\n";
    }
  }
}

// ----------------------- Physics / BCs for glass bowl ------------------------

static const double Theta_plate = 1.0;
static const double Theta_inf   = 0.0;

static const double Bi_plate = 5.0;
static const double Bi_air   = 1.0;

static const double HeaterFlux = 1.0;

double source_s(const double x[3], double t)
{
  (void)x; (void)t;
  return 0.0;
}

double diffusivity_kappa(const double x[3])
{
  (void)x;
  return 1.0;
}

// ------------------------- Tet geometry utilities ----------------------------

static void precompute_tet_affine(Mesh& mesh) {
  int NT = mesh.num_tets;
  mesh.tet_v0.resize(3 * NT);
  mesh.tet_A.resize(9 * NT);
  mesh.tet_detJ.resize(NT);
  mesh.tet_invA.resize(9 * NT);
  mesh.tet_invAT.resize(9 * NT);

  for (int K=0; K<NT; ++K) {
    const int* tet = &mesh.tets[4*K];
    int v0_id = tet[0];
    int v1_id = tet[1];
    int v2_id = tet[2];
    int v3_id = tet[3];

    double v0[3], v1[3], v2[3], v3[3];
    for (int i=0;i<3;++i) {
      v0[i] = mesh.pts[3*v0_id + i];
      v1[i] = mesh.pts[3*v1_id + i];
      v2[i] = mesh.pts[3*v2_id + i];
      v3[i] = mesh.pts[3*v3_id + i];
    }

    mesh.tet_v0[3*K+0] = v0[0];
    mesh.tet_v0[3*K+1] = v0[1];
    mesh.tet_v0[3*K+2] = v0[2];

    double* A = &mesh.tet_A[9*K];
    for (int i=0;i<3;++i) {
      A[i*3+0] = v1[i] - v0[i];
      A[i*3+1] = v2[i] - v0[i];
      A[i*3+2] = v3[i] - v0[i];
    }

    double det =
      A[0]*(A[4]*A[8] - A[5]*A[7]) -
      A[1]*(A[3]*A[8] - A[5]*A[6]) +
      A[2]*(A[3]*A[7] - A[4]*A[6]);
    mesh.tet_detJ[K] = det;
    if (std::fabs(det) < 1e-14)
      throw std::runtime_error("Degenerate tet in precompute_tet_affine");

    double invdet = 1.0 / det;
    double* invA = &mesh.tet_invA[9*K];

    invA[0] =  (A[4]*A[8] - A[5]*A[7]) * invdet;
    invA[1] = -(A[1]*A[8] - A[2]*A[7]) * invdet;
    invA[2] =  (A[1]*A[5] - A[2]*A[4]) * invdet;

    invA[3] = -(A[3]*A[8] - A[5]*A[6]) * invdet;
    invA[4] =  (A[0]*A[8] - A[2]*A[6]) * invdet;
    invA[5] = -(A[0]*A[5] - A[2]*A[3]) * invdet;

    invA[6] =  (A[3]*A[7] - A[4]*A[6]) * invdet;
    invA[7] = -(A[0]*A[7] - A[1]*A[6]) * invdet;
    invA[8] =  (A[0]*A[4] - A[1]*A[3]) * invdet;

    double* invAT = &mesh.tet_invAT[9*K];
    for (int i=0;i<3;++i)
      for (int j=0;j<3;++j)
        invAT[i*3 + j] = invA[j*3 + i];
  }
}

static void map_tet_point(const Mesh& mesh, int K,
                          const double r[3], double x[3]) {
  const double* v0 = &mesh.tet_v0[3*K];
  const double* A  = &mesh.tet_A[9*K];
  for (int i=0;i<3;++i) {
    x[i] = v0[i] +
           A[i*3+0]*r[0] + A[i*3+1]*r[1] + A[i*3+2]*r[2];
  }
}

static void phys_to_ref(const Mesh& mesh, int K,
                        const double x[3], double r[3]) {
  const double* v0   = &mesh.tet_v0[3*K];
  const double* invA = &mesh.tet_invA[9*K];
  double dx[3] = {x[0]-v0[0], x[1]-v0[1], x[2]-v0[2]};
  for (int i=0;i<3;++i) {
    r[i] = invA[i*3+0]*dx[0] +
           invA[i*3+1]*dx[1] +
           invA[i*3+2]*dx[2];
  }
}

// ------------------------- Face geometry + adjacency -------------------------

struct FaceAdj {
  int num_faces = 0;
  std::vector<int> nodes;  // 3*num_faces
  std::vector<int> left;   // owning tet
  std::vector<int> right;  // neighbor tet or -1
  std::vector<int> tag;    // 0 interior, >0 boundary tag
};

static inline void sort3(int &a, int &b, int &c) {
  if (a > b) std::swap(a,b);
  if (b > c) std::swap(b,c);
  if (a > b) std::swap(a,b);
}

FaceAdj build_face_adjacency(const Mesh& mesh) {
  struct FaceRec { int a,b,c,tet; };
  std::vector<FaceRec> T;
  T.reserve(4 * mesh.num_tets);

  for (int K=0; K<mesh.num_tets; ++K) {
    const int* t = &mesh.tets[4*K];
    int v0=t[0], v1=t[1], v2=t[2], v3=t[3];
    int faces[4][3] = {
      {v0,v1,v2},
      {v0,v1,v3},
      {v0,v2,v3},
      {v1,v2,v3}
    };
    for (int lf=0; lf<4; ++lf) {
      int a=faces[lf][0], b=faces[lf][1], c=faces[lf][2];
      sort3(a,b,c);
      FaceRec fr{a,b,c,K};
      T.push_back(fr);
    }
  }

  std::sort(T.begin(), T.end(),
    [](const FaceRec& p, const FaceRec& q){
      if (p.a != q.a) return p.a < q.a;
      if (p.b != q.b) return p.b < q.b;
      return p.c < q.c;
    });

  struct BFace { int a,b,c,tag; };
  std::vector<BFace> B;
  B.reserve(mesh.num_bfaces);
  for (int f=0; f<mesh.num_bfaces; ++f) {
    int a = mesh.bfaces_nodes[3*f+0];
    int b = mesh.bfaces_nodes[3*f+1];
    int c = mesh.bfaces_nodes[3*f+2];
    sort3(a,b,c);
    BFace bf{a,b,c, mesh.bfaces_tags[f]};
    B.push_back(bf);
  }
  std::sort(B.begin(), B.end(),
    [](const BFace& p, const BFace& q){
      if (p.a != q.a) return p.a < q.a;
      if (p.b != q.b) return p.b < q.b;
      return p.c < q.c;
    });

  FaceAdj fa;
  int i=0, j=0;
  int Tn=(int)T.size(), Bn=(int)B.size();

  while (i < Tn) {
    int a=T[i].a, b=T[i].b, c=T[i].c;
    int tet0=T[i].tet, tet1=-1;
    int ii=i+1;
    if (ii<Tn && T[ii].a==a && T[ii].b==b && T[ii].c==c) {
      tet1=T[ii].tet; ++ii;
      if (ii<Tn && T[ii].a==a && T[ii].b==b && T[ii].c==c)
        throw std::runtime_error("Non-manifold face in adjacency");
    }

    while (j < Bn &&
           (B[j].a < a || (B[j].a==a && (B[j].b<b ||
           (B[j].b==b && B[j].c<c))))) {
      ++j;
    }

    int tag=0;
    if (j<Bn && B[j].a==a && B[j].b==b && B[j].c==c)
      tag = B[j].tag;

    fa.nodes.push_back(a);
    fa.nodes.push_back(b);
    fa.nodes.push_back(c);
    fa.left.push_back(tet0);
    fa.right.push_back(tet1);
    fa.tag.push_back(tag);
    ++fa.num_faces;
    i=ii;
  }
  return fa;
}

static void face_vertices(const Mesh& mesh, const FaceAdj& fa,
                          int f, double Xf[3][3]) {
  int a=fa.nodes[3*f+0], b=fa.nodes[3*f+1], c=fa.nodes[3*f+2];
  int ids[3]={a,b,c};
  for (int k=0;k<3;++k) {
    int vid = ids[k];
    Xf[k][0] = mesh.pts[3*vid+0];
    Xf[k][1] = mesh.pts[3*vid+1];
    Xf[k][2] = mesh.pts[3*vid+2];
  }
}

static double face_area(const double Xf[3][3]) {
  double v1[3], v2[3], cp[3];
  for (int i=0;i<3;++i) {
    v1[i]=Xf[1][i]-Xf[0][i];
    v2[i]=Xf[2][i]-Xf[0][i];
  }
  cp[0]=v1[1]*v2[2]-v1[2]*v2[1];
  cp[1]=v1[2]*v2[0]-v1[0]*v2[2];
  cp[2]=v1[0]*v2[1]-v1[1]*v2[0];
  double n = std::sqrt(cp[0]*cp[0]+cp[1]*cp[1]+cp[2]*cp[2]);
  return 0.5*n;
}

static void map_face_point(const double Xf[3][3],
                           double r, double s, double x[3]) {
  double mu0 = 1.0 - r - s;
  double mu1 = r;
  double mu2 = s;
  for (int i=0;i<3;++i)
    x[i] = mu0*Xf[0][i] + mu1*Xf[1][i] + mu2*Xf[2][i];
}

// ------------------------------ Sparse stuff ---------------------------------

struct Triplet {
  int row;
  int col;
  double val;
};

struct SparseCSC {
  int nrows = 0;
  int ncols = 0;
  std::vector<int> Ap;  // size ncols+1
  std::vector<int> Ai;  // size nnz
  std::vector<double> Ax; // size nnz
};

SparseCSC triplets_to_csc_rect(int nrows, int ncols,
                               const std::vector<Triplet>& T) {
  SparseCSC A;
  A.nrows = nrows;
  A.ncols = ncols;
  int nnz = (int)T.size();
  A.Ap.assign(ncols+1, 0);
  A.Ai.assign(nnz, 0);
  A.Ax.assign(nnz, 0.0);

  for (int k=0;k<nnz;++k) {
    int c = T[k].col;
    if (c < 0 || c >= ncols)
      throw std::runtime_error("triplets_to_csc_rect: bad col index");
    A.Ap[c+1]++;
  }
  for (int j=0;j<ncols;++j)
    A.Ap[j+1] += A.Ap[j];

  std::vector<int> next(A.Ap.begin(), A.Ap.end());
  for (int k=0;k<nnz;++k) {
    int r = T[k].row;
    int c = T[k].col;
    double v = T[k].val;
    if (r < 0 || r >= nrows)
      throw std::runtime_error("triplets_to_csc_rect: bad row index");
    int pos = next[c]++;
    A.Ai[pos] = r;
    A.Ax[pos] = v;
  }

  for (int j=0;j<ncols;++j) {
    int start = A.Ap[j];
    int end   = A.Ap[j+1];
    int len   = end - start;
    if (len <= 1) continue;

    for (int k=start+1; k<end; ++k) {
      int   ri = A.Ai[k];
      double rv = A.Ax[k];
      int p = k-1;
      while (p >= start && A.Ai[p] > ri) {
        A.Ai[p+1] = A.Ai[p];
        A.Ax[p+1] = A.Ax[p];
        --p;
      }
      A.Ai[p+1] = ri;
      A.Ax[p+1] = rv;
    }

    int w = start;
    for (int k=start; k<end; ++k) {
      if (w == start || A.Ai[k] != A.Ai[w-1]) {
        A.Ai[w] = A.Ai[k];
        A.Ax[w] = A.Ax[k];
        ++w;
      } else {
        A.Ax[w-1] += A.Ax[k];
      }
    }
    int shrink = end - w;
    if (shrink > 0) {
      for (int j2=j+1;j2<=ncols;++j2)
        A.Ap[j2] -= shrink;
    }
  }

  A.Ai.resize(A.Ap[ncols]);
  A.Ax.resize(A.Ap[ncols]);
  return A;
}

void csc_matvec(const SparseCSC& A,
                const std::vector<double>& x,
                std::vector<double>& y)
{
  int n = A.nrows;
  int m = A.ncols;
  if ((int)x.size() != m)
    throw std::runtime_error("csc_matvec: dimension mismatch");
  y.assign(n, 0.0);
  for (int j=0; j<m; ++j) {
    double xj = x[j];
    if (xj == 0.0) continue;
    int start = A.Ap[j];
    int end   = A.Ap[j+1];
    for (int p=start; p<end; ++p) {
      int i = A.Ai[p];
      y[i] += A.Ax[p] * xj;
    }
  }
}

void csc_matvec_transpose(const SparseCSC& A,
                          const std::vector<double>& x,
                          std::vector<double>& y)
{
  int n = A.nrows;
  int m = A.ncols;
  if ((int)x.size() != n)
    throw std::runtime_error("csc_matvec_transpose: dimension mismatch");
  y.assign(m, 0.0);
  for (int j=0; j<m; ++j) {
    int start = A.Ap[j];
    int end   = A.Ap[j+1];
    double sum = 0.0;
    for (int p=start; p<end; ++p) {
      int i = A.Ai[p];
      sum += A.Ax[p] * x[i];
    }
    y[j] = sum;
  }
}

// -------------------- Local face-wise QR (interpolatory ID) ------------------

static std::vector<int>
select_independent_rows_face(const std::vector<double>& Gpp,
                             const std::vector<double>& Gpm,
                             int Mloc)
{
  int mA = 2 * Mloc;
  int nA = Mloc;
  std::vector<double> A(mA * nA);

  for (int ell = 0; ell < Mloc; ++ell) {
    for (int r = 0; r < mA; ++r) {
      double val = 0.0;
      if (r < Mloc) {
        int i_plus = r;
        val = Gpp[ell * Mloc + i_plus];
      } else {
        int i_minus = r - Mloc;
        val = -Gpm[ell * Mloc + i_minus];
      }
      A[r + ell * mA] = val;
    }
  }

  std::vector<lapack_int> jpvt(nA, 0);
  std::vector<double> tau(std::min(mA, nA));

  lapack_int info = LAPACKE_dgeqp3(
    LAPACK_COL_MAJOR,
    (lapack_int)mA,
    (lapack_int)nA,
    A.data(),
    (lapack_int)mA,
    jpvt.data(),
    tau.data()
  );
  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgeqp3 failed in select_independent_rows_face");
  }

  int minmn = std::min(mA, nA);
  double max_diag = 0.0;
  for (int i = 0; i < minmn; ++i) {
    double rii = std::fabs(A[i + i*mA]);
    if (rii > max_diag) max_diag = rii;
  }

  double tol = 1e-10;
  int rank = 0;
  if (max_diag > 0.0) {
    for (int i = 0; i < minmn; ++i) {
      double rii = std::fabs(A[i + i*mA]);
      if (rii > tol * max_diag) ++rank;
    }
  } else {
    rank = 0;
  }

  struct Pair { int new_pos; int col; };
  std::vector<Pair> pivots;
  pivots.reserve(rank);
  for (int j = 0; j < nA; ++j) {
    int new_pos = (int)jpvt[j] - 1;
    if (0 <= new_pos && new_pos < rank) {
      pivots.push_back({new_pos, j});
    }
  }
  std::sort(pivots.begin(), pivots.end(),
            [](const Pair& a, const Pair& b){ return a.new_pos < b.new_pos; });

  std::vector<int> independent_ells;
  independent_ells.reserve(rank);
  for (const auto& p : pivots) {
    independent_ells.push_back(p.col);
  }
  return independent_ells;
}

// --------------------------- CN dual system container ------------------------

struct CNDualSystem {
  int Ndof = 0;
  int Nlam = 0;
  int num_faces = 0;
  int num_tets  = 0;
  int Mloc      = 0;

  SparseCSC K00;                // Ndof x Ndof (SPD)
  SparseCSC B;                  // Nlam x Ndof

  std::vector<Triplet> B_trip;  // reduced B in triplet form (for row access)

  std::vector<double> rhs_primal; // length Ndof
  std::vector<double> d;          // length Nlam

  std::vector<int> lam_face;      // length Nlam, which face each λ comes from
  std::vector<int> face_left;     // per face
  std::vector<int> face_right;    // per face

  std::vector<double> K_elem;     // per tet local K00 blocks, size num_tets*Mloc*Mloc
  // NEW: store per-tet mass and stiffness matrices so we can rebuild rhs each step
  std::vector<double> M_elem;     // size num_tets*Mloc*Mloc
  std::vector<double> A_elem;     // size num_tets*Mloc*Mloc

};

// ---------------------- UMFPACK-based K00 solver (reused) --------------------

struct K00Solver {
  const SparseCSC* A;
  void* Numeric;

  K00Solver(const SparseCSC& A_) : A(&A_), Numeric(nullptr) {
    int n = A->ncols;
    void* Symbolic = nullptr;
    int status = umfpack_di_symbolic(
      n, n,
      A->Ap.data(), A->Ai.data(), A->Ax.data(),
      &Symbolic, nullptr, nullptr);
    if (status != UMFPACK_OK) {
      umfpack_di_report_status(nullptr, status);
      throw std::runtime_error("UMFPACK symbolic factorization of K00 failed");
    }

    status = umfpack_di_numeric(
      A->Ap.data(), A->Ai.data(), A->Ax.data(),
      Symbolic, &Numeric, nullptr, nullptr);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
      umfpack_di_report_status(nullptr, status);
      Numeric = nullptr;
      throw std::runtime_error("UMFPACK numeric factorization of K00 failed");
    }
  }

  ~K00Solver() {
    if (Numeric) {
      umfpack_di_free_numeric(&Numeric);
      Numeric = nullptr;
    }
  }

  void solve(const std::vector<double>& b,
             std::vector<double>& x) const {
    int n = A->ncols;
    if ((int)b.size() != n)
      throw std::runtime_error("K00Solver::solve: rhs size mismatch");
    x.assign(n, 0.0);
    int status = umfpack_di_solve(
      UMFPACK_A,
      A->Ap.data(), A->Ai.data(), A->Ax.data(),
      x.data(), b.data(),
      Numeric, nullptr, nullptr);
    if (status != UMFPACK_OK) {
      umfpack_di_report_status(nullptr, status);
      throw std::runtime_error("UMFPACK solve for K00 failed");
    }
  }
};

// ------------- CG on Schur complement S (matrix-free, face-local precond) ----
//
// S = B K00^{-1} B^T, applied matrix-free via K00Solver and sparse matvecs.
// Preconditioner: for each interior face f, build
//   M_f ≈ B_f Kloc_f^{-1} B_f^T
// where Kloc_f is the 2-tet local block of K00, and B_f are λ-rows on face f.

std::vector<double> solve_schur_cg(const CNDualSystem& sys,
                                   const K00Solver& Ksolver,
                                   const std::vector<double>& bS,
                                   double tol, int maxit)
{
  int Nlam = sys.Nlam;
  int Ndof = sys.Ndof;
  int F    = sys.num_faces;
  int Mt   = sys.Mloc;
  int NT   = sys.num_tets;

  if ((int)bS.size() != Nlam)
    throw std::runtime_error("solve_schur_cg: bS size mismatch");
  if ((int)sys.lam_face.size() != Nlam)
    throw std::runtime_error("solve_schur_cg: lam_face size mismatch");
  if ((int)sys.face_left.size()  != F ||
      (int)sys.face_right.size() != F)
    throw std::runtime_error("solve_schur_cg: face_left/right size mismatch");

  auto dot = [](const std::vector<double>& a,
                const std::vector<double>& b) {
    double s = 0.0;
    int n = (int)a.size();
    for (int i=0;i<n;++i) s += a[i]*b[i];
    return s;
  };
  auto norm2 = [&](const std::vector<double>& a) {
    return std::sqrt(dot(a,a));
  };

  // Row-wise access for B via reduced triplets
  std::vector<std::vector<std::pair<int,double>>> Brow(Nlam);
  for (const auto& t : sys.B_trip) {
    if (t.row < 0 || t.row >= Nlam) continue;
    Brow[t.row].push_back({t.col, t.val});
  }

  // Group λ by face
  int max_face_id = -1;
  for (int l=0;l<Nlam;++l)
    if (sys.lam_face[l] > max_face_id) max_face_id = sys.lam_face[l];
  int Fused = std::max(F, max_face_id+1);

  std::vector<std::vector<int>> face_lams(Fused);
  for (int l=0;l<Nlam;++l) {
    int f = sys.lam_face[l];
    if (f < 0 || f >= Fused) continue;
    face_lams[f].push_back(l);
  }

  // Block sizes and offsets for M_f
  std::vector<int> block_dim(Fused, 0);
  std::vector<int> block_offset(Fused+1, 0);
  int total_entries = 0;
  for (int f=0; f<Fused; ++f) {
    int k = (int)face_lams[f].size();
    block_dim[f] = k;
    block_offset[f] = total_entries;
    total_entries += k*k;
  }
  block_offset[Fused] = total_entries;

  std::vector<double> blocks(total_entries, 0.0);

  // Build per-face M_f using local 2-tet K00 blocks
  for (int f=0; f<Fused; ++f) {
    int k = block_dim[f];
    if (k == 0) continue;
    int off = block_offset[f];

    if (f >= F) {
      // out-of-range face id from lam_face; skip
      for (int i=0;i<k;++i)
        for (int j=0;j<k;++j)
          blocks[off + i + j*k] = (i==j ? 1.0 : 0.0);
      continue;
    }

    int Kp = sys.face_left[f];
    int Km = sys.face_right[f]; // -1 for boundary faces

    int nloc = (Km >= 0 ? 2*Mt : Mt);
    if (Kp < 0 || Kp >= NT) {
      for (int i=0;i<k;++i)
        for (int j=0;j<k;++j)
          blocks[off + i + j*k] = (i==j ? 1.0 : 0.0);
      continue;
    }

    std::vector<double> Kloc(nloc*nloc, 0.0);
    const double* Kep = &sys.K_elem[(size_t)Kp*Mt*Mt];
    for (int i=0;i<Mt;++i) {
      for (int j=0;j<Mt;++j) {
        Kloc[i*nloc + j] = Kep[i*Mt + j];
      }
    }
    if (Km >= 0 && Km < NT) {
      const double* Kem = &sys.K_elem[(size_t)Km*Mt*Mt];
      for (int i=0;i<Mt;++i) {
        for (int j=0;j<Mt;++j) {
          Kloc[(Mt+i)*nloc + (Mt+j)] = Kem[i*Mt + j];
        }
      }
    }

    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nloc, Kloc.data(), nloc);
    if (info != 0) {
      std::cerr << "  Warning: dpotrf failed on Kloc for face " << f
                << " (nloc=" << nloc << ", info=" << info
                << "), using identity M_f.\n";
      for (int i=0;i<k;++i)
        for (int j=0;j<k;++j)
          blocks[off + i + j*k] = (i==j ? 1.0 : 0.0);
      continue;
    }

    std::vector<double> BfacT(nloc*k, 0.0);
    const std::vector<int>& lam_list = face_lams[f];

    int I0p = Kp*Mt;
    int I0m = (Km >= 0 ? Km*Mt : -1);

    for (int idx=0; idx<k; ++idx) {
      int lam = lam_list[idx];
      const auto& row = Brow[lam];
      for (auto& entry : row) {
        int col = entry.first;
        double val = entry.second;
        int jloc = -1;
        if (col >= I0p && col < I0p + Mt) {
          jloc = col - I0p;
        } else if (Km >= 0 && col >= I0m && col < I0m + Mt) {
          jloc = Mt + (col - I0m);
        } else {
          continue;
        }
        BfacT[jloc + idx*nloc] += val;
      }
    }

    std::vector<double> Y = BfacT;
    info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', nloc, k,
                          Kloc.data(), nloc, Y.data(), nloc);
    if (info != 0) {
      std::cerr << "  Warning: dpotrs failed on Kloc for face " << f
                << " (info=" << info << "), using identity M_f.\n";
      for (int i=0;i<k;++i)
        for (int j=0;j<k;++j)
          blocks[off + i + j*k] = (i==j ? 1.0 : 0.0);
      continue;
    }

    double* Mf = blocks.data() + off;
    for (int col=0; col<k; ++col) {
      for (int row=0; row<k; ++row) {
        double sum = 0.0;
        for (int j=0;j<nloc;++j) {
          double bjr = BfacT[j + row*nloc];
          double yjs = Y[j + col*nloc];
          sum += bjr * yjs;
        }
        Mf[row + col*k] = sum;
      }
    }

    for (int i=0;i<k;++i) {
      for (int j=i+1;j<k;++j) {
        double aij = Mf[i + j*k];
        double aji = Mf[j + i*k];
        double sym = 0.5*(aij+aji);
        Mf[i + j*k] = sym;
        Mf[j + i*k] = sym;
      }
    }

    info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', k, Mf, k);
    if (info != 0) {
      std::cerr << "  Warning: dpotrf failed on M_f for face " << f
                << " (k=" << k << ", info=" << info
                << "), using identity.\n";
      for (int i=0;i<k*k;++i) Mf[i] = 0.0;
      for (int i=0;i<k;++i) Mf[i + i*k] = 1.0;
    }
  }

  auto apply_precond = [&](const std::vector<double>& r,
                           std::vector<double>& z)
  {
    z.assign(Nlam, 0.0);
    for (int f=0; f<Fused; ++f) {
      int k = block_dim[f];
      if (k == 0) continue;
      int off = block_offset[f];
      double* Mf = blocks.data() + off;
      const std::vector<int>& lam_list = face_lams[f];
      if ((int)lam_list.size() != k) continue;

      std::vector<double> r_block(k);
      for (int i=0;i<k;++i) {
        int lam = lam_list[i];
        r_block[i] = r[lam];
      }

      int info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', k, 1,
                                Mf, k, r_block.data(), k);
      if (info != 0) {
        // fallback: identity on this block
      }

      for (int i=0;i<k;++i) {
        int lam = lam_list[i];
        z[lam] = r_block[i];
      }
    }
  };

  std::vector<double> lambda(Nlam, 0.0);
  std::vector<double> r = bS;

  double bnorm = norm2(bS);
  double bTol  = 1e-14;
  if (bnorm < bTol) {
    std::cout << "  CG-Schur: ||bS|| = " << bnorm
              << " < " << bTol << ", trivial lambda ≈ 0\n";
    return lambda;
  }

  std::vector<double> z(Nlam);
  apply_precond(r, z);

  std::vector<double> p = z;
  double rho_old = dot(r,z);

  std::vector<double> w(Ndof), zK(Ndof), Sp(Nlam);

  std::cout << "  CG-Schur (face-local): it 0, relres = 1\n";

  for (int it=1; it<=maxit; ++it) {
    csc_matvec_transpose(sys.B, p, w);
    Ksolver.solve(w, zK);
    csc_matvec(sys.B, zK, Sp);

    double pSp = dot(p, Sp);
    if (std::fabs(pSp) < 1e-30) {
      std::cerr << "  CG-Schur breakdown: p^T S p ~ 0\n";
      break;
    }

    double alpha = rho_old / pSp;
    for (int i=0;i<Nlam;++i)
      lambda[i] += alpha * p[i];
    for (int i=0;i<Nlam;++i)
      r[i] -= alpha * Sp[i];

    double relres = norm2(r) / bnorm;
    std::cout << "  CG-Schur (face-local): it " << it
              << ", relres = " << relres << "\n";

    if (relres < tol) break;

    apply_precond(r, z);

    double rho_new = dot(r,z);
    if (std::fabs(rho_old) < 1e-30) {
      std::cerr << "  CG-Schur breakdown: rho_old ~ 0\n";
      break;
    }
    double beta = rho_new / rho_old;
    for (int i=0;i<Nlam;++i)
      p[i] = z[i] + beta * p[i];
    rho_old = rho_new;
  }

  return lambda;
}

// --------------------------- CN assembly (dual) ------------------------------

CNDualSystem assemble_cn_step_dual(const Mesh& mesh_in,
                                   int degree,
                                   const QuadRule& tet_q,
                                   const QuadRule& tri_q,
                                   double t_n, double t_np1,
                                   double dt,
                                   const std::vector<double>& u_n)
{
  if (tet_q.dim != 3 || tri_q.dim != 2)
    throw std::runtime_error("quad dims must be (3,2)");

  Mesh mesh = mesh_in;
  precompute_tet_affine(mesh);
  FaceAdj fa = build_face_adjacency(mesh);

  // vertical extent
  double min_z =  1e300;
  double max_z = -1e300;
  for (int n = 0; n < mesh.num_nodes; ++n) {
    double z = mesh.pts[3*n + 2];
    if (z < min_z) min_z = z;
    if (z > max_z) max_z = z;
  }
  double H = max_z - min_z;
  if (H <= 0.0)
    throw std::runtime_error("Mesh has non-positive vertical extent");
  double eps_bottom = 0.02 * H;

  double max_r_bottom = 0.0;
  for (int n = 0; n < mesh.num_nodes; ++n) {
    double z = mesh.pts[3*n + 2];
    if (z - min_z <= eps_bottom) {
      double x = mesh.pts[3*n + 0];
      double y = mesh.pts[3*n + 1];
      double r = std::sqrt(x*x + y*y);
      if (r > max_r_bottom) max_r_bottom = r;
    }
  }
  double HeaterRadius = 0.5 * max_r_bottom;

  ReferenceBasis3D rb = make_ref_basis_3d(degree);
  RefTetData rt = make_ref_tet_data(rb, tet_q);
  int Mloc = rb.M;

  DofMap dofs(Mloc, mesh.num_tets);
  int Ndof = dofs.num_dofs;

  if (!u_n.empty() && (int)u_n.size() != Ndof)
    throw std::runtime_error("u_n size mismatch");

  std::vector<double> Rloc_all((size_t)mesh.num_tets * Mloc * Mloc, 0.0);
  auto Rloc = [&](int K, int i, int j) -> double& {
    return Rloc_all[(size_t)K * Mloc * Mloc + i * Mloc + j];
  };

  std::vector<double> K_elem_all((size_t)mesh.num_tets * Mloc * Mloc, 0.0);
  auto Kloc_tet = [&](int K, int i, int j) -> double& {
    return K_elem_all[(size_t)K * Mloc * Mloc + i * Mloc + j];
  };
  // NEW: per-tet mass and stiffness blocks
  std::vector<double> M_elem_all((size_t)mesh.num_tets * Mloc * Mloc, 0.0);
  auto Mloc_tet = [&](int K, int i, int j) -> double& {
    return M_elem_all[(size_t)K * Mloc * Mloc + i * Mloc + j];
  };

  std::vector<double> A_elem_all((size_t)mesh.num_tets * Mloc * Mloc, 0.0);
  auto Aloc_tet = [&](int K, int i, int j) -> double& {
    return A_elem_all[(size_t)K * Mloc * Mloc + i * Mloc + j];
  };


  std::vector<double> rhs_primal(Ndof, 0.0);
  std::vector<Triplet> Kp_trip;
  Kp_trip.reserve((size_t)mesh.num_tets * Mloc * Mloc);

  const int D = 3;
  int  nq_tet = rt.nq;
  int  nq_tri = tri_q.npts;
  double t_mid = 0.5*(t_n + t_np1);

  double total_qN = 0.0;
  double total_gR = 0.0;
  std::vector<int> tag_count(10, 0);

  // Robin / Neumann contributions
  for (int f=0; f<fa.num_faces; ++f) {
    if (fa.right[f] != -1) continue;

    int tag     = fa.tag[f];
    int K_owner = fa.left[f];

    if (tag >= 0 && tag < (int)tag_count.size())
      tag_count[tag]++;

    double Xf[3][3];
    face_vertices(mesh, fa, f, Xf);
    double JF = face_area(Xf);

    for (int q=0; q<nq_tri; ++q) {
      double rq = tri_q.x[q*2+0];
      double sq = tri_q.x[q*2+1];
      double wq = tri_q.w[q] * JF;
      double xq[3];
      map_face_point(Xf, rq, sq, xq);

      double a_val  = 0.0;
      double gR_val = 0.0;
      double qN_val = 0.0;

      const double z = xq[2];
      const double r = std::sqrt(xq[0]*xq[0] + xq[1]*xq[1]);
      const bool on_bottom = (z - min_z <= eps_bottom);

      if (tag == 1) {
        if (on_bottom) {
          if (r < HeaterRadius) {
            qN_val = HeaterFlux;
          } else {
            a_val  = Bi_plate;
            gR_val = Bi_plate * Theta_plate;
          }
        } else {
          a_val  = Bi_air;
          gR_val = Bi_air * Theta_inf;
        }
      } else if (tag == 2) {
        a_val  = Bi_air;
        gR_val = Bi_air * Theta_inf;
      } else {
        a_val  = 0.0;
        gR_val = 0.0;
        qN_val = 0.0;
      }

      total_qN += wq * qN_val;
      total_gR += wq * gR_val;

      double rK[3];
      phys_to_ref(mesh, K_owner, xq, rK);

      std::vector<double> Vp(Mloc);
      jsimplex::Basis<3,double>::eval_all(
        rK, 3, 1, 1,
        rb.kappa.data(), rb.degree,
        rb.alpha.data(), rb.tail.data(),
        rb.inv_h.data(),
        Vp.data(), 1,
        nullptr
      );

      for (int i = 0; i < Mloc; ++i) {
        int I = dofs.dof_index(K_owner, i);
        double phi_i = Vp[i];
        rhs_primal[I] += dt * wq * (qN_val + gR_val) * phi_i;
      }

      if (std::fabs(a_val) > 0.0) {
        for (int i = 0; i < Mloc; ++i) {
          double phi_i = Vp[i];
          for (int j = 0; j < Mloc; ++j) {
            double phi_j = Vp[j];
            double contrib = wq * a_val * phi_i * phi_j;
            Rloc(K_owner, i, j) += contrib;
          }
        }
      }
    }
  }

  std::cout << "Boundary tag counts:\n";
  for (int k=0;k<(int)tag_count.size(); ++k)
    std::cout << "  tag " << k << ": " << tag_count[k] << " faces\n";
  std::cout << "Total heater/plate load integrals:\n";
  std::cout << "  ∫ qN ds   ≈ " << total_qN << "\n";
  std::cout << "  ∫ gR ds   ≈ " << total_gR << "\n";

  // Volume assembly (M + 0.5 dt A + 0.5 dt R)
  for (int K=0; K<mesh.num_tets; ++K) {
    double detA = mesh.tet_detJ[K];
    double J = std::fabs(detA);
    const double* invAT = &mesh.tet_invAT[9*K];

    std::vector<double> MlocK(Mloc*Mloc, 0.0);
    std::vector<double> AlocK(Mloc*Mloc, 0.0);
    std::vector<double> flocK(Mloc, 0.0);

    for (int q=0;q<nq_tet;++q) {
      double wq = tet_q.w[q] * J;
      double rref[3] = {
        tet_q.x[q*3+0],
        tet_q.x[q*3+1],
        tet_q.x[q*3+2]
      };
      double xq[3];
      map_tet_point(mesh, K, rref, xq);

      double s_val     = source_s(xq, t_mid);
      double kappa_val = diffusivity_kappa(xq);

      std::vector<double> grad_phys(Mloc*3);
      for (int m=0;m<Mloc;++m) {
        const double* ghat = &rt.dVref[(q + m*nq_tet)*D];
        double ghx = ghat[0], ghy = ghat[1], ghz = ghat[2];
        grad_phys[m*3+0] =
          invAT[0]*ghx + invAT[1]*ghy + invAT[2]*ghz;
        grad_phys[m*3+1] =
          invAT[3]*ghx + invAT[4]*ghy + invAT[5]*ghz;
        grad_phys[m*3+2] =
          invAT[6]*ghx + invAT[7]*ghy + invAT[8]*ghz;
      }

      for (int i=0;i<Mloc;++i) {
        double phi_i = rt.Vref[q + i*nq_tet];
        flocK[i] += wq * s_val * phi_i;
        for (int j=0;j<Mloc;++j) {
          double phi_j = rt.Vref[q + j*nq_tet];
          MlocK[i*Mloc+j] += wq * phi_i * phi_j;
          double gx_i = grad_phys[i*3+0];
          double gy_i = grad_phys[i*3+1];
          double gz_i = grad_phys[i*3+2];
          double gx_j = grad_phys[j*3+0];
          double gy_j = grad_phys[j*3+1];
          double gz_j = grad_phys[j*3+2];
          double dotg = gx_i*gx_j + gy_i*gy_j + gz_i*gz_j;
          AlocK[i*Mloc+j] += wq * kappa_val * dotg;
        }
      }
    }

    for (int i=0;i<Mloc;++i) {
      int I = dofs.dof_index(K,i);
      double rhs_add = dt * flocK[i];
      if (!u_n.empty()) {
        double Miun=0.0, Aiun=0.0;
        for (int j=0;j<Mloc;++j) {
          int Jd = dofs.dof_index(K,j);
          Miun += MlocK[i*Mloc+j] * u_n[Jd];
          Aiun += AlocK[i*Mloc+j] * u_n[Jd];
        }
        rhs_add += Miun - 0.5*dt*Aiun;
      }
      rhs_primal[I] += rhs_add;
    }
    for (int i=0;i<Mloc;++i) {
      int I = dofs.dof_index(K,i);
      for (int j=0;j<Mloc;++j) {
        int Jd = dofs.dof_index(K,j);

        double Mij = MlocK[i*Mloc+j];
        double Aij = AlocK[i*Mloc+j];

        // Store per-tet M and A for later rhs updates
        Mloc_tet(K, i, j) = Mij;
        Aloc_tet(K, i, j) = Aij;

        double vol_val = Mij + 0.5*dt*Aij;
        double rob_val = 0.5*dt * Rloc(K, i, j);
        double val = vol_val + rob_val;

        if (std::fabs(val) > 0.0) {
          Kp_trip.push_back({I,Jd,val});
        }
        Kloc_tet(K,i,j) += val;
      }
    }
  }

  // Constraints
  std::vector<Triplet> Btrip;
  std::vector<double>  d_full;
  std::vector<int>     row_face;
  int m_rows = 0;
  const double tol_small = 1e-14;

  for (int f=0; f<fa.num_faces; ++f) {
    if (fa.right[f] == -1) continue;
    int Kp_tet = fa.left[f];
    int Km_tet = fa.right[f];

    double Xf[3][3];
    face_vertices(mesh, fa, f, Xf);
    double JF = face_area(Xf);

    std::vector<double> Gpp(Mloc*Mloc, 0.0);
    std::vector<double> Gpm(Mloc*Mloc, 0.0);

    for (int q=0;q<nq_tri;++q) {
      double rq = tri_q.x[q*2+0];
      double sq = tri_q.x[q*2+1];
      double wq = tri_q.w[q] * JF;

      double xq[3];
      map_face_point(Xf, rq, sq, xq);

      double r_plus[3], r_minus[3];
      phys_to_ref(mesh, Kp_tet, xq, r_plus);
      phys_to_ref(mesh, Km_tet, xq, r_minus);

      std::vector<double> Vp_plus(Mloc), Vp_minus(Mloc);
      jsimplex::Basis<3,double>::eval_all(
        r_plus, 3, 1, 1,
        rb.kappa.data(), rb.degree,
        rb.alpha.data(), rb.tail.data(),
        rb.inv_h.data(),
        Vp_plus.data(), 1,
        nullptr
      );
      jsimplex::Basis<3,double>::eval_all(
        r_minus, 3, 1, 1,
        rb.kappa.data(), rb.degree,
        rb.alpha.data(), rb.tail.data(),
        rb.inv_h.data(),
        Vp_minus.data(), 1,
        nullptr
      );

      for (int ell=0; ell<Mloc; ++ell) {
        double phi_ell_plus = Vp_plus[ell];
        for (int i=0;i<Mloc;++i) {
          double phi_i_plus  = Vp_plus[i];
          double phi_i_minus = Vp_minus[i];
          Gpp[ell*Mloc + i] += wq * phi_i_plus  * phi_ell_plus;
          Gpm[ell*Mloc + i] += wq * phi_i_minus * phi_ell_plus;
        }
      }
    }

    std::vector<int> independent_ells =
      select_independent_rows_face(Gpp, Gpm, Mloc);

    for (int idx = 0; idx < (int)independent_ells.size(); ++idx) {
      int ell = independent_ells[idx];
      int row = m_rows++;
      double rhs = 0.0;

      row_face.push_back(f);

      for (int i=0;i<Mloc;++i) {
        double val_p = Gpp[ell*Mloc + i];
        double val_m = Gpm[ell*Mloc + i];

        if (std::fabs(val_p) > tol_small) {
          int Iplus = dofs.dof_index(Kp_tet, i);
          Btrip.push_back({row, Iplus, val_p});
        }
        if (std::fabs(val_m) > tol_small) {
          int Iminus = dofs.dof_index(Km_tet, i);
          Btrip.push_back({row, Iminus, -val_m});
        }
      }
      d_full.push_back(rhs);
    }
  }
  int m_full = m_rows;

  // Row scaling of B and d_full to improve conditioning before SPQR.
  // Each constraint row i is scaled by 1 / max_j |B_ij|; RHS d_full is
  // scaled by the same factor.
  if (m_full > 0) {
    std::vector<double> row_norm(m_full, 0.0);
    for (const auto &bt : Btrip) {
      int r = bt.row;
      double a = std::fabs(bt.val);
      if (a > row_norm[r]) row_norm[r] = a;
    }
    const double floor_norm = 1e-12;
    for (int i = 0; i < m_full; ++i) {
      if (row_norm[i] < floor_norm) row_norm[i] = 1.0;
    }
    for (auto &bt : Btrip) {
      double s = 1.0 / row_norm[bt.row];
      bt.val *= s;
    }
    for (int i = 0; i < m_full; ++i) {
      double s = 1.0 / row_norm[i];
      d_full[i] *= s;
    }
  }

  SparseCSC B_csc = triplets_to_csc_rect(m_full, Ndof, Btrip);
  int nnzB = (int)B_csc.Ax.size();

  //int m_full = m_rows;

  //SparseCSC B_csc = triplets_to_csc_rect(m_full, Ndof, Btrip);
  //int nnzB = (int)B_csc.Ax.size();

  int Nlam = 0;
  std::vector<Triplet> B_trip_red;
  std::vector<double>  d_red;
  std::vector<int>     lam_face;

  if (m_full > 0) {
    cholmod_common Common, *cc = &Common;
    cholmod_l_start(cc);

    cholmod_sparse* Bch = cholmod_l_allocate_sparse(
      (SuiteSparse_long)m_full,
      (SuiteSparse_long)Ndof,
      (SuiteSparse_long)nnzB,
      1,
      1,
      0,
      CHOLMOD_REAL,
      cc
    );
    if (!Bch) throw std::runtime_error("cholmod_l_allocate_sparse(B) failed");

    SuiteSparse_long* Bp = (SuiteSparse_long*)Bch->p;
    SuiteSparse_long* Bi = (SuiteSparse_long*)Bch->i;
    double*           Bx = (double*)Bch->x;

    for (int j=0; j<=Ndof; ++j) Bp[j] = B_csc.Ap[j];
    for (int k=0; k<nnzB; ++k) {
      Bi[k] = B_csc.Ai[k];
      Bx[k] = B_csc.Ax[k];
    }
    Bch->nrow  = m_full;
    Bch->ncol  = Ndof;
    Bch->nzmax = nnzB;

    cholmod_sparse* Ach = cholmod_l_transpose(Bch, 1, cc);
    if (!Ach) throw std::runtime_error("cholmod_l_transpose(B) failed");

    int ordering = SPQR_ORDERING_DEFAULT;
    double tol   = SPQR_DEFAULT_TOL;
    SuiteSparse_long econ = (SuiteSparse_long)Ach->nrow;
    int getCTX = 0;

    cholmod_sparse *Zsparse = nullptr, *Rch = nullptr, *Hch = nullptr;
    cholmod_dense  *Zdense  = nullptr, *HTau = nullptr;
    SuiteSparse_long *E = nullptr, *HPinv = nullptr;

    SuiteSparse_long rank = SuiteSparseQR<double>(
      ordering, tol, econ, getCTX,
      Ach,
      nullptr,
      nullptr,
      &Zsparse,
      &Zdense,
      &Rch,
      &E,
      &Hch,
      &HPinv,
      &HTau,
      cc
    );
    if (rank < 0 || E == nullptr)
      throw std::runtime_error("SuiteSparseQR failed on B^T");

    Nlam = (int)rank;
    std::cout << "  Global SPQR on B^T: rank = " << rank
              << " nlam = " << Nlam << " out of m_full = " << m_full
              << " constraints\n";

    std::vector<int> row_keep_map(m_full, -1);
    for (int k=0; k<Nlam; ++k) {
      int r = (int)E[k];
      if (r < 0 || r >= m_full)
        throw std::runtime_error("SPQR returned bad row index");
      row_keep_map[r] = k;
    }

    B_trip_red.reserve(Btrip.size());
    d_red.assign(Nlam, 0.0);
    lam_face.assign(Nlam, -1);

    for (const auto &bt : Btrip) {
      int old_row = bt.row;
      int new_row = row_keep_map[old_row];
      if (new_row < 0) continue;
      B_trip_red.push_back({new_row, bt.col, bt.val});
    }
    for (int k=0; k<Nlam; ++k) {
      int old_row = (int)E[k];
      d_red[k]    = d_full[old_row];
      lam_face[k] = row_face[old_row];
    }

    if (Zsparse) cholmod_l_free_sparse(&Zsparse, cc);
    if (Zdense)  cholmod_l_free_dense(&Zdense, cc);
    if (Rch)     cholmod_l_free_sparse(&Rch, cc);
    if (Hch)     cholmod_l_free_sparse(&Hch, cc);
    if (HTau)    cholmod_l_free_dense(&HTau, cc);
    if (HPinv)   SuiteSparse_free(HPinv);
    if (E)       SuiteSparse_free(E);
    cholmod_l_free_sparse(&Ach, cc);
    cholmod_l_free_sparse(&Bch, cc);
    cholmod_l_finish(cc);
  } else {
    Nlam = 0;
    B_trip_red.clear();
    d_red.clear();
    lam_face.clear();
  }

  // ---------------- ROW SCALING OF B (and d) ------------------
  if (Nlam > 0) {
    std::vector<double> row_norm2(Nlam, 0.0);
    for (const auto &bt : B_trip_red) {
      int r = bt.row;
      double v = bt.val;
      row_norm2[r] += v*v;
    }
    std::vector<double> scale(Nlam, 1.0);
    for (int i = 0; i < Nlam; ++i) {
      if (row_norm2[i] > 0.0) {
        scale[i] = 1.0 / std::sqrt(row_norm2[i]);
      } else {
        scale[i] = 1.0;
      }
    }
    // Apply scaling to B_trip_red and d_red
    for (auto &bt : B_trip_red) {
      bt.val *= scale[bt.row];
    }
    for (int i = 0; i < Nlam; ++i) {
      d_red[i] *= scale[i];
    }
  }
  // ------------------------------------------------------------

  SparseCSC K00_csc  = triplets_to_csc_rect(Ndof, Ndof, Kp_trip);

  std::vector<double> diag(Ndof, 0.0);
  for (int j=0; j<K00_csc.ncols; ++j) {
    for (int p = K00_csc.Ap[j]; p < K00_csc.Ap[j+1]; ++p) {
      int i = K00_csc.Ai[p];
      if (i == j) diag[i] += K00_csc.Ax[p];
    }
  }
  double min_diag = 1e300, max_diag = -1e300, sum_diag = 0.0;
  int nz_diag = 0;
  for (int i=0;i<Ndof;++i) {
    if (diag[i] != 0.0) {
      nz_diag++;
      if (diag[i] < min_diag) min_diag = diag[i];
      if (diag[i] > max_diag) max_diag = diag[i];
      sum_diag += diag[i];
    }
  }
  std::cout << "K00 diag stats:\n"
            << "  nz_diag       = " << nz_diag << " / " << Ndof << "\n"
            << "  min diag      = " << min_diag << "\n"
            << "  max diag      = " << max_diag << "\n"
            << "  avg diag(nz)  = " << (nz_diag ? sum_diag/nz_diag : 0.0) << "\n";

  SparseCSC Bred_csc = triplets_to_csc_rect(Nlam, Ndof, B_trip_red);

  CNDualSystem sys;
  sys.Ndof       = Ndof;
  sys.Nlam       = Nlam;
  sys.num_faces  = fa.num_faces;
  sys.num_tets   = mesh.num_tets;
  sys.Mloc       = Mloc;
  sys.K00        = std::move(K00_csc);
  sys.B          = std::move(Bred_csc);
  sys.B_trip     = std::move(B_trip_red);
  sys.rhs_primal = std::move(rhs_primal);
  sys.d          = std::move(d_red);
  sys.lam_face   = std::move(lam_face);
  sys.face_left  = fa.left;
  sys.face_right = fa.right;
  sys.K_elem     = std::move(K_elem_all);
  sys.M_elem     = std::move(M_elem_all);  // NEW
  sys.A_elem     = std::move(A_elem_all);  // NEW
  return sys;
}

// ------------------------- VTK mesh loaders ----------------------------------

Mesh load_volume_mesh_vtk(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs) throw std::runtime_error("Cannot open volume VTK: " + filename);

  std::string token;
  Mesh mesh;
  int num_pts=0, num_cells=0;
  std::vector<std::vector<int>> cells;
  std::vector<int> cell_types;

  while (ifs >> token) {
    if (token == "POINTS") {
      std::string type_str;
      ifs >> num_pts >> type_str;
      mesh.num_nodes = num_pts;
      mesh.pts.resize(3*num_pts);
      for (int i=0;i<3*num_pts;++i) ifs >> mesh.pts[i];
    } else if (token == "CELLS") {
      int total_ints;
      ifs >> num_cells >> total_ints;
      cells.resize(num_cells);
      for (int c=0;c<num_cells;++c) {
        int k; ifs >> k;
        cells[c].resize(k);
        for (int j=0;j<k;++j) ifs >> cells[c][j];
      }
    } else if (token == "CELL_TYPES") {
      int nct; ifs >> nct;
      cell_types.resize(nct);
      for (int c=0;c<nct;++c) ifs >> cell_types[c];
    }
  }
  ifs.close();

  if (num_pts==0 || num_cells==0 || cell_types.empty())
    throw std::runtime_error("Volume VTK missing data");

  for (int c=0;c<num_cells;++c) {
    if (cell_types[c]==10 && cells[c].size()==4) {
      mesh.tets.push_back(cells[c][0]);
      mesh.tets.push_back(cells[c][1]);
      mesh.tets.push_back(cells[c][2]);
      mesh.tets.push_back(cells[c][3]);
    }
  }
  mesh.num_tets = (int)mesh.tets.size()/4;
  if (mesh.num_tets == 0)
    throw std::runtime_error("No tets found in volume VTK");
  return mesh;
}

void load_boundary_from_vtk(const std::string& filename, Mesh& mesh) {
  std::ifstream ifs(filename.c_str());
  if (!ifs) throw std::runtime_error("Cannot open boundary VTK: " + filename);

  std::string token;
  int num_pts=0, num_cells=0;
  std::vector<std::vector<int>> cells;
  std::vector<int> cell_types;
  std::vector<int> cell_tags;
  bool have_cell_data=false;
  int cell_data_count=0;

  while (ifs >> token) {
    if (token=="POINTS") {
      std::string type_str;
      ifs >> num_pts >> type_str;
      std::vector<double> dummy(3*num_pts);
      for (int i=0;i<3*num_pts;++i) ifs >> dummy[i];
    } else if (token=="CELLS") {
      int total_ints;
      ifs >> num_cells >> total_ints;
      cells.resize(num_cells);
      for (int c=0;c<num_cells;++c) {
        int k; ifs >> k;
        cells[c].resize(k);
        for (int j=0;j<k;++j) ifs >> cells[c][j];
      }
    } else if (token=="CELL_TYPES") {
      int nct; ifs >> nct;
      cell_types.resize(nct);
      for (int c=0;c<nct;++c) ifs >> cell_types[c];
    } else if (token=="CELL_DATA") {
      ifs >> cell_data_count;
      have_cell_data = true;
      std::string scalars,name,type_str,lookup,lutname;
      int ncomp;
      ifs >> scalars >> name >> type_str >> ncomp;
      ifs >> lookup >> lutname;
      cell_tags.resize(cell_data_count);
      for (int i=0;i<cell_data_count;++i) ifs >> cell_tags[i];
    }
  }
  ifs.close();

  if (num_cells==0 || cell_types.empty() || !have_cell_data)
    throw std::runtime_error("Boundary VTK missing cells/types/data");
  if ((int)cell_tags.size() != num_cells)
    throw std::runtime_error("Boundary VTK CELL_DATA size mismatch");

  for (int c=0;c<num_cells;++c) {
    if (cell_types[c]==5 && cells[c].size()==3) {
      mesh.bfaces_nodes.push_back(cells[c][0]);
      mesh.bfaces_nodes.push_back(cells[c][1]);
      mesh.bfaces_nodes.push_back(cells[c][2]);
      mesh.bfaces_tags.push_back(cell_tags[c]);
    }
  }
  mesh.num_bfaces = (int)mesh.bfaces_nodes.size()/3;
  if (mesh.num_bfaces==0)
    throw std::runtime_error("No boundary triangles in boundary VTK");
}

// Write a cell-wise scalar solution (one value per tet) in VTK legacy format.
// We take the constant-mode coefficient on each tet: u[K*Mloc + 0].
void write_vtk_cell_solution(const std::string& filename,
                             const Mesh& mesh,
                             const std::vector<double>& u,
                             int Mloc)
{
  if (mesh.num_tets * Mloc != (int)u.size()) {
    std::cerr << "write_vtk_cell_solution: size mismatch, "
              << "num_tets*Mloc = " << mesh.num_tets * Mloc
              << ", u.size() = "   << u.size() << "\n";
    throw std::runtime_error("write_vtk_cell_solution: size mismatch");
  }

  std::ofstream out(filename.c_str());
  if (!out) {
    throw std::runtime_error("Could not open VTK output file: " + filename);
  }

  out << "# vtk DataFile Version 2.0\n";
  out << "CN solution\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  // Points
  out << "POINTS " << mesh.num_nodes << " double\n";
  for (int i = 0; i < mesh.num_nodes; ++i) {
    double x = mesh.pts[3*i + 0];
    double y = mesh.pts[3*i + 1];
    double z = mesh.pts[3*i + 2];
    out << x << " " << y << " " << z << "\n";
  }

  // Tetra cells
  int num_tets = mesh.num_tets;
  out << "CELLS " << num_tets << " " << num_tets * 5 << "\n";
  for (int K = 0; K < num_tets; ++K) {
    int v0 = mesh.tets[4*K + 0];
    int v1 = mesh.tets[4*K + 1];
    int v2 = mesh.tets[4*K + 2];
    int v3 = mesh.tets[4*K + 3];
    out << 4 << " " << v0 << " " << v1 << " " << v2 << " " << v3 << "\n";
  }

  // Cell types: VTK_TETRA = 10
  out << "CELL_TYPES " << num_tets << "\n";
  for (int K = 0; K < num_tets; ++K) {
    out << 10 << "\n";
  }

  // Cell data: one scalar per tet, from constant mode
  out << "CELL_DATA " << num_tets << "\n";
  out << "SCALARS u double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int K = 0; K < num_tets; ++K) {
    double uK = u[K * Mloc + 0];  // constant basis coefficient on tet K
    out << uK << "\n";
  }
}


// ----------------------------------- main ------------------------------------

int main(int argc, char** argv)
{
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0]
              << " volume.vtk boundary_bfaces.vtk tet_quad.txt tri_quad.txt degree dt nsteps\n";
    return EXIT_FAILURE;
  }

  std::string vol_vtk   = argv[1];
  std::string bnd_vtk   = argv[2];
  std::string tet_qfile = argv[3];
  std::string tri_qfile = argv[4];
  int degree            = std::atoi(argv[5]);
  double dt             = std::atof(argv[6]);

  try {
    Mesh mesh = load_volume_mesh_vtk(vol_vtk);
    load_boundary_from_vtk(bnd_vtk, mesh);

    std::cout << "Volume mesh: nodes=" << mesh.num_nodes
              << ", tets=" << mesh.num_tets << "\n";
    std::cout << "Boundary faces: " << mesh.num_bfaces << "\n";

    QuadRule tet_q = load_quad(tet_qfile, 3);
    QuadRule tri_q = load_quad(tri_qfile, 2);
    std::cout << "Tet quad: npts=" << tet_q.npts << "\n";
    std::cout << "Tri quad: npts=" << tri_q.npts << "\n";

    ReferenceBasis3D rb = make_ref_basis_3d(degree);
    int Mloc = rb.M;
    int Ndof = Mloc * mesh.num_tets;
    std::cout << "Degree=" << degree << ", Mloc=" << Mloc
              << ", Ndof=" << Ndof << "\n";

    std::vector<double> u_n(Ndof, 0.0);

    double t0 = 0.0;
    double t1 = dt;

    std::cout << "Assembling CN dual system (K00 + B + rhs)...\n";
    CNDualSystem sys = assemble_cn_step_dual(
      mesh, degree, tet_q, tri_q, t0, t1, dt, u_n);

    std::cout << "Dual system: Ndof=" << sys.Ndof
              << ", Nlam=" << sys.Nlam
              << ", nnz(K00)=" << sys.K00.Ap.back()
              << ", nnz(B)="   << sys.B.Ap.back() << "\n";
    K00Solver Ksolver(sys.K00);

    // Time-stepping parameters
    int nsteps = 10;
    if (argc >= 8) {
      nsteps = std::atoi(argv[7]);
      if (nsteps < 1) nsteps = 1;
    }
    std::cout << "Time-stepping: nsteps = " << nsteps
              << ", dt = " << dt << "\n";

    std::vector<double> u_np1(sys.Ndof, 0.0);
    // u_n was initialized to zero above and will be updated each step

    for (int step = 0; step < nsteps; ++step) {
      double t_n   = step * dt;
      double t_np1 = (step + 1) * dt;
      double t_mid = 0.5 * (t_n + t_np1);

      (void)t_mid; // currently unused, but kept for future time-dependent source

      // --- Build rhs for this step: rhs = rhs_0 + (M - 0.5 dt A) * u_n ---
      std::vector<double> rhs(sys.Ndof);
      rhs = sys.rhs_primal; // static part (source + boundary loads for u_n = 0)

      const int Mloc_step = sys.Mloc;
      const int NT_step   = sys.num_tets;

      for (int K = 0; K < NT_step; ++K) {
        for (int i = 0; i < Mloc_step; ++i) {
          int I = K * Mloc_step + i;
          double Miun = 0.0;
          double Aiun = 0.0;
          for (int j = 0; j < Mloc_step; ++j) {
            int J = K * Mloc_step + j;
            double Mij = sys.M_elem[(size_t)K*Mloc_step*Mloc_step + i*Mloc_step + j];
            double Aij = sys.A_elem[(size_t)K*Mloc_step*Mloc_step + i*Mloc_step + j];
            Miun += Mij * u_n[J];
            Aiun += Aij * u_n[J];
          }
          rhs[I] += Miun - 0.5 * dt * Aiun;
        }
      }

      // --- Solve constrained CN system with updated rhs ---
      if (sys.Nlam == 0) {
        std::cout << "Step " << step
                  << ": no constraints, solving K00 u = rhs directly...\n";
        Ksolver.solve(rhs, u_np1);
      } else {
        // Build Schur RHS bS = B K00^{-1} rhs - d
        std::vector<double> z_f, bS(sys.Nlam);
        Ksolver.solve(rhs, z_f);
        csc_matvec(sys.B, z_f, bS);
        for (int l = 0; l < sys.Nlam; ++l)
          bS[l] -= sys.d[l];

        auto dot = [](const std::vector<double>& a,
                      const std::vector<double>& b)
        {
          double s = 0.0;
          int n = (int)a.size();
          for (int i = 0; i < n; ++i) s += a[i] * b[i];
          return s;
        };
        auto norm2 = [&](const std::vector<double>& a)
        {
          return std::sqrt(dot(a, a));
        };

        std::vector<double> Bt_bS(sys.Ndof);
        csc_matvec_transpose(sys.B, bS, Bt_bS);

        double bS_norm    = norm2(bS);
        double Bt_bS_norm = norm2(Bt_bS);

        std::cout << "Step " << step << ":\n";
        std::cout << "  Debug: ||bS||_2      = " << bS_norm    << "\n";
        std::cout << "         ||B^T bS||_2 = " << Bt_bS_norm << "\n";
        std::cout << "         ratio = ||B^T bS|| / ||bS|| = "
                  << (Bt_bS_norm / (bS_norm + 1e-300)) << "\n\n";

        std::cout << "Solving Schur complement S lambda = bS with CG...\n";
        double tol   = 1e-8;
        int    maxit = 1000;
        std::vector<double> lambda = solve_schur_cg(
          sys, Ksolver, bS, tol, maxit);

        std::vector<double> Btlam(sys.Ndof), w(sys.Ndof);
        csc_matvec_transpose(sys.B, lambda, Btlam);
        for (int i = 0; i < sys.Ndof; ++i)
          w[i] = rhs[i] - Btlam[i];

        std::cout << "Solving K00 u = rhs - B^T lambda...\n";
        Ksolver.solve(w, u_np1);
      }

      // --- Diagnostics for this step ---
      double norm_u2 = 0.0;
      for (int i = 0; i < sys.Ndof; ++i)
        norm_u2 += u_np1[i] * u_np1[i];
      norm_u2 = std::sqrt(norm_u2);

      std::cout << "Step " << step
                << ", t = [" << t_n << ", " << t_np1 << "]"
                << ", ||u^{n+1}||_2 = " << std::setprecision(17) << norm_u2
                << "\n";

      // --- Write VTK for this step ---
      try {
        std::ostringstream oss;
        oss << "cn_tet_quad_step_"
            << std::setw(4) << std::setfill('0') << step
            << ".vtk";
        std::string out_vtk = oss.str();
        write_vtk_tet_quad_solution(out_vtk, mesh, tet_q, rb, u_np1);
        std::cout << "  Wrote tet-quadrature solution to " << out_vtk << "\n";
      } catch (const std::exception& ex_vtk) {
        std::cerr << "Warning: failed to write VTK solution at step "
                  << step << ": " << ex_vtk.what() << "\n";
      }

      // Prepare for next step
      u_n = u_np1;
    }

    // ------------------------------------------------------------------------
    // Optional debug: assemble full KKT and solve directly with UMFPACK
    // to compare against Schur–CG solution u_np1.
    // ------------------------------------------------------------------------
    {
      int Ndof = sys.Ndof;
      int Nlam = sys.Nlam;
      int Ntot = Ndof + Nlam;

      if (Ntot < 200000 && nsteps == 1) {
        std::cout << "\n[DEBUG] Solving full KKT with UMFPACK for comparison...\n";

        std::vector<Triplet> K_trip;
        K_trip.reserve(sys.K00.Ap.back() + 2*sys.B.Ap.back());

        for (int j = 0; j < Ndof; ++j) {
          for (int p = sys.K00.Ap[j]; p < sys.K00.Ap[j+1]; ++p) {
            int i   = sys.K00.Ai[p];
            double v = sys.K00.Ax[p];
            K_trip.push_back({i, j, v});
          }
        }

        for (int j = 0; j < Ndof; ++j) {
          for (int p = sys.B.Ap[j]; p < sys.B.Ap[j+1]; ++p) {
            int rowB  = sys.B.Ai[p];
            double v  = sys.B.Ax[p];
            int iK    = j;
            int jK    = Ndof + rowB;
            K_trip.push_back({iK, jK, v});
          }
        }

        for (int j = 0; j < Ndof; ++j) {
          for (int p = sys.B.Ap[j]; p < sys.B.Ap[j+1]; ++p) {
            int rowB  = sys.B.Ai[p];
            double v  = sys.B.Ax[p];
            int iK    = Ndof + rowB;
            int jK    = j;
            K_trip.push_back({iK, jK, v});
          }
        }

        SparseCSC K_csc = triplets_to_csc_rect(Ntot, Ntot, K_trip);
        int nnzK = K_csc.Ap.back();
        std::cout << "[DEBUG] KKT: Ntot=" << Ntot
                  << ", nnz=" << nnzK << "\n";

        std::vector<double> rhsKKT(Ntot, 0.0);
        for (int i = 0; i < Ndof; ++i)
          rhsKKT[i] = sys.rhs_primal[i];
        for (int l = 0; l < Nlam; ++l)
          rhsKKT[Ndof + l] = sys.d[l];

        void* Symbolic = nullptr;
        void* Numeric  = nullptr;
        int status = umfpack_di_symbolic(
          Ntot, Ntot,
          K_csc.Ap.data(), K_csc.Ai.data(), K_csc.Ax.data(),
          &Symbolic, nullptr, nullptr
        );
        if (status != UMFPACK_OK) {
          umfpack_di_report_status(nullptr, status);
          std::cerr << "[DEBUG] UMFPACK symbolic factorization of KKT failed\n";
        } else {
          status = umfpack_di_numeric(
            K_csc.Ap.data(), K_csc.Ai.data(), K_csc.Ax.data(),
            Symbolic, &Numeric, nullptr, nullptr
          );
          umfpack_di_free_symbolic(&Symbolic);

          if (status != UMFPACK_OK) {
            umfpack_di_report_status(nullptr, status);
            std::cerr << "[DEBUG] UMFPACK numeric factorization of KKT failed\n";
          } else {
            std::vector<double> xKKT(Ntot, 0.0);
            status = umfpack_di_solve(
              UMFPACK_A,
              K_csc.Ap.data(), K_csc.Ai.data(), K_csc.Ax.data(),
              xKKT.data(), rhsKKT.data(),
              Numeric, nullptr, nullptr
            );
            umfpack_di_free_numeric(&Numeric);

            if (status != UMFPACK_OK) {
              umfpack_di_report_status(nullptr, status);
              std::cerr << "[DEBUG] UMFPACK solve of KKT failed\n";
            } else {
              std::vector<double> u_direct(Ndof);
              for (int i = 0; i < Ndof; ++i)
                u_direct[i] = xKKT[i];

              auto dot = [](const std::vector<double>& a,
                            const std::vector<double>& b) {
                double s = 0.0;
                int n = (int)a.size();
                for (int i = 0; i < n; ++i) s += a[i]*b[i];
                return s;
              };
              auto norm2 = [&](const std::vector<double>& v) {
                return std::sqrt(dot(v,v));
              };

              std::vector<double> diff(Ndof);
              for (int i = 0; i < Ndof; ++i)
                diff[i] = u_np1[i] - u_direct[i];

              double norm_ud   = norm2(u_direct);
              double norm_diff = norm2(diff);
              double rel_err   = norm_diff / (norm_ud + 1e-30);

              std::cout << "[DEBUG] ||u_direct||_2    = " << norm_ud   << "\n";
              std::cout << "[DEBUG] ||u_schur - u_direct||_2 = " << norm_diff << "\n";
              std::cout << "[DEBUG] relative error (Schur vs full KKT) = "
                        << rel_err << "\n";
            }
          }
        }
      } else {
        std::cout << "\n[DEBUG] Skipping full KKT UMFPACK solve (Ntot too large: "
                  << (sys.Ndof + sys.Nlam) << ")\n";
      }
    }

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

