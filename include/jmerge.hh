#ifndef JMERGE_HH
#define JMERGE_HH

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <lapacke.h>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <jdetail.hh>
#include <jnode.hh>

namespace jsimplex {
namespace hps_detail {

template<class Real>
struct LapackGesv;

template<>
struct LapackGesv<float>
{
  static inline lapack_int run(
    lapack_int n,
    lapack_int nrhs,
    float* A,
    lapack_int lda,
    lapack_int* ipiv,
    float* B,
    lapack_int ldb)
  {
    return LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
  }
};

template<>
struct LapackGesv<double>
{
  static inline lapack_int run(
    lapack_int n,
    lapack_int nrhs,
    double* A,
    lapack_int lda,
    lapack_int* ipiv,
    double* B,
    lapack_int ldb)
  {
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, B, ldb);
  }
};

inline std::vector<int> expand_face_positions(
  const std::vector<int>& face_pos,
  int kf
)
{
  if (kf <= 0)
  {
    throw std::invalid_argument("expand_face_positions: kf must be positive");
  }

  std::vector<int> idx;
  idx.reserve((std::size_t)face_pos.size() * kf);

  for (int p : face_pos)
  {
    if (p < 0)
    {
      throw std::invalid_argument("expand_face_positions: negative face position");
    }
    for (int r = 0; r < kf; ++r)
    {
      idx.push_back(p * kf + r);
    }
  }

  return idx;
}

template<int D>
inline std::map<DSimplexFaceKey<D>,int> make_face_position_map(
  const std::vector<DSimplexFaceKey<D>>& faces
)
{
  std::map<DSimplexFaceKey<D>,int> pos;
  for (int i = 0; i < static_cast<int>(faces.size()); ++i)
  {
    const auto ins = pos.insert(std::make_pair(faces[(std::size_t)i], i));
    if (!ins.second)
    {
      throw std::runtime_error("make_face_position_map: duplicate face key in node boundary");
    }
  }
  return pos;
}

template<class Real>
inline void extract_matrix_colmajor(
  const std::vector<Real>& A,
  int A_rows,
  const std::vector<int>& row_idx,
  const std::vector<int>& col_idx,
  std::vector<Real>& out
)
{
  if (A_rows <= 0)
  {
    throw std::invalid_argument("extract_matrix_colmajor: A_rows must be positive");
  }
  if (A.size() % static_cast<std::size_t>(A_rows) != 0)
  {
    throw std::invalid_argument("extract_matrix_colmajor: inconsistent A size");
  }

  const int A_cols = static_cast<int>(A.size() / static_cast<std::size_t>(A_rows));
  const int m = static_cast<int>(row_idx.size());
  const int n = static_cast<int>(col_idx.size());
  out.assign((std::size_t)m * n, Real(0));

  for (int j = 0; j < n; ++j)
  {
    const int cj = col_idx[(std::size_t)j];
    if (cj < 0 || cj >= A_cols)
    {
      throw std::out_of_range("extract_matrix_colmajor: column index out of range");
    }

    for (int i = 0; i < m; ++i)
    {
      const int ri = row_idx[(std::size_t)i];
      if (ri < 0 || ri >= A_rows)
      {
        throw std::out_of_range("extract_matrix_colmajor: row index out of range");
      }

      out[(std::size_t)i + (std::size_t)m * j] =
        A[(std::size_t)ri + (std::size_t)A_rows * cj];
    }
  }
}

template<class Real>
inline void extract_vector(
  const std::vector<Real>& x,
  const std::vector<int>& idx,
  std::vector<Real>& out
)
{
  out.assign(idx.size(), Real(0));
  for (int i = 0; i < static_cast<int>(idx.size()); ++i)
  {
    const int k = idx[(std::size_t)i];
    if (k < 0 || k >= static_cast<int>(x.size()))
    {
      throw std::out_of_range("extract_vector: index out of range");
    }
    out[(std::size_t)i] = x[(std::size_t)k];
  }
}

template<class Real>
inline void add_in_place(std::vector<Real>& y, const std::vector<Real>& x)
{
  if (y.size() != x.size())
  {
    throw std::invalid_argument("add_in_place: size mismatch");
  }
  for (std::size_t i = 0; i < y.size(); ++i)
  {
    y[i] += x[i];
  }
}

template<class Real>
inline void scale_in_place(std::vector<Real>& y, Real alpha)
{
  for (Real& v : y)
  {
    v *= alpha;
  }
}

template<class Real>
inline void dense_solve_in_place(
  int n,
  int nrhs,
  const std::vector<Real>& H_in,
  std::vector<Real>& B_inout
)
{
  static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value,
                "dense_solve_in_place supports Real=float or Real=double");

  if (n <= 0)
  {
    throw std::invalid_argument("dense_solve_in_place: n must be positive");
  }
  if (nrhs < 0)
  {
    throw std::invalid_argument("dense_solve_in_place: nrhs must be nonnegative");
  }
  if (static_cast<int>(H_in.size()) != n * n)
  {
    throw std::invalid_argument("dense_solve_in_place: H has wrong size");
  }
  if (static_cast<int>(B_inout.size()) != n * nrhs)
  {
    throw std::invalid_argument("dense_solve_in_place: RHS has wrong size");
  }
  if (nrhs == 0)
  {
    return;
  }

  std::vector<Real> H = H_in;
  std::vector<lapack_int> ipiv((std::size_t)n, 0);

  const lapack_int info = LapackGesv<Real>::run(
    (lapack_int)n,
    (lapack_int)nrhs,
    H.data(),
    (lapack_int)n,
    ipiv.data(),
    B_inout.data(),
    (lapack_int)n);

  if (info != 0)
  {
    throw std::runtime_error("dense_solve_in_place: LAPACK gesv failed or matrix is singular");
  }
}

template<class Real>
inline std::vector<Real> matmul_colmajor(
  const std::vector<Real>& A,
  int m,
  int k,
  const std::vector<Real>& B,
  int n
)
{
  if (m < 0 || k < 0 || n < 0)
  {
    throw std::invalid_argument("matmul_colmajor: negative dimension");
  }
  if (static_cast<int>(A.size()) != m * k)
  {
    throw std::invalid_argument("matmul_colmajor: A size mismatch");
  }
  if (static_cast<int>(B.size()) != k * n)
  {
    throw std::invalid_argument("matmul_colmajor: B size mismatch");
  }

  std::vector<Real> C((std::size_t)m * n, Real(0));
  if (m == 0 || k == 0 || n == 0)
  {
    return C;
  }

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    m,
    n,
    k,
    Real(1),
    A.data(),
    m,
    B.data(),
    k,
    Real(0),
    C.data(),
    m);

  return C;
}

template<class Real>
inline void set_block_colmajor(
  std::vector<Real>& A,
  int A_rows,
  int row0,
  int col0,
  const std::vector<Real>& B,
  int B_rows,
  int B_cols
)
{
  if (A_rows <= 0 || row0 < 0 || col0 < 0 || B_rows < 0 || B_cols < 0)
  {
    throw std::invalid_argument("set_block_colmajor: invalid dimensions");
  }
  if (static_cast<int>(B.size()) != B_rows * B_cols)
  {
    throw std::invalid_argument("set_block_colmajor: B size mismatch");
  }
  if (A.size() % static_cast<std::size_t>(A_rows) != 0)
  {
    throw std::invalid_argument("set_block_colmajor: A size mismatch");
  }
  const int A_cols = static_cast<int>(A.size() / static_cast<std::size_t>(A_rows));
  if (row0 + B_rows > A_rows || col0 + B_cols > A_cols)
  {
    throw std::out_of_range("set_block_colmajor: block out of range");
  }

  for (int j = 0; j < B_cols; ++j)
  {
    for (int i = 0; i < B_rows; ++i)
    {
      A[(std::size_t)(row0 + i) + (std::size_t)A_rows * (col0 + j)] =
        B[(std::size_t)i + (std::size_t)B_rows * j];
    }
  }
}

template<class Real>
inline void axpy_matrix_in_place(
  std::vector<Real>& Y,
  const std::vector<Real>& X,
  Real alpha = Real(1)
)
{
  if (Y.size() != X.size())
  {
    throw std::invalid_argument("axpy_matrix_in_place: size mismatch");
  }
  for (std::size_t i = 0; i < Y.size(); ++i)
  {
    Y[i] += alpha * X[i];
  }
}

} // namespace hps_detail

/*
  Merge two HPS nodes A and B.

  Contract:
    A and B expose mu = S lambda + b on their ordered boundary_faces.

  The shared interface is boundary_faces(A) intersection boundary_faces(B).
  Parent boundary ordering is

    [ A exterior faces in A order, B exterior faces in B order ].

  The returned node stores S_parent, b_parent and the reconstruction formula

    gamma = R_A lambda_AE + R_B lambda_BE + r.
*/
template<int D, class Real>
inline Node<D,Real> merge_nodes(
  const Node<D,Real>& A,
  const Node<D,Real>& B,
  int child_A_id = -1,
  int child_B_id = -1
)
{
  A.validate();
  B.validate();

  if (A.kf != B.kf)
  {
    throw std::invalid_argument("merge_nodes: child kf mismatch");
  }

  const int kf = A.kf;
  using FaceKey = DSimplexFaceKey<D>;

  const auto posA = hps_detail::make_face_position_map<D>(A.boundary_faces);
  const auto posB = hps_detail::make_face_position_map<D>(B.boundary_faces);

  std::vector<FaceKey> EA_faces;
  std::vector<FaceKey> EB_faces;
  std::vector<FaceKey> I_faces;
  std::vector<int> A_E_pos;
  std::vector<int> A_I_pos;
  std::vector<int> B_E_pos;
  std::vector<int> B_I_pos;

  for (int i = 0; i < static_cast<int>(A.boundary_faces.size()); ++i)
  {
    const FaceKey& key = A.boundary_faces[(std::size_t)i];
    const auto itB = posB.find(key);
    if (itB == posB.end())
    {
      EA_faces.push_back(key);
      A_E_pos.push_back(i);
    }
    else
    {
      I_faces.push_back(key);
      A_I_pos.push_back(i);
      B_I_pos.push_back(itB->second);
    }
  }

  for (int i = 0; i < static_cast<int>(B.boundary_faces.size()); ++i)
  {
    const FaceKey& key = B.boundary_faces[(std::size_t)i];
    if (posA.find(key) == posA.end())
    {
      EB_faces.push_back(key);
      B_E_pos.push_back(i);
    }
  }

  if (I_faces.empty())
  {
    throw std::invalid_argument("merge_nodes: children do not share a boundary face");
  }

  const std::vector<int> A_E_idx = hps_detail::expand_face_positions(A_E_pos, kf);
  const std::vector<int> A_I_idx = hps_detail::expand_face_positions(A_I_pos, kf);
  const std::vector<int> B_E_idx = hps_detail::expand_face_positions(B_E_pos, kf);
  const std::vector<int> B_I_idx = hps_detail::expand_face_positions(B_I_pos, kf);

  const int nAE = static_cast<int>(A_E_idx.size());
  const int nBE = static_cast<int>(B_E_idx.size());
  const int nI = static_cast<int>(A_I_idx.size());

  if (static_cast<int>(B_I_idx.size()) != nI)
  {
    throw std::runtime_error("merge_nodes: inconsistent interface scalar sizes");
  }

  const int nbA = A.nb();
  const int nbB = B.nb();

  std::vector<Real> SA_EE, SA_EI, SA_IE, SA_II;
  std::vector<Real> SB_EE, SB_EI, SB_IE, SB_II;
  std::vector<Real> bA_E, bA_I, bB_E, bB_I;

  hps_detail::extract_matrix_colmajor(A.S, nbA, A_E_idx, A_E_idx, SA_EE);
  hps_detail::extract_matrix_colmajor(A.S, nbA, A_E_idx, A_I_idx, SA_EI);
  hps_detail::extract_matrix_colmajor(A.S, nbA, A_I_idx, A_E_idx, SA_IE);
  hps_detail::extract_matrix_colmajor(A.S, nbA, A_I_idx, A_I_idx, SA_II);

  hps_detail::extract_matrix_colmajor(B.S, nbB, B_E_idx, B_E_idx, SB_EE);
  hps_detail::extract_matrix_colmajor(B.S, nbB, B_E_idx, B_I_idx, SB_EI);
  hps_detail::extract_matrix_colmajor(B.S, nbB, B_I_idx, B_E_idx, SB_IE);
  hps_detail::extract_matrix_colmajor(B.S, nbB, B_I_idx, B_I_idx, SB_II);

  hps_detail::extract_vector(A.b, A_E_idx, bA_E);
  hps_detail::extract_vector(A.b, A_I_idx, bA_I);
  hps_detail::extract_vector(B.b, B_E_idx, bB_E);
  hps_detail::extract_vector(B.b, B_I_idx, bB_I);

  std::vector<Real> H = SA_II;
  hps_detail::add_in_place(H, SB_II);

  std::vector<Real> R_A = SA_IE;
  hps_detail::dense_solve_in_place<Real>(nI, nAE, H, R_A);
  hps_detail::scale_in_place(R_A, Real(-1));

  std::vector<Real> R_B = SB_IE;
  hps_detail::dense_solve_in_place<Real>(nI, nBE, H, R_B);
  hps_detail::scale_in_place(R_B, Real(-1));

  // Precompute the source-only interface inverse and its exterior transfers.
  // These depend only on the child DtN operators, not on the current source.
  std::vector<Real> R_f((std::size_t)nI * nI, Real(0));
  for (int i = 0; i < nI; ++i)
  {
    R_f[(std::size_t)i + (std::size_t)nI * i] = Real(1);
  }
  hps_detail::dense_solve_in_place<Real>(nI, nI, H, R_f);
  hps_detail::scale_in_place(R_f, Real(-1));

  std::vector<Real> P_A =
    hps_detail::matmul_colmajor(SA_EI, nAE, nI, R_f, nI);
  std::vector<Real> P_B =
    hps_detail::matmul_colmajor(SB_EI, nBE, nI, R_f, nI);

  std::vector<Real> source_interface = bA_I;
  hps_detail::add_in_place(source_interface, bB_I);
  std::vector<Real> r =
    hps_detail::matmul_colmajor(R_f, nI, nI, source_interface, 1);

  std::vector<Real> TL = SA_EE;
  std::vector<Real> prod = hps_detail::matmul_colmajor(SA_EI, nAE, nI, R_A, nAE);
  hps_detail::axpy_matrix_in_place(TL, prod);

  std::vector<Real> TR = hps_detail::matmul_colmajor(SA_EI, nAE, nI, R_B, nBE);
  std::vector<Real> BL = hps_detail::matmul_colmajor(SB_EI, nBE, nI, R_A, nAE);

  std::vector<Real> BR = SB_EE;
  prod = hps_detail::matmul_colmajor(SB_EI, nBE, nI, R_B, nBE);
  hps_detail::axpy_matrix_in_place(BR, prod);

  const int nP = nAE + nBE;
  Node<D,Real> P;
  P.kf = kf;
  P.boundary_faces.reserve(EA_faces.size() + EB_faces.size());
  P.boundary_faces.insert(P.boundary_faces.end(), EA_faces.begin(), EA_faces.end());
  P.boundary_faces.insert(P.boundary_faces.end(), EB_faces.begin(), EB_faces.end());
  P.S.assign((std::size_t)nP * nP, Real(0));
  P.b.assign((std::size_t)nP, Real(0));

  hps_detail::set_block_colmajor(P.S, nP, 0, 0, TL, nAE, nAE);
  hps_detail::set_block_colmajor(P.S, nP, 0, nAE, TR, nAE, nBE);
  hps_detail::set_block_colmajor(P.S, nP, nAE, 0, BL, nBE, nAE);
  hps_detail::set_block_colmajor(P.S, nP, nAE, nAE, BR, nBE, nBE);

  std::vector<Real> btop = bA_E;
  std::vector<Real> brcorr =
    hps_detail::matmul_colmajor(P_A, nAE, nI, source_interface, 1);
  hps_detail::add_in_place(btop, brcorr);

  std::vector<Real> bbot = bB_E;
  brcorr =
    hps_detail::matmul_colmajor(P_B, nBE, nI, source_interface, 1);
  hps_detail::add_in_place(bbot, brcorr);

  for (int i = 0; i < nAE; ++i)
  {
    P.b[(std::size_t)i] = btop[(std::size_t)i];
  }
  for (int i = 0; i < nBE; ++i)
  {
    P.b[(std::size_t)nAE + i] = bbot[(std::size_t)i];
  }

  P.is_leaf = false;
  P.is_merge = true;
  P.merge.child_A = child_A_id;
  P.merge.child_B = child_B_id;
  P.merge.EA_faces = EA_faces;
  P.merge.EB_faces = EB_faces;
  P.merge.I_faces = I_faces;
  P.merge.A_E_idx = A_E_idx;
  P.merge.A_I_idx = A_I_idx;
  P.merge.B_E_idx = B_E_idx;
  P.merge.B_I_idx = B_I_idx;
  P.merge.R_A = R_A;
  P.merge.R_B = R_B;
  P.merge.r = r;
  P.merge.R_f = R_f;
  P.merge.P_A = P_A;
  P.merge.P_B = P_B;
  P.merge.nAE = nAE;
  P.merge.nBE = nBE;
  P.merge.nI = nI;

  P.validate();
  return P;
}

template<int D, class Real>
inline void update_merge_source(
  Node<D,Real>& parent,
  const Node<D,Real>& A,
  const Node<D,Real>& B)
{
  parent.validate();
  A.validate();
  B.validate();

  if (!parent.is_merge)
  {
    throw std::invalid_argument(
      "update_merge_source: parent is not a merge node");
  }

  auto& md = parent.merge;
  if (parent.kf != A.kf || parent.kf != B.kf)
  {
    throw std::invalid_argument(
      "update_merge_source: child kf mismatch");
  }

  std::vector<Real> bA_E, bA_I, bB_E, bB_I;
  hps_detail::extract_vector(A.b, md.A_E_idx, bA_E);
  hps_detail::extract_vector(A.b, md.A_I_idx, bA_I);
  hps_detail::extract_vector(B.b, md.B_E_idx, bB_E);
  hps_detail::extract_vector(B.b, md.B_I_idx, bB_I);

  std::vector<Real> source_interface = bA_I;
  hps_detail::add_in_place(source_interface, bB_I);

  md.r = hps_detail::matmul_colmajor(
    md.R_f, md.nI, md.nI, source_interface, 1);

  std::vector<Real> btop = bA_E;
  std::vector<Real> correction = hps_detail::matmul_colmajor(
    md.P_A, md.nAE, md.nI, source_interface, 1);
  hps_detail::add_in_place(btop, correction);

  std::vector<Real> bbot = bB_E;
  correction = hps_detail::matmul_colmajor(
    md.P_B, md.nBE, md.nI, source_interface, 1);
  hps_detail::add_in_place(bbot, correction);

  parent.b.assign((std::size_t)(md.nAE + md.nBE), Real(0));
  std::copy(btop.begin(), btop.end(), parent.b.begin());
  std::copy(
    bbot.begin(),
    bbot.end(),
    parent.b.begin() + md.nAE);

  parent.validate();
}

/*
  High-order PHYSICAL co-normal observations on selected exterior faces.
  Face rows are in faces[] order, kH consecutive modes per face.

    q_H = H * lambda + z.

  H: (faces.size()*kH) x node.nb(), column-major.
  z: faces.size()*kH.
  Wf: (faces.size()*kH) x leaf.source_dim, leaf nodes only.

  This is deliberately separate from Node: the standard degree-n HPS
  skeleton, numerical flux, and merge algebra are unchanged.
*/
template<int D, class Real>
struct HighConormalObservation
{
  using FaceKey = DSimplexFaceKey<D>;

  int kH = 0;
  std::vector<FaceKey> faces;
  std::vector<Real> H;
  std::vector<Real> z;
  std::vector<Real> Wf;

  int nrows() const { return kH * static_cast<int>(faces.size()); }
};

/* Build the leaf observation from the already tested Q_H*Ulam, Q_H*Uf. */
template<int D, class Real>
inline HighConormalObservation<D,Real> make_high_conormal_leaf_observation(
  const Node<D,Real>& leaf,
  int kH,
  const std::vector<int>& selected_local_faces,
  const std::vector<Real>& H,
  const std::vector<Real>& Wf)
{
  if (!leaf.is_leaf || kH < 0 ||
      static_cast<int>(selected_local_faces.size()) > leaf.nfaces())
    throw std::invalid_argument("high leaf observation: invalid leaf/faces");

  HighConormalObservation<D,Real> out;
  out.kH = kH;
  for (int f : selected_local_faces)
  {
    if (f < 0 || f >= leaf.nfaces() ||
        std::find(out.faces.begin(), out.faces.end(),
                  leaf.boundary_faces[(std::size_t)f]) != out.faces.end())
      throw std::invalid_argument("high leaf observation: invalid/duplicate face");
    out.faces.push_back(leaf.boundary_faces[(std::size_t)f]);
  }
  const int nh = out.nrows();
  if (H.size() != (std::size_t)nh * leaf.nb() ||
      Wf.size() != (std::size_t)nh * leaf.source_dim)
    throw std::invalid_argument("high leaf observation: matrix size mismatch");

  out.H = H;
  out.Wf = Wf;
  out.z.assign((std::size_t)nh, Real(0));
  return out;
}

/* Refresh only the source-dependent leaf offset; H and Wf remain fixed. */
template<int D, class Real>
inline void update_high_conormal_leaf_source(
  HighConormalObservation<D,Real>& obs,
  const Node<D,Real>& leaf,
  const Real* f_int)
{
  if (!leaf.is_leaf ||
      obs.H.size() != (std::size_t)obs.nrows() * leaf.nb() ||
      obs.Wf.size() != (std::size_t)obs.nrows() * leaf.source_dim ||
      (leaf.source_dim > 0 && obs.nrows() > 0 && !f_int))
    throw std::invalid_argument("high leaf source: incompatible observation/source");

  obs.z.assign((std::size_t)obs.nrows(), Real(0));
  for (int j = 0; j < leaf.source_dim; ++j)
    for (int i = 0; i < obs.nrows(); ++i)
      obs.z[(std::size_t)i] +=
        obs.Wf[(std::size_t)i + (std::size_t)obs.nrows() * j] * f_int[j];
}

namespace high_conormal_detail {

template<int D, class Real>
inline int face_slot(
  const HighConormalObservation<D,Real>& obs,
  const DSimplexFaceKey<D>& key)
{
  const auto it = std::find(obs.faces.begin(), obs.faces.end(), key);
  return it == obs.faces.end() ? -1 : static_cast<int>(it - obs.faces.begin());
}

template<int D, class Real>
inline void check_child(
  const Node<D,Real>& child,
  const HighConormalObservation<D,Real>& obs,
  const std::vector<DSimplexFaceKey<D>>& exterior)
{
  if (obs.kH < 0 ||
      obs.H.size() != (std::size_t)obs.nrows() * child.nb() ||
      obs.z.size() != (std::size_t)obs.nrows())
    throw std::invalid_argument("high merge: child observation size mismatch");
  for (std::size_t i = 0; i < obs.faces.size(); ++i)
  {
    if (std::find(exterior.begin(), exterior.end(), obs.faces[i]) == exterior.end())
      throw std::invalid_argument("high merge: selected face is not exterior");
    if (std::find(obs.faces.begin(), obs.faces.begin() + i, obs.faces[i])
        != obs.faces.begin() + i)
      throw std::invalid_argument("high merge: duplicate selected face");
  }
}

} // namespace high_conormal_detail

/*
  Refresh z after the ordinary update_merge_source(parent,A,B) has refreshed r.
  Child observations must already have their current source offsets.
*/
template<int D, class Real>
inline void update_high_conormal_merge_source(
  const Node<D,Real>& parent,
  const Node<D,Real>& A,
  const HighConormalObservation<D,Real>& obsA,
  const Node<D,Real>& B,
  const HighConormalObservation<D,Real>& obsB,
  HighConormalObservation<D,Real>& out)
{
  if (!parent.is_merge || parent.kf != A.kf || parent.kf != B.kf ||
      out.kH != obsA.kH || out.kH != obsB.kH ||
      obsA.H.size() != (std::size_t)obsA.nrows() * A.nb() ||
      obsB.H.size() != (std::size_t)obsB.nrows() * B.nb() ||
      obsA.z.size() != (std::size_t)obsA.nrows() ||
      obsB.z.size() != (std::size_t)obsB.nrows() ||
      out.H.size() != (std::size_t)out.nrows() * parent.nb() ||
      parent.merge.r.size() != (std::size_t)parent.merge.nI)
    throw std::invalid_argument("high merge source: incompatible dimensions");
  const auto& md = parent.merge;
  const int nh = out.nrows();
  out.z.assign((std::size_t)nh, Real(0));

  for (int f = 0; f < static_cast<int>(out.faces.size()); ++f)
  {
    const int fa = high_conormal_detail::face_slot<D,Real>(obsA, out.faces[(std::size_t)f]);
    const int fb = high_conormal_detail::face_slot<D,Real>(obsB, out.faces[(std::size_t)f]);
    if ((fa >= 0) == (fb >= 0))
      throw std::invalid_argument("high merge source: missing/ambiguous face");
    const bool fromA = fa >= 0;
    const auto& childObs = fromA ? obsA : obsB;
    const auto& I_idx = fromA ? md.A_I_idx : md.B_I_idx;
    const int row0 = (fromA ? fa : fb) * out.kH;
    for (int k = 0; k < out.kH; ++k)
    {
      Real value = childObs.z[(std::size_t)(row0 + k)];
      for (int i = 0; i < md.nI; ++i)
        value += childObs.H[(std::size_t)(row0 + k)
          + (std::size_t)childObs.nrows() * I_idx[(std::size_t)i]]
          * md.r[(std::size_t)i];
      out.z[(std::size_t)(f * out.kH + k)] = value;
    }
  }
}

/*
  Compose the high-flux observation through the *existing* HPS merge.
  All output face rows follow parent.boundary_faces order; inactive children
  contribute zero rows. No high interface unknowns are introduced.
*/
template<int D, class Real>
inline HighConormalObservation<D,Real> merge_high_conormal_observations(
  const Node<D,Real>& parent,
  const Node<D,Real>& A,
  const HighConormalObservation<D,Real>& obsA,
  const Node<D,Real>& B,
  const HighConormalObservation<D,Real>& obsB)
{
  if (!parent.is_merge || parent.kf != A.kf || parent.kf != B.kf ||
      obsA.kH != obsB.kH)
    throw std::invalid_argument("high merge: incompatible nodes/kH");
  const auto& md = parent.merge;
  if (md.nAE + md.nBE != parent.nb() || md.nI <= 0)
    throw std::invalid_argument("high merge: invalid interface partition");
  high_conormal_detail::check_child<D,Real>(A, obsA, md.EA_faces);
  high_conormal_detail::check_child<D,Real>(B, obsB, md.EB_faces);

  HighConormalObservation<D,Real> out;
  out.kH = obsA.kH;
  for (const auto& face : parent.boundary_faces)
  {
    if (high_conormal_detail::face_slot<D,Real>(obsA, face) >= 0 ||
        high_conormal_detail::face_slot<D,Real>(obsB, face) >= 0)
      out.faces.push_back(face);
  }

  const int nh = out.nrows();
  out.H.assign((std::size_t)nh * parent.nb(), Real(0));

  for (int f = 0; f < static_cast<int>(out.faces.size()); ++f)
  {
    const int fa = high_conormal_detail::face_slot<D,Real>(obsA, out.faces[(std::size_t)f]);
    const int fb = high_conormal_detail::face_slot<D,Real>(obsB, out.faces[(std::size_t)f]);
    if ((fa >= 0) == (fb >= 0))
      throw std::runtime_error("high merge: ambiguous observation ownership");

    const bool fromA = fa >= 0;
    const auto& childObs = fromA ? obsA : obsB;
    const auto& E_idx = fromA ? md.A_E_idx : md.B_E_idx;
    const auto& I_idx = fromA ? md.A_I_idx : md.B_I_idx;
    const int nE = fromA ? md.nAE : md.nBE;
    const int parent_col0 = fromA ? 0 : md.nAE;
    const int child_row0 = (fromA ? fa : fb) * out.kH;

    for (int k = 0; k < out.kH; ++k)
    {
      const int child_row = child_row0 + k;
      const int parent_row = f * out.kH + k;

      // Direct dependence on this child's surviving exterior trace.
      for (int j = 0; j < nE; ++j)
        out.H[(std::size_t)parent_row + (std::size_t)nh * (parent_col0 + j)] +=
          childObs.H[(std::size_t)child_row
            + (std::size_t)childObs.nrows() * E_idx[(std::size_t)j]];

      // Dependence of eliminated interface trace on *both* exterior sides.
      for (int i = 0; i < md.nI; ++i)
      {
        const Real hij = childObs.H[(std::size_t)child_row
          + (std::size_t)childObs.nrows() * I_idx[(std::size_t)i]];
        for (int j = 0; j < md.nAE; ++j)
          out.H[(std::size_t)parent_row + (std::size_t)nh * j] +=
            hij * md.R_A[(std::size_t)i + (std::size_t)md.nI * j];
        for (int j = 0; j < md.nBE; ++j)
          out.H[(std::size_t)parent_row + (std::size_t)nh * (md.nAE + j)] +=
            hij * md.R_B[(std::size_t)i + (std::size_t)md.nI * j];
      }
    }
  }

  update_high_conormal_merge_source(parent, A, obsA, B, obsB, out);
  return out;
}

template<int D, class Real>
inline void merge_compute_gamma(
  const Node<D,Real>& parent,
  const Real* lambda_AE,
  const Real* lambda_BE,
  Real* gamma_out
)
{
  if (!lambda_AE || !lambda_BE || !gamma_out)
  {
    throw std::invalid_argument("merge_compute_gamma: null input/output");
  }
  if (!parent.is_merge)
  {
    throw std::invalid_argument("merge_compute_gamma: parent is not a merge node");
  }

  const auto& md = parent.merge;
  const int nI = md.nI;
  const int nAE = md.nAE;
  const int nBE = md.nBE;

  for (int i = 0; i < nI; ++i)
  {
    gamma_out[i] = md.r[(std::size_t)i];
  }

  if (nAE > 0)
  {
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      nI,
      1,
      nAE,
      Real(1),
      md.R_A.data(),
      nI,
      lambda_AE,
      nAE,
      Real(1),
      gamma_out,
      nI);
  }

  if (nBE > 0)
  {
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      nI,
      1,
      nBE,
      Real(1),
      md.R_B.data(),
      nI,
      lambda_BE,
      nBE,
      Real(1),
      gamma_out,
      nI);
  }
}

/*
  Split lambda_parent according to the parent ordering [EA, EB], compute gamma,
  and scatter full child traces back into each child's native boundary ordering.
*/
template<int D, class Real>
inline void merge_reconstruct_child_traces(
  const Node<D,Real>& parent,
  const Node<D,Real>& A,
  const Node<D,Real>& B,
  const Real* lambda_parent,
  std::vector<Real>& lambda_A,
  std::vector<Real>& lambda_B,
  std::vector<Real>* gamma_out = nullptr
)
{
  if (!lambda_parent)
  {
    throw std::invalid_argument("merge_reconstruct_child_traces: null parent trace");
  }
  if (!parent.is_merge)
  {
    throw std::invalid_argument("merge_reconstruct_child_traces: parent is not a merge node");
  }

  const auto& md = parent.merge;
  const int nAE = md.nAE;
  const int nBE = md.nBE;
  const int nI = md.nI;

  const Real* lambda_AE = lambda_parent;
  const Real* lambda_BE = lambda_parent + nAE;

  std::vector<Real> gamma((std::size_t)nI, Real(0));
  merge_compute_gamma<D,Real>(parent, lambda_AE, lambda_BE, gamma.data());

  lambda_A.assign((std::size_t)A.nb(), Real(0));
  lambda_B.assign((std::size_t)B.nb(), Real(0));

  for (int i = 0; i < nAE; ++i)
  {
    lambda_A[(std::size_t)md.A_E_idx[(std::size_t)i]] = lambda_AE[i];
  }
  for (int i = 0; i < nBE; ++i)
  {
    lambda_B[(std::size_t)md.B_E_idx[(std::size_t)i]] = lambda_BE[i];
  }
  for (int i = 0; i < nI; ++i)
  {
    lambda_A[(std::size_t)md.A_I_idx[(std::size_t)i]] = gamma[(std::size_t)i];
    lambda_B[(std::size_t)md.B_I_idx[(std::size_t)i]] = gamma[(std::size_t)i];
  }

  if (gamma_out)
  {
    *gamma_out = gamma;
  }
}

} // namespace jsimplex

#endif // JMERGE_HH
