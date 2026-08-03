#ifndef JMULT_H
#define JMULT_H

#include <vector>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include <jbasis.hh>
#include <jdetail.hh>
#include <jquad_tprod.hh>

namespace jsimplex
{

template<class Real>
struct CSCView
{
  int nrow = 0;
  int ncol = 0;
  const int* colptr = nullptr;   // ncol+1
  const int* rowind = nullptr;   // nnz
  const Real* x = nullptr;       // nnz

  inline int nnz() const { return colptr ? colptr[ncol] : 0; }
};

template<class Real>
struct CSR
{
  int n = 0;
  std::vector<int> indptr;   // n+1
  std::vector<int> indices;  // nnz
  std::vector<Real> data;    // nnz
};

template<class Real>
static inline void csc_matvec(const CSCView<Real>& A, const Real* x, Real* y)
{
  std::memset(y, 0, (std::size_t)A.nrow * sizeof(Real));
  for (int j = 0; j < A.ncol; ++j)
  {
    const Real xj = x[j];
    const int p0 = A.colptr[j];
    const int p1 = A.colptr[j + 1];
    for (int p = p0; p < p1; ++p)
      y[A.rowind[p]] += A.x[p] * xj;
  }
}

static inline long long comb_int(long long n, long long k)
{
  if (k < 0 || k > n) return 0;
  k = std::min(k, n - k);
  long long num = 1, den = 1;
  for (long long i = 1; i <= k; ++i)
  {
    num *= (n - k + i);
    den *= i;
  }
  return num / den;
}

static inline int m_hom(int D, int j)
{
  return (int)comb_int((long long)j + D - 1, (long long)D - 1);
}

/* Build degree offsets from alpha_table (same logic as python)
   alpha_table: M x D, row-major, ints
   off length p+2
*/
static inline void degree_offsets_from_alpha(const int* alpha_table, int M, int D, int p,
                                             std::vector<int>& off_out)
{
  off_out.assign(p + 2, 0);
  // compute degrees
  std::vector<int> deg((std::size_t)M);
  for (int k = 0; k < M; ++k)
  {
    int s = 0;
    for (int i = 0; i < D; ++i) s += alpha_table[k * D + i];
    deg[k] = s;
  }
  int k = 0;
  for (int j = 0; j <= p; ++j)
  {
    off_out[j] = k;
    while (k < M && deg[k] == j) ++k;
  }
  off_out[p + 1] = k;
  // you can assert off_out[p+1] == M if alpha_table truncated to p
}

/* Canonical pivot rows (line-by-line port of pivot_rows_D_canonical) */
static inline void pivot_rows_D_canonical(const int* alpha_table_p, const int* off_p,
                                         int j, int D,
                                         std::vector<int>& piv_out)
{
  const int s0 = off_p[j];
  const int s1 = off_p[j + 1];
  const int mj = s1 - s0;

  piv_out.clear();
  piv_out.reserve((std::size_t)m_hom(D, j + 1));

  // piv.extend(range(0,mj))
  for (int r = 0; r < mj; ++r) piv_out.push_back(r);

  // for i=1..D-1: prefix==0 selection
  for (int i = 1; i < D; ++i)
  {
    for (int r = 0; r < mj; ++r)
    {
      bool all0 = true;
      for (int k = 0; k < i; ++k)
      {
        if (alpha_table_p[(s0 + r) * D + k] != 0) { all0 = false; break; }
      }
      if (all0) piv_out.push_back(i * mj + r);
    }
  }

  const int mj1 = m_hom(D, j + 1);
  assert((int)piv_out.size() == mj1);
}

/* Solve upper-tri CSR many RHS: Bt * X = RHS
   Bt: n x n upper triangular in CSR
   RHS: n x nrhs row-major in (n, nrhs)
   X overwritten in-place in X (same as python logic)
*/
template<class Real>
static inline void solve_upper_tri_sparse_many_rhs(const CSR<Real>& Bt,
                                                   Real* X, int nrhs,
                                                   bool check_diag=true)
{
  const int n = Bt.n;
  for (int i = n - 1; i >= 0; --i)
  {
    const int rs = Bt.indptr[i];
    const int re = Bt.indptr[i + 1];

    Real diag = (Real)0;
    bool have_diag = false;

    Real* Xi = X + (std::size_t)i * nrhs;

    for (int k = rs; k < re; ++k)
    {
      const int j = Bt.indices[k];
      const Real a = Bt.data[k];
      if (j == i)
      {
        diag = a;
        have_diag = true;
      }
      else
      {
        const Real* Xj = X + (std::size_t)j * nrhs;
        // Xi -= a * Xj
        for (int t = 0; t < nrhs; ++t) Xi[t] -= a * Xj[t];
      }
    }

    if (check_diag) assert(have_diag);
    assert(diag != (Real)0);

    const Real invd = (Real)1 / diag;
    for (int t = 0; t < nrhs; ++t) Xi[t] *= invd;
  }
}

/* Dense small matrix stored row-major */
template<class Real>
struct DenseMat
{
  int nrow = 0, ncol = 0;
  std::vector<Real> a; // size nrow*ncol row-major

  DenseMat() = default;
  DenseMat(int r, int c) : nrow(r), ncol(c), a((std::size_t)r * c, (Real)0) {}

  inline Real& operator()(int r, int c) { return a[(std::size_t)r * ncol + c]; }
  inline Real  operator()(int r, int c) const { return a[(std::size_t)r * ncol + c]; }
};

/* Build dense block Ji_p[row0:row1, col0:col1] from CSC, row-major output */
template<class Real>
static inline void csc_extract_dense_block(const CSCView<Real>& A,
                                           int row0, int row1,
                                           int col0, int col1,
                                           DenseMat<Real>& B)
{
  const int nr = row1 - row0;
  const int nc = col1 - col0;
  B = DenseMat<Real>(nr, nc);

  for (int cj = col0; cj < col1; ++cj)
  {
    const int p0 = A.colptr[cj];
    const int p1 = A.colptr[cj + 1];
    for (int p = p0; p < p1; ++p)
    {
      const int ri = A.rowind[p];
      if (ri < row0 || ri >= row1) continue;
      B(ri - row0, cj - col0) = A.x[p];
    }
  }
}

/* Convert dense upper-tri matrix U (n x n) to CSR with all entries (including zeros filtered by eps) */
template<class Real>
static inline void dense_upper_to_csr(const DenseMat<Real>& U, CSR<Real>& out, Real eps=(Real)0)
{
  assert(U.nrow == U.ncol);
  const int n = U.nrow;
  out.n = n;
  out.indptr.assign((std::size_t)n + 1, 0);
  out.indices.clear();
  out.data.clear();

  for (int i = 0; i < n; ++i)
  {
    out.indptr[i] = (int)out.indices.size();
    for (int j = i; j < n; ++j)
    {
      const Real v = U(i, j);
      if (eps == (Real)0 || std::abs(v) > eps)
      {
        out.indices.push_back(j);
        out.data.push_back(v);
      }
    }
  }
  out.indptr[n] = (int)out.indices.size();
}

/* --------------------------
   MultByQClenshaw workspace
   -------------------------- */

template<int D, class Real>
struct MultByQClenshawWorkspace
{
  int p = -1;
  int MK = 0;
  std::size_t max_block_size = 0;

  // Recurrence blocks v_j, j=1,...,p.  The two extra slots preserve the
  // indexing used by the Clenshaw back substitution.
  std::vector<std::vector<Real>> v;

  // Reused level-j scratch.  Each vector is allocated to the largest
  // homogeneous block m_p*MK, and apply() uses only the active prefix.
  std::vector<Real> rj;
  std::vector<Real> termA;
  std::vector<Real> termJ;
  std::vector<Real> s;
  std::vector<Real> termC;

  // Reused degree-zero scratch.
  std::vector<Real> r0;
  std::vector<Real> termA0;
  std::vector<Real> termJ0;
  std::vector<Real> v0;
  std::vector<Real> termC0;

  // Scratch for one J_fun coordinate matvec.
  std::vector<Real> tmpJ;

  void init(int p_in, int MK_in)
  {
    assert(p_in >= 0);
    assert(MK_in >= 0);

    p = p_in;
    MK = MK_in;

    v.clear();
    v.resize((std::size_t)p + 3);
    for (int j = 1; j <= p; ++j)
    {
      const std::size_t nj =
        (std::size_t)m_hom(D, j) * (std::size_t)MK;
      v[j].resize(nj);
    }

    max_block_size =
      (std::size_t)m_hom(D, p) * (std::size_t)MK;

    rj.resize(max_block_size);
    termA.resize(max_block_size);
    termJ.resize(max_block_size);
    s.resize(max_block_size);
    termC.resize(max_block_size);

    const std::size_t n0 = (std::size_t)MK;
    r0.resize(n0);
    termA0.resize(n0);
    termJ0.resize(n0);
    v0.resize(n0);
    termC0.resize(n0);
    tmpJ.resize(n0);
  }

  bool compatible(int p_in, int MK_in) const
  {
    return p == p_in && MK == MK_in;
  }
};

/* --------------------------
   MultByQClenshaw (ported)
   -------------------------- */

template<int D, class Real>
class MultByQClenshaw
{
public:
  // Inputs like python: D, p, K, kappa. assume_symmetric controls C-block extraction.
  int p = 0;
  int K = 0;
  bool assume_symmetric = true;

  // sizes
  int Mp = 0;
  int MK = 0;

  // alpha_table_p (Mp x D) row-major
  const int* alpha_p = nullptr;
  int alpha_p_M = 0;

  // degree offsets off_p (p+2)
  std::vector<int> off_p;

  // Jacobi matrices (CSC) for poly and fun spaces
  CSCView<Real> J_poly[D];
  CSCView<Real> J_fun[D];

  // per level (j=0..p-1)
  std::vector<DenseMat<Real>> Aeff;  // (m_{j+1} x m_j)
  std::vector<DenseMat<Real>> Beff;  // (m_{j+1} x m_{j+1})
  std::vector<DenseMat<Real>> Ceff;  // (m_{j+1} x m_{j-1}) stored at j, empty if j==0
  std::vector<std::vector<int>> sigma_i;
  std::vector<std::vector<int>> sigma_b;
  std::vector<CSR<Real>> BT_csr;     // Bt = Beff[j]^T in CSR (upper-tri)

  // Compatibility workspace used by the existing three-argument apply().
  // External C++ callers may instead supply their own workspace to the
  // four-argument overload, allowing one immutable plan per thread-shared
  // coefficient structure and one mutable workspace per concurrent apply.
  mutable MultByQClenshawWorkspace<D, Real> default_workspace;

  void init(int p_, int K_,
            const int* alpha_p_, int alpha_p_rows,
            const CSCView<Real> J_poly_in[D],
            const CSCView<Real> J_fun_in[D],
            bool assume_symmetric_=true)
  {
    p = p_;
    K = K_;
    assume_symmetric = assume_symmetric_;

    alpha_p = alpha_p_;
    alpha_p_M = alpha_p_rows;

    // Mp is alpha_p_rows; MK is J_fun dimension
    Mp = alpha_p_rows;
    MK = J_fun_in[0].nrow;

    for (int i = 0; i < D; ++i)
    {
      J_poly[i] = J_poly_in[i];
      J_fun[i]  = J_fun_in[i];
    }

    // off_p from alpha_p
    degree_offsets_from_alpha(alpha_p, Mp, D, p, off_p);

    Aeff.assign((std::size_t)p, DenseMat<Real>());
    Beff.assign((std::size_t)p, DenseMat<Real>());
    Ceff.assign((std::size_t)p, DenseMat<Real>());
    sigma_i.assign((std::size_t)p, std::vector<int>());
    sigma_b.assign((std::size_t)p, std::vector<int>());
    BT_csr.assign((std::size_t)p, CSR<Real>());

    default_workspace.init(p, MK);

    for (int j = 0; j < p; ++j)
      precompute_level(j);
  }

  /* apply(q_coeffs_p, c_coeffs_K) -> y_K (length MK)

     This overload preserves the existing API and reuses the workspace owned
     by the plan.  As before, simultaneous calls on the same plan are not
     thread-safe.  Use the four-argument overload with one workspace per
     concurrent caller when thread-safe shared-plan use is needed.
  */
  void apply(const Real* q_coeffs_p,
             const Real* c_coeffs_K,
             Real* y_out) const
  {
    apply(q_coeffs_p, c_coeffs_K, y_out, default_workspace);
  }

  /* Workspace-explicit apply.  No dynamic allocation occurs after the
     workspace has been initialized for this plan.
  */
  void apply(const Real* q_coeffs_p,
             const Real* c_coeffs_K,
             Real* y_out,
             MultByQClenshawWorkspace<D, Real>& work) const
  {
    assert(q_coeffs_p);
    assert(c_coeffs_K);
    assert(y_out);
    assert(work.compatible(p, MK));

    // p==0 returns r0 = q0 tensor cK, i.e. scalar multiplication.
    if (p == 0)
    {
      const Real q0 = q_coeffs_p[0];
      for (int k = 0; k < MK; ++k) y_out[k] = q0 * c_coeffs_K[k];
      return;
    }

    // Highest-degree initialization: build r_p directly in v_p, then solve.
    {
      Real* vp = work.v[p].data();
      build_r_block(p, q_coeffs_p, c_coeffs_K, vp);
      solve_BT_kronI(p - 1, vp);
    }

    // Descending back substitution for j=p-1,...,1.
    for (int j = p - 1; j >= 1; --j)
    {
      const std::size_t nblock =
        (std::size_t)m_hom(D, j) * (std::size_t)MK;

      Real* rj = work.rj.data();
      Real* termA = work.termA.data();
      Real* termJ = work.termJ.data();
      Real* s = work.s.data();
      Real* vj = work.v[j].data();

      build_r_block(j, q_coeffs_p, c_coeffs_K, rj);
      apply_AkronI_T(j, work.v[j + 1].data(), termA);
      apply_JT(j, work.v[j + 1].data(), termJ, work.tmpJ.data());

      // s = r_j - (termA - termJ)
      std::memcpy(s, rj, nblock * sizeof(Real));
      for (std::size_t t = 0; t < nblock; ++t)
        s[t] -= (termA[t] - termJ[t]);

      // Subtract the C contribution when v_{j+2} is present.
      if (j + 1 <= p - 1)
      {
        Real* termC = work.termC.data();
        apply_CkronI_T(j + 1, work.v[j + 2].data(), termC);
        for (std::size_t t = 0; t < nblock; ++t) s[t] -= termC[t];
      }

      std::memcpy(vj, s, nblock * sizeof(Real));
      solve_BT_kronI(j - 1, vj);
    }

    // Final degree-zero block.
    {
      Real* r0 = work.r0.data();
      Real* termA0 = work.termA0.data();
      Real* termJ0 = work.termJ0.data();
      Real* v0 = work.v0.data();

      build_r0(q_coeffs_p, c_coeffs_K, r0);
      apply_AkronI_T(0, work.v[1].data(), termA0);
      apply_JT(0, work.v[1].data(), termJ0, work.tmpJ.data());

      std::memcpy(v0, r0, (std::size_t)MK * sizeof(Real));
      for (int k = 0; k < MK; ++k)
        v0[k] -= (termA0[k] - termJ0[k]);

      if (p >= 2)
      {
        Real* termC0 = work.termC0.data();
        apply_CkronI_T(1, work.v[2].data(), termC0);
        for (int k = 0; k < MK; ++k) v0[k] -= termC0[k];
      }

      std::memcpy(y_out, v0, (std::size_t)MK * sizeof(Real));
    }
  }

private:
  void precompute_level(int j)
  {
    const int mj = m_hom(D, j);
    const int mj1 = m_hom(D, j + 1);

    const int sj0 = off_p[j];
    const int sj1 = off_p[j + 1];
    const int sjp10 = off_p[j + 1];
    const int sjp11 = off_p[j + 2];

    // pivot rows (length mj1)
    std::vector<int> piv;
    pivot_rows_D_canonical(alpha_p, off_p.data(), j, D, piv);

    // Build Aeff/Beff/Ceff by extracting blocks from each Ji_p and then picking rows.
    DenseMat<Real> Aeff_j(mj1, mj);
    DenseMat<Real> Beff_j(mj1, mj1);
    DenseMat<Real> Ceff_j;
    if (j >= 1) Ceff_j = DenseMat<Real>(mj1, m_hom(D, j - 1));

    // Pre-extract per-dimension blocks (dense)
    // A_block: (mj x mj) from Ji_p[sj0:sj1, sj0:sj1]
    // B_block: (mj x mj1) from Ji_p[sj0:sj1, sjp10:sjp11]
    // C_block: (mj x m_{j-1}) from Ji_p[sj0:sj1, sm10:sm11]  (or transpose from previous, if symmetric)
    const int sm10 = (j >= 1) ? off_p[j - 1] : 0;
    const int sm11 = (j >= 1) ? off_p[j] : 0;

    for (int dim = 0; dim < D; ++dim)
    {
      DenseMat<Real> Ablk, Bblk, Cblk;
      csc_extract_dense_block(J_poly[dim], sj0, sj1, sj0, sj1, Ablk);
      csc_extract_dense_block(J_poly[dim], sj0, sj1, sjp10, sjp11, Bblk);

      if (j >= 1)
      {
        if (assume_symmetric)
        {
          // Bj_m1 = Ji_p[sm10:sm11, sj0:sj1], then C = Bj_m1^T
          DenseMat<Real> Bj_m1;
          csc_extract_dense_block(J_poly[dim], sm10, sm11, sj0, sj1, Bj_m1);
          // transpose into Cblk: shape (mj x m_{j-1})
          const int mr = sj1 - sj0;       // mj
          const int mc = sm11 - sm10;     // m_{j-1}
          Cblk = DenseMat<Real>(mr, mc);
          for (int r = 0; r < Bj_m1.nrow; ++r)
            for (int c = 0; c < Bj_m1.ncol; ++c)
              Cblk(c, r) = Bj_m1(r, c);
        }
        else
        {
          csc_extract_dense_block(J_poly[dim], sj0, sj1, sm10, sm11, Cblk);
        }
      }

      // Fill selected rows into Aeff/Beff/Ceff according to piv mapping
      for (int r_sel = 0; r_sel < mj1; ++r_sel)
      {
        const int r_stack = piv[r_sel];
        const int i = r_stack / mj;
        const int b = r_stack - i * mj;
        if (i != dim) continue;

        // Aeff row r_sel = Ablk row b
        for (int c = 0; c < mj; ++c) Aeff_j(r_sel, c) = Ablk(b, c);

        // Beff row r_sel = Bblk row b
        for (int c = 0; c < mj1; ++c) Beff_j(r_sel, c) = Bblk(b, c);

        if (j >= 1)
        {
          const int mjm1 = m_hom(D, j - 1);
          for (int c = 0; c < mjm1; ++c) Ceff_j(r_sel, c) = Cblk(b, c);
        }
      }
    }

    Aeff[j] = std::move(Aeff_j);
    Beff[j] = std::move(Beff_j);
    if (j >= 1) Ceff[j] = std::move(Ceff_j);

    // sigma maps (same as python)
    sigma_i[j].assign((std::size_t)mj1, 0);
    sigma_b[j].assign((std::size_t)mj1, 0);
    for (int r_sel = 0; r_sel < mj1; ++r_sel)
    {
      const int r_stack = piv[r_sel];
      const int i = r_stack / mj;
      const int b = r_stack - i * mj;
      sigma_i[j][r_sel] = i;
      sigma_b[j][r_sel] = b;
    }

    // Bt = Beff^T in CSR (upper-tri)
    // Since Beff is (mj1 x mj1), Bt is also (mj1 x mj1).
    DenseMat<Real> Bt_dense(mj1, mj1);
    for (int r = 0; r < mj1; ++r)
      for (int c = 0; c < mj1; ++c)
        Bt_dense(r, c) = Beff[j](c, r);

    dense_upper_to_csr(Bt_dense, BT_csr[j], (Real)0);
  }

  // r_j = kron(qj, cK) stored as (m_j, MK) row-major
  void build_r_block(int j, const Real* q_coeffs_p, const Real* cK, Real* out) const
  {
    const int s0 = off_p[j];
    const int s1 = off_p[j + 1];
    const int mj = s1 - s0;
    for (int a = 0; a < mj; ++a)
    {
      const Real qa = q_coeffs_p[s0 + a];
      Real* row = out + (std::size_t)a * MK;
      for (int k = 0; k < MK; ++k) row[k] = qa * cK[k];
    }
  }

  void build_r0(const Real* q_coeffs_p, const Real* cK, Real* out) const
  {
    // j=0 => m_hom(D,0)=1, q segment length 1 at off_p[0]:off_p[1]
    const Real q0 = q_coeffs_p[off_p[0]];
    for (int k = 0; k < MK; ++k) out[k] = q0 * cK[k];
  }

  // (Aeff[j] ⊗ I)^T w, w in (m_{j+1}*MK), out in (m_j*MK)
  void apply_AkronI_T(int j, const Real* w, Real* out) const
  {
    const DenseMat<Real>& A = Aeff[j];
    const int mj = A.ncol;
    const int mj1 = A.nrow;

    // W is (mj1, MK) row-major; out is (mj, MK) row-major
    std::memset(out, 0, (std::size_t)mj * MK * sizeof(Real));

    for (int r = 0; r < mj1; ++r)
    {
      const Real* Wr = w + (std::size_t)r * MK;
      for (int c = 0; c < mj; ++c)
      {
        const Real a = A(r, c); // A(r,c)
        if (a == (Real)0) continue;
        Real* Yc = out + (std::size_t)c * MK;
        for (int k = 0; k < MK; ++k) Yc[k] += a * Wr[k];
      }
    }
  }

  // (Ceff[j_plus_1] ⊗ I)^T w, where w=v_{j+2} and output is in m_j*MK
  void apply_CkronI_T(int j_plus_1, const Real* w, Real* out) const
  {
    // python: _apply_CkronI_T(j_plus_1, w) uses Ceff[j_plus_1]
    const DenseMat<Real>& C = Ceff[j_plus_1];
    const int mj = C.ncol;
    const int mj2 = C.nrow;

    std::memset(out, 0, (std::size_t)mj * MK * sizeof(Real));

    for (int r = 0; r < mj2; ++r)
    {
      const Real* Wr = w + (std::size_t)r * MK;
      for (int c = 0; c < mj; ++c)
      {
        const Real a = C(r, c);
        if (a == (Real)0) continue;
        Real* Yc = out + (std::size_t)c * MK;
        for (int k = 0; k < MK; ++k) Yc[k] += a * Wr[k];
      }
    }
  }

  // J_j^T w using sigma maps:
  // out[b,:] += J_fun[i] * W[ell,:]
  void apply_JT(int j, const Real* w, Real* out, Real* tmpJ) const
  {
    const int mj = m_hom(D, j);
    const int mj1 = (int)sigma_i[j].size();
    std::memset(out, 0, (std::size_t)mj * MK * sizeof(Real));

    for (int ell = 0; ell < mj1; ++ell)
    {
      const int i = sigma_i[j][ell];
      const int b = sigma_b[j][ell];
      const Real* Well = w + (std::size_t)ell * MK;

      // tmpJ = J_fun[i] @ Well
      csc_matvec(J_fun[i], Well, tmpJ);

      Real* outb = out + (std::size_t)b * MK;
      for (int k = 0; k < MK; ++k) outb[k] += tmpJ[k];
    }
  }

  // Solve (Beff[j_minus_1]^T ⊗ I) v = s, in-place on s
  void solve_BT_kronI(int j_minus_1, Real* s_inout) const
  {
    const CSR<Real>& Bt = BT_csr[j_minus_1];
    const int mj = Bt.n;
    // s_inout is (mj, MK) row-major
    solve_upper_tri_sparse_many_rhs(Bt, s_inout, MK, true);
  }
};


/*
  Direct Galerkin materialization of a restricted multiplication operator.

    M_q^{R<-N} = V_R^T diag(W .* q(X)) V_N,

  where q is represented in the same orthonormal Jacobi family through degree
  p. The quadrature rule and the full basis table are immutable and shared;
  the caller owns one workspace per concurrent assembly.
*/
template<int D, class Real>
class MultByQQuadraturePlan
{
public:
  int max_degree = 0;
  int q_order = 0;
  int nq = 0;
  int Mmax = 0;
  std::array<Real, D + 1> kappa{};
  std::vector<Real> X; // row-major nq x D
  std::vector<Real> W; // nq
  std::vector<Real> V; // column-major nq x dim(Pi_max_degree)

  MultByQQuadraturePlan() = default;

  MultByQQuadraturePlan(
    const Real* kappa_in,
    int max_degree_in,
    int q_order_in)
    : max_degree(max_degree_in), q_order(q_order_in)
  {
    if (!kappa_in)
      throw std::invalid_argument(
        "MultByQQuadraturePlan: null kappa");
    if (max_degree < 0 || q_order <= 0)
      throw std::invalid_argument(
        "MultByQQuadraturePlan: invalid degree/order");

    for (int i = 0; i <= D; ++i)
      kappa[(std::size_t)i] = kappa_in[i];

    const unsigned int nq_unsigned =
      QuadMapped<D,Real>::npoints((unsigned int)q_order);
    if (nq_unsigned > (unsigned int)std::numeric_limits<int>::max())
      throw std::overflow_error(
        "MultByQQuadraturePlan: quadrature size overflow");
    nq = (int)nq_unsigned;
    Mmax = Basis<D,Real>::dim_Pi(max_degree);

    X.assign((std::size_t)nq * D, Real(0));
    W.assign((std::size_t)nq, Real(0));
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_order,
      kappa.data(),
      X.data(),
      W.data());
    if (built != nq)
      throw std::runtime_error(
        "MultByQQuadraturePlan: quadrature build failed");

    std::vector<int> alpha;
    std::vector<int> tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(
      max_degree,
      kappa.data(),
      alpha,
      tail,
      invh);

    V.assign((std::size_t)nq * Mmax, Real(0));
    Basis<D,Real>::eval_all(
      X.data(), D, 1, nq,
      kappa.data(), max_degree,
      alpha.data(), tail.data(), invh.data(),
      V.data(), nq, nullptr);
  }

  int size_for_degree(int degree) const
  {
    if (degree < 0 || degree > max_degree)
      throw std::out_of_range(
        "MultByQQuadraturePlan: requested degree is unavailable");
    return Basis<D,Real>::dim_Pi(degree);
  }
};

template<class Real>
struct MultByQQuadratureWorkspace
{
  std::vector<Real> q_values;
  std::vector<Real> weighted_input_panel;

  void resize(int nq, int panel_width)
  {
    if (nq < 0 || panel_width < 0)
      throw std::invalid_argument(
        "MultByQQuadratureWorkspace: invalid dimensions");
    q_values.resize((std::size_t)nq);
    weighted_input_panel.resize(
      (std::size_t)nq * (std::size_t)panel_width);
  }
};

template<int D, class Real>
void assemble_restricted_mult_quadrature(
  const MultByQQuadraturePlan<D,Real>& plan,
  const Real* q_coefficients,
  int coefficient_degree,
  int input_degree,
  int output_degree,
  MultByQQuadratureWorkspace<Real>& workspace,
  Real* M_out)
{
  if (!q_coefficients || !M_out)
    throw std::invalid_argument(
      "assemble_restricted_mult_quadrature: null array");
  if (coefficient_degree < 0 || input_degree < 0 || output_degree < 0 ||
      coefficient_degree > plan.max_degree ||
      input_degree > plan.max_degree ||
      output_degree > plan.max_degree)
    throw std::invalid_argument(
      "assemble_restricted_mult_quadrature: degree outside plan");

  const int Mp = plan.size_for_degree(coefficient_degree);
  const int MN = plan.size_for_degree(input_degree);
  const int MR = plan.size_for_degree(output_degree);
  const int nq = plan.nq;

  // Keep the temporary panel near eight MiB for double precision while
  // retaining enough columns for efficient GEMM at moderate dimensions.
  const std::size_t target_entries = (std::size_t)1 << 20;
  const int panel_width = std::max(
    1,
    std::min(
      MN,
      std::min(
        64,
        (int)std::max<std::size_t>(
          1,
          target_entries / (std::size_t)std::max(nq, 1)))));
  workspace.resize(nq, panel_width);

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    nq,
    1,
    Mp,
    Real(1),
    plan.V.data(),
    nq,
    q_coefficients,
    Mp,
    Real(0),
    workspace.q_values.data(),
    nq);

  for (int first_column = 0;
       first_column < MN;
       first_column += panel_width)
  {
    const int width = std::min(
      panel_width,
      MN - first_column);

    for (int local_column = 0;
         local_column < width;
         ++local_column)
    {
      const Real* input_column =
        plan.V.data()
        + (std::size_t)nq
          * (std::size_t)(first_column + local_column);
      Real* weighted_column =
        workspace.weighted_input_panel.data()
        + (std::size_t)nq * (std::size_t)local_column;

      for (int point = 0; point < nq; ++point)
      {
        weighted_column[(std::size_t)point] =
          plan.W[(std::size_t)point]
          * workspace.q_values[(std::size_t)point]
          * input_column[(std::size_t)point];
      }
    }

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans,
      CblasNoTrans,
      MR,
      width,
      nq,
      Real(1),
      plan.V.data(),
      nq,
      workspace.weighted_input_panel.data(),
      nq,
      Real(0),
      M_out + (std::size_t)MR * (std::size_t)first_column,
      MR);
  }
}

} // namespace jsimplex

#endif // JMULT_H
