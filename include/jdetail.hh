#ifndef JDETAIL_H
#define JDETAIL_H

#include <type_traits>
#include <lapacke.h>
#include <cblas.h>
#include <iostream> 
namespace jsimplex
{

namespace detail
{


/* Classical 1D Jacobi basis P_n^{(a,b)} on [-1,1].

   This struct provides a stable 3-term recurrence to evaluate
   P_k^{(a,b)}(x) at a single point x.

   Base Cases:
     P_0^{(a,b)}(x) = 1
     P_1^{(a,b)}(x) = 0.5 * [ (a + b + 2) x + (a - b) ]

   Interface:
     - n degree
     - a,b > -1
     - x in [-1,1]
    Return:
     - P_k^{(a,b)}(x), 

   Notes:
     - This is the *classical* Jacobi orthogonalization (non-orthonormal).
     - Intended for internal use by simplex basis evaluators. */
template<class Real>
struct BasisClassic1D
{

  /* Single-value evaluator: orthogonal Jacobi P_n^{(a,b)}(x) */
  static inline Real eval_n(int n, Real a, Real b, Real x)
  {
    if (n == 0)
    {
      return Real(1.0);
    }
  
    Real apb = a + b;
    Real aa  = a * a;
    Real bb  = b * b;
  
    // P_0(x)
    Real Pkm1 = Real(1.0);
  
    // P_1(x)
    Real Pk = Real(0.5) *
              ( Real(2.0) * (a + Real(1.0)) +
                (apb + Real(2.0)) * (x - Real(1.0)) );
  
    if (n == 1)
    {
      return Pk;
    }
  
    // n >= 2: recurrence for k = 2..n
    for (int k = 2; k <= n; ++k)
    {
      Real kR    = static_cast<Real>(k);
      Real k2    = Real(2.0) * kR;
      Real k2apb = k2 + apb;
  
      Real q1 =  k2 * (kR + apb) * (k2apb - Real(2.0));
      Real q2 = (k2apb - Real(1.0)) * (aa - bb);
      Real q3 = (k2apb - Real(2.0)) *
                (k2apb - Real(1.0)) *
                 k2apb;
      Real q4 = Real(2.0) * (kR + a - Real(1.0)) *
                            (kR + b - Real(1.0)) *
                             k2apb;
  
      Real Pkp1 = ( (q2 + q3 * x) * Pk - q4 * Pkm1 ) / q1;
  
      Pkm1 = Pk;
      Pk   = Pkp1;
    }
  
    return Pk;
  }
}; // BasisClassic1D

template<class Real>
struct BlasGemm;

// float specialization
template<>
struct BlasGemm<float>
{
  static inline void run(CBLAS_ORDER order,
                         CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB,
                         int M, int N, int K,
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc)
  {
    cblas_sgemm(order, transA, transB, M, N, K,
                alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

// double specialization
template<>
struct BlasGemm<double>
{
  static inline void run(CBLAS_ORDER order,
                         CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB,
                         int M, int N, int K,
                         double alpha,
                         const double* A, int lda,
                         const double* B, int ldb,
                         double beta,
                         double* C, int ldc)
  {
    cblas_dgemm(order, transA, transB, M, N, K,
                alpha, A, lda, B, ldb, beta, C, ldc);
  }
};




/* Symmetric tridiagonal eigen-decomposition:
     - For Real = float  -> LAPACKE_sstevd
     - For Real = double -> LAPACKE_dstevd
   Used for Golub-Welsch in 1D Gauss-Jacobi. */
template<class Real>
struct LapackStevd;

template<>
struct LapackStevd<float> // float specialization
{
  static inline lapack_int run(lapack_int n, float* d, float* e,
                        float* Z, lapack_int ldz)
  {
    return LAPACKE_sstevd(LAPACK_COL_MAJOR, 'V', n, d, e, Z, ldz);
  }
};

template<>
struct LapackStevd<double> // double specialization
{
  static inline lapack_int run(lapack_int n, double* d, double* e,
                        double* Z, lapack_int ldz)
  {
    return LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', n, d, e, Z, ldz);
  }
};

template<class Real>
struct LapackGelsd;

template<>
struct LapackGelsd<float>
{
  static inline lapack_int run(lapack_int m, lapack_int n, lapack_int nrhs,
                               float* A, lapack_int lda,
                               float* B, lapack_int ldb,
                               float* S, float rcond, lapack_int* rank)
  {
    return LAPACKE_sgelsd(LAPACK_COL_MAJOR, m, n, nrhs,
                          A, lda,
                          B, ldb,
                          S, rcond, rank);
  }
};

template<>
struct LapackGelsd<double>
{
  static inline lapack_int run(lapack_int m, lapack_int n, lapack_int nrhs,
                               double* A, lapack_int lda,
                               double* B, lapack_int ldb,
                               double* S, double rcond, lapack_int* rank)
  {
    return LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, n, nrhs,
                          A, lda,
                          B, ldb,
                          S, rcond, rank);
  }
};

template<class Real>
struct LapackGesdd;

template<>
struct LapackGesdd<float>
{
  static inline lapack_int run(
      char jobz,
      lapack_int m, lapack_int n,
      float* A, lapack_int lda,
      float* S, float* U, lapack_int ldu,
      float* VT, lapack_int ldvt )
  {
    return LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz,
                          m, n, A, lda, S,
                          U, ldu,
                          VT, ldvt);
  }
};

template<>
struct LapackGesdd<double>
{
  static inline lapack_int run(
      char jobz,
      lapack_int m, lapack_int n,
      double* A, lapack_int lda,
      double* S, double* U, lapack_int ldu,
      double* VT, lapack_int ldvt )
  {
    return LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz,
                          m, n, A, lda, S,
                          U, ldu,
                          VT, ldvt);
  }
};

// ------------------------------------------------------------
// cond_2(V): 2-norm condition number via SVD.
// Makes an internal copy of A because gesdd overwrites input.
// ------------------------------------------------------------
template<class Real>
static Real cond_number(int m, int n, const Real* A)
{
  if (m <= 0 || n <= 0) return Real(0);

  // Copy matrix so we don't overwrite V_opt
  Real* Acopy = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(m*n), sizeof(Real)));
  if (!Acopy) return Real(0);

  for (int i = 0; i < m*n; ++i)
    Acopy[i] = A[i];

  lapack_int M = m;
  lapack_int N = n;
  lapack_int Sdim = (M < N ? M : N);

  Real* S = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(Sdim), sizeof(Real)));
  if (!S)
  {
    std::free(Acopy);
    return Real(0);
  }

  // Dummy U, VT because LAPACK still checks arguments,
  // even when jobz='N'.
  Real* U_dummy  = static_cast<Real*>(std::calloc(1, sizeof(Real)));
  Real* VT_dummy = static_cast<Real*>(std::calloc(1, sizeof(Real)));

  lapack_int ldu  = 1;
  lapack_int ldvt = 1;

  lapack_int info = LapackGesdd<Real>::run(
      'N',        // jobz: compute singular values only
      M, N,
      Acopy, M,   // A, lda
      S,
      U_dummy, ldu,
      VT_dummy, ldvt
  );

  Real cond = Real(0);

  if (info == 0)
  {
    Real smax = S[0];
    Real smin = S[0];
    for (int i = 1; i < Sdim; ++i)
    {
      if (S[i] > smax) smax = S[i];
      if (S[i] < smin) smin = S[i];
    }
    if (smin > Real(0))
      cond = smax / smin;
    else
      cond = Real(1e300);  // effectively infinite
  }
  else
  {
    cond = Real(0);
  }

  std::free(Acopy);
  std::free(S);
  std::free(U_dummy);
  std::free(VT_dummy);

  return cond;
}

/* Jacobi ON (orthonormal) tridiagonal for w(x)=(1-x)^a(1+x)^b on [-1,1].
   Fills:
     d[0..n-1]   = main diagonal
     e[0..n-2]   = off diagonal (super- and sub-diagonal, same values)
   Real can be float or double. */
template<class Real>
inline void jacobi_tridiag_ON(int n, Real a, Real b, Real* d, Real* e)
{
  if (n <= 0 || !d || (n>1 && !e)) return;

  // bvec: main diagonal entries
  // avecON: off-diagonal (positive) entries
  Real* bvec   = (Real*) std::calloc((size_t)n, sizeof(Real));
  Real* avecON = (Real*) std::calloc((size_t)n, sizeof(Real));
  if (!bvec || !avecON) { std::free(bvec); std::free(avecON); return; }

  const Real apb = a + b;
  const Real aa  = a * a;
  const Real bb  = b * b;

  bvec[0]   = -(Real(0.5) * (a - b)) / (Real(0.5) * (a + b) + Real(1));
  avecON[0] = (Real(2) / (a + b + Real(2))) *
              std::sqrt( (a + Real(1)) * (b + Real(1)) / (a + b + Real(3)) );

  for (int i = 1; i < n; ++i)
  {
    const Real ii   = Real(i);
    const Real k2   = Real(2) * ii;
    const Real k2apb= k2 + apb;

    const Real av =
      ( (k2apb + Real(1)) * (k2apb + Real(2)) ) /
      ( Real(2) * (ii + Real(1)) * (ii + apb + Real(1)) );

    const Real bv =
      ( (aa - bb) * (k2apb + Real(1)) ) /
      ( Real(2) * (ii + Real(1)) * (ii + apb + Real(1)) * (k2apb) );

    bvec[i] = -bv / av;

    avecON[i] =
      ( Real(2) / (a + b + Real(2) * ii + Real(2)) ) *
      std::sqrt( (a + ii + Real(1)) * (b + ii + Real(1)) *
                 (ii + Real(1)) * (a + b + ii + Real(1)) /
                 ( (a + b + Real(2) * ii + Real(1)) *
                   (a + b + Real(2) * ii + Real(3)) ) );
  }

  for (int i = 0; i < n; ++i) d[i] = bvec[i];
  for (int i = 0; i+1 < n; ++i) e[i] = avecON[i];

  std::free(bvec); std::free(avecON);
}


/* 1D Gauss-Jacobi on [0,1] for weight t^k_1 (1-t)^k_2,

   Inputs:
     n     : number of nodes
     kappa : k_1,k_2 > -1/2

   Outputs:
     t[0..n-1] : nodes in [0,1]
     w[0..n-1] : weights */
template<class Real>
inline void gauss_jacobi_unit(unsigned int n, const Real* kappa,
                              Real* t, Real* w)
{
  if (n == 0) return;

  // map κ -> (a,b) on [-1,1]: a = κ2-1/2, b = κ1-1/2 (then map to [0,1])
  const Real a = kappa[1] - Real(0.5);
  const Real b = kappa[0] - Real(0.5);

  Real* d = (Real*) std::malloc(n * sizeof(Real));
  Real* e = (Real*) std::malloc((n > 1 ? (n - 1) : 1) * sizeof(Real));
  Real* Z = (Real*) std::calloc((size_t)n * (size_t)n, sizeof(Real));
  if (!d || !e || !Z) { std::free(d); std::free(e); std::free(Z); return; }

  jacobi_tridiag_ON<Real>((int)n, a, b, d, e);

  // eigendecompose tridiagonal (d,e) with LAPACK
  const lapack_int N = (lapack_int)n;
  if (LapackStevd<Real>::run(N, d, e, Z, N) != 0)
  {
    for (unsigned int i=0;i<n;++i){ t[i]=Real(0); w[i]=Real(0); }
    std::free(d); std::free(e); std::free(Z); return;
  }

  // nodes (map to [0,1]) and weights from first component
  for (unsigned int i = 0; i < n; ++i)
  {
    const Real u  = d[i];
    const Real v0 = Z[0 + i * N];
    t[i] = (u + Real(1)) * Real(0.5);
    w[i] = v0 * v0;
  }

  std::free(d); std::free(e); std::free(Z);
}


/* Build n-point Gauss–Legendre quadrature on [0,1]:
   - Using the orthonormal Legendre Jacobi matrix (a = b = 0)
   - Nodes t[0..n-1] in [0,1]
   - Weights w[0..n-1], sum(w) ≈ 1 (normalized measure)

   This mirrors gauss_jacobi_unit, but with a=b=0 specialized,
   and works for Real = float or double via LapackStevd. */
template<class Real>
inline void legendre_unit(unsigned int n, Real* t, Real* w)
{
  static_assert(std::is_same<Real,float>::value  ||
                std::is_same<Real,double>::value,
                "legendre_unit requires Real=float or Real=double");

  if (n == 0 || !t || !w)
  {
    return;
  }

  // Diagonal d[0..n-1], off-diagonal e[0..n-2] for symmetric tridiagonal.
  Real* d = (Real*) std::malloc(n * sizeof(Real));
  if (!d)
  {
    return;
  }

  Real* e = (Real*) std::malloc((n > 1 ? (n - 1) : 1) * sizeof(Real));
  if (!e)
  {
    std::free(d);
    return;
  }

  // For Legendre (a = b = 0), the orthonormal recurrence gives:
  // diagonal entries: 0, off-diagonal: a_i = (i+1) / sqrt((2i+1)(2i+3))
  for (unsigned int i = 0; i < n; ++i)
  {
    d[i] = Real(0.0);

    if (i + 1 < n)
    {
      Real ip1      = Real(i) + Real(1.0);
      Real two_i_p1 = Real(2.0) * Real(i) + Real(1.0);
      Real two_i_p3 = Real(2.0) * Real(i) + Real(3.0);

      e[i] = ip1 / std::sqrt(two_i_p1 * two_i_p3);
    }
  }

  // Eigenvectors Z (n x n), column-major for LAPACK.
  Real* Z = (Real*) std::calloc(n * n, sizeof(Real));
  if (!Z)
  {
    std::free(d);
    std::free(e);
    return;
  }

  lapack_int N    = (lapack_int) n;
  lapack_int info = detail::LapackStevd<Real>::run(N, d, e, Z, N);

  std::free(e);

  if (info != 0)
  {
    // LAPACK failure: zero outputs and bail.
    for (unsigned int i = 0; i < n; ++i)
    {
      t[i] = Real(0.0);
      w[i] = Real(0.0);
    }
    std::free(d);
    std::free(Z);
    return;
  }

  // Nodes: u_i = d[i] in [-1,1].
  // Weights: w_i = (first component of eigenvector i)^2.
  for (unsigned int i = 0; i < n; ++i)
  {
    Real u_i = d[i];
    Real v0  = Z[0 + i * N];
    Real wi  = v0 * v0;

    // Map from [-1,1] to [0,1].
    t[i] = (u_i + Real(1.0)) * Real(0.5);
    w[i] = wi;
  }

  std::free(d);
  std::free(Z);
}

/* log_pochhammer(a, n) = log (a)_n = log Gamma(a+n) - log Gamma(a) */
template<class Real>
inline Real log_pochhammer(Real a, int n)
{
  if (n <= 0)
  {
    return Real(0);
  }

  double ad  = static_cast<double>(a);
  double val = std::lgamma(ad + static_cast<double>(n))
             - std::lgamma(ad);

  return static_cast<Real>(val);
}

/* convert dense to CSC */
template<class Real>
static inline std::size_t compress_dense_to_csc(int M,
                                                const Real* A,     // row-major MxM
                                                int** colptr_out,
                                                int** rowind_out,
                                                Real** x_out)
{
  int* colnnz = (int*) std::malloc((std::size_t)M * sizeof(int));
  if (!colnnz)
  {
    std::cerr << "compress_dense_to_csc: alloc colnnz failed\n";
    std::exit(1);
  }
  for (int j = 0; j < M; ++j) colnnz[j] = 0;

  // count nnz per column
  for (int j = 0; j < M; ++j)
  {
    int cnt = 0;
    for (int i = 0; i < M; ++i)
    {
      const Real v = A[(std::size_t)i * (std::size_t)M + (std::size_t)j];
      if (v != Real(0)) ++cnt;
    }
    colnnz[j] = cnt;
  }

  int* colptr = (int*) std::malloc((std::size_t)(M + 1) * sizeof(int));
  if (!colptr)
  {
    std::cerr << "compress_dense_to_csc: alloc colptr failed\n";
    std::exit(1);
  }

  colptr[0] = 0;
  for (int j = 0; j < M; ++j)
  {
    colptr[j + 1] = colptr[j] + colnnz[j];
  }

  const int nnz = colptr[M];

  int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
  Real* x     = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
  if ((!rowind && nnz) || (!x && nnz))
  {
    std::cerr << "compress_dense_to_csc: alloc nnz arrays failed\n";
    std::exit(1);
  }

  int* wpos = (int*) std::malloc((std::size_t)M * sizeof(int));
  if (!wpos)
  {
    std::cerr << "compress_dense_to_csc: alloc wpos failed\n";
    std::exit(1);
  }
  for (int j = 0; j < M; ++j) wpos[j] = colptr[j];

  // fill (unsorted rows per column, but deterministic)
  for (int j = 0; j < M; ++j)
  {
    for (int i = 0; i < M; ++i)
    {
      const Real v = A[(std::size_t)i * (std::size_t)M + (std::size_t)j];
      if (v == Real(0)) continue;

      const int pos = wpos[j]++;
      rowind[pos] = i;
      x[pos] = v;
    }
  }

  std::free(colnnz);
  std::free(wpos);

  *colptr_out = colptr;
  *rowind_out = rowind;
  *x_out      = x;

  return (std::size_t)nnz;
}

/* C-linkage for LSMR f90 routines and double precision driver */
extern "C" {

void lsmr_c_set_options(
  const double* atol,
  const double* btol,
  const double* conlim,
  const int* itnlim,
  const int* nout,
  const int* localsize,
  const int* ctest);

void lsmr_c_step(
  const int* m,
  const int* n,
  int* action,
  double* u,
  double* v,
  const double* b,
  const double* damp,
  double* x,
  int* istop,
  int* itn,
  int* stat,
  double* normr,
  double* normA,
  double* condA,
  double* normb,
  double* normx,
  double* normAr);

} // extern "C"

inline void apply_A_colmajor(
  int m,
  int n,
  const double* A,
  const double* x,
  double* y)
{
  // y = y + A*x, A(i,j)=A[i + m*j].
  for (int j = 0; j < n; ++j)
  {
    const double xj = x[j];
    const double* Aj = A + static_cast<std::ptrdiff_t>(m) * j;

    for (int i = 0; i < m; ++i)
    {
      y[i] += Aj[i] * xj;
    }
  }
}

inline void apply_AT_colmajor(
  int m,
  int n,
  const double* A,
  const double* y,
  double* x)
{
  // x = x + A^T*y, A(i,j)=A[i + m*j].
  for (int j = 0; j < n; ++j)
  {
    const double* Aj = A + static_cast<std::ptrdiff_t>(m) * j;
    double sum = 0.0;

    for (int i = 0; i < m; ++i)
    {
      sum += Aj[i] * y[i];
    }

    x[j] += sum;
  }
}

inline void cleanup_lsmr_state(
  int m,
  int n,
  double* u,
  double* v,
  const double* b,
  double damp,
  double* x,
  int& istop,
  int& itn,
  int& stat,
  double& normr,
  double& normA,
  double& condA,
  double& normb,
  double& normx,
  double& normAr)
{
  int action = 10;

  lsmr_c_step(
    &m,
    &n,
    &action,
    u,
    v,
    b,
    &damp,
    x,
    &istop,
    &itn,
    &stat,
    &normr,
    &normA,
    &condA,
    &normb,
    &normx,
    &normAr);
}

template<class Real>
struct LsmrOptions
{
  Real damp = Real(0);
  Real atol = Real(1.0e-12);
  Real btol = Real(1.0e-12);
  Real conlim = Real(1.0e12);
  int itnlim = 500;
  int nout = 0;
  int localsize = 0;
  int ctest = 3;
};

template<class Real>
struct LsmrInfo
{
  int istop = -1;
  int itn = -1;
  int stat = 0;
  Real normr = Real(0);
  Real normA = Real(0);
  Real condA = Real(0);
  Real normb = Real(0);
  Real normx = Real(0);
  Real normAr = Real(0);
};

inline int lsmr_dense_solve_colmajor_double(
  int m,
  int n,
  const double* A_colmajor,
  const double* b,
  double* x,
  const LsmrOptions<double>& opt,
  LsmrInfo<double>* info)
{
  if (m <= 0 || n <= 0)
  {
    return -1;
  }

  if (A_colmajor == nullptr || b == nullptr || x == nullptr || info == nullptr)
  {
    return -2;
  }

  try
  {
    double* u = (double*) calloc(static_cast<std::size_t>(m), sizeof(double));
    double* v = (double*) calloc(static_cast<std::size_t>(n), sizeof(double));
    //std::vector<double> u(static_cast<std::size_t>(m), 0.0);
    //std::vector<double> v(static_cast<std::size_t>(n), 0.0);

    std::fill(x, x + n, 0.0);

    const double atol = opt.atol;
    const double btol = opt.btol;
    const double conlim = opt.conlim;
    const int itnlim = opt.itnlim;
    const int nout = opt.nout;
    const int localsize = opt.localsize;
    const int ctest = opt.ctest;
    const double damp = opt.damp;

    lsmr_c_set_options(
      &atol,
      &btol,
      &conlim,
      &itnlim,
      &nout,
      &localsize,
      &ctest);

    int action = 0;
    int istop = 0;
    int itn = 0;
    int stat = 0;

    double normr = 0.0;
    double normA = 0.0;
    double condA = 0.0;
    double normb = 0.0;
    double normx = 0.0;
    double normAr = 0.0;

    while (true)
    {
      lsmr_c_step(
        &m,
        &n,
        &action,
        u,
        v,
        b,
        &damp,
        x,
        &istop,
        &itn,
        &stat,
        &normr,
        &normA,
        &condA,
        &normb,
        &normx,
        &normAr);

      if (action == 0)
      {
        break;
      }

      if (action == 1)
      {
        // v = v + A^T*u.
        apply_AT_colmajor(m, n, A_colmajor, u, v);
      }
      else if (action == 2)
      {
        // u = u + A*v.
        apply_A_colmajor(m, n, A_colmajor, v, u);
      }
      else
      {
        cleanup_lsmr_state(
          m,
          n,
          u,
          v,
          b,
          damp,
          x,
          istop,
          itn,
          stat,
          normr,
          normA,
          condA,
          normb,
          normx,
          normAr);
        return -4;
      }
    }

    cleanup_lsmr_state(
      m,
      n,
      u,
      v,
      b,
      damp,
      x,
      istop,
      itn,
      stat,
      normr,
      normA,
      condA,
      normb,
      normx,
      normAr);

    info->istop = istop;
    info->itn = itn;
    info->stat = stat;
    info->normr = normr;
    info->normA = normA;
    info->condA = condA;
    info->normb = normb;
    info->normx = normx;
    info->normAr = normAr;
    if (u) { free(u); u = 0; }
    if (v) { free(v); v = 0; }
    return 0;
  }
  catch (...)
  {
    return -3;
  }
}

template<class Real>
inline LsmrOptions<double> to_double_options(const LsmrOptions<Real>& opt)
{
  LsmrOptions<double> out;
  out.damp = static_cast<double>(opt.damp);
  out.atol = static_cast<double>(opt.atol);
  out.btol = static_cast<double>(opt.btol);
  out.conlim = static_cast<double>(opt.conlim);
  out.itnlim = opt.itnlim;
  out.nout = opt.nout;
  out.localsize = opt.localsize;
  out.ctest = opt.ctest;
  return out;
}

template<class Real>
inline void copy_info_from_double(const LsmrInfo<double>& src, LsmrInfo<Real>* dst)
{
  if (dst == nullptr)
  {
    return;
  }

  dst->istop = src.istop;
  dst->itn = src.itn;
  dst->stat = src.stat;
  dst->normr = static_cast<Real>(src.normr);
  dst->normA = static_cast<Real>(src.normA);
  dst->condA = static_cast<Real>(src.condA);
  dst->normb = static_cast<Real>(src.normb);
  dst->normx = static_cast<Real>(src.normx);
  dst->normAr = static_cast<Real>(src.normAr);
}

} // namespace detail

} // namespace jsimplex

#endif // JDETAIL_H
