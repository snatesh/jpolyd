#ifndef JDETAIL_H
#define JDETAIL_H

#include <type_traits>
#include <lapacke.h>
#include <cblas.h>

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


} // namespace detail

} // namespace jsimplex

#endif // JDETAIL_H
