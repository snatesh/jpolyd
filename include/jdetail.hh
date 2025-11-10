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
  static lapack_int run(lapack_int n, float* d, float* e,
                        float* Z, lapack_int ldz)
  {
    return LAPACKE_sstevd(LAPACK_COL_MAJOR, 'V', n, d, e, Z, ldz);
  }
};

template<>
struct LapackStevd<double> // double specialization
{
  static lapack_int run(lapack_int n, double* d, double* e,
                        double* Z, lapack_int ldz)
  {
    return LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', n, d, e, Z, ldz);
  }
};

/* Build n-point Gauss–Legendre quadrature on [0,1]:
   - Using the orthonormal Legendre Jacobi matrix (a = b = 0)
   - Nodes t[0..n-1] in [0,1]
   - Weights w[0..n-1], sum(w) ≈ 1 (normalized measure)

   This mirrors gauss_jacobi_unit, but with a=b=0 specialized,
   and works for Real = float or double via LapackStevd. */
template<class Real>
static void legendre_unit(unsigned int n, Real* t, Real* w)
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
