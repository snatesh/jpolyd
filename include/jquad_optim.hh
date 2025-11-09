#ifndef JQUAD_OPTIM_H
#define JQUAD_OPTIM_H

#include <cstdlib>
#include <cmath>
#include "jdetail.hh"
#include "jmat.hh"

namespace jsimplex
{

/* Compute Jacobi–Gaussian (optimized) quadrature on the d-simplex.

   D = 1 : exact Gauss–Jacobi on [0,1] for given \kappa = (k_1,k_2).
   D > 1 : optimized Gaussian-like rule via approximate joint eigenvalue 
           decomposition and nonlinear optimization (future work).

   Weight function:
     W_k(x) \propto x_1^{k_1−1/2} ⋯ x_d^{k_d−1/2} (1−\sum x_i)^{k_{d+1}−1/2}.
*/
template<int D, class Real>
struct QuadOptim;

/* ---------------- D = 1 specialization ---------------- */
template<class Real>
struct QuadOptim<1, Real>
{
  /* Construct an n-point Gauss–Jacobi rule on [0,1]
     for the normalized weight:
       W_κ(x) ∝ x^{k_1−1/2} (1−x)^{k_2−1/2}.
     Inputs:
       n      : number of points
       kappa  : array [2] (k_1,k_2)
     Outputs:
       t      : nodes in [0,1]
       w      : weights (sum ≈ 1)
     Returns:
       number of points (n) on success, 0 on failure.
  */
  static int build(unsigned int n, const Real* kappa, Real* t, Real* w)
  {
    if (!kappa || !t || !w || n == 0)
      return 0;

    gauss_jacobi_unit(n, kappa, t, w);
    return static_cast<int>(n);
  }

private:

  /* 1D Gauss-Jacobi on [0,1] for weight t^k_1 (1-t)^k_2,

     Inputs:
       n     : number of nodes
       kappa : k_1,k_2 > -1/2

     Outputs:
       t[0..n-1] : nodes in [0,1]
       w[0..n-1] : weights */

  static void gauss_jacobi_unit(unsigned int n, Real* kappa,
                                Real* t, Real* w)
  {
    /* For now, only float and double are supported by LAPACK. */
    static_assert(std::is_same<Real,float>::value  ||
                  std::is_same<Real,double>::value,
                  "gauss_jacobi_unit requires Real=float or Real=double");

    if (n == 0)
    {
      return;
    }

    // Jacobi parameters on [-1,1] for (1-u)^a (1+u)^b:
    // under change of variable, we find a = k_2 - 1/2, b = k_1-1/2
    Real a = kappa[1] - Real(0.5);
    Real b = kappa[0] - Real(0.5);

    // Build full Jacobi matrix J (n x n, row-major) in type Real.
    Real* J = (Real*) std::calloc(n * n, sizeof(Real));
    if (!J) { return; }

    jsimplex::JMat<1,Real>::build(n, a, b, J);

    // Extract diagonal d[0..n-1] and off-diagonal e[0..n-2]
    //   for the symmetric tridiagonal form.
    Real* d = (Real*) std::malloc(n * sizeof(Real));
    if (!d) { std::free(J); return; }

    Real* e = (Real*) std::malloc((n > 1 ? (n - 1) : 1) * sizeof(Real));
    if (!e) { std::free(J); std::free(d); return; }

    for (unsigned int i = 0; i < n; ++i)
    {
      d[i] = J[i + n * i];
      if (i + 1 < n)
      {
        e[i] = J[i + n * (i + 1)];
      }
    }

    // Eigenvectors Z (n x n), column-major for LAPACK. 
    Real* Z = (Real*) std::calloc(n * n, sizeof(Real));
    if (!Z) { std::free(J); std::free(d); std::free(e); return; }

    std::free(J);

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
    //   Weights: w_i = (first component of eigenvector i)^2.
    //   In column-major Z, first component is Z[0 + i*N]. 
    for (unsigned int i = 0; i < n; ++i)
    {
      Real u_i = d[i];
      Real v0  = Z[0 + i * N];   // first component of i-th eigenvector
      Real wi  = v0 * v0;

      t[i] = (u_i + Real(1.0)) * Real(0.5);  // map to [0,1] 
      w[i] = wi;
    }

    std::free(d);
    std::free(Z);
  }
}; // struct QuadOptim<1,Real>


/* ---------------- General D > 1 template ---------------- */
template<int D, class Real>
struct QuadOptim
{
  /* Placeholder for future implementation:
     Optimized Gaussian-like quadrature for the
     Jacobi weight on the D-simplex. */
  static int build(unsigned int n, const Real* kappa,
                   Real* points, Real* weights)
  {
    // TODO: Use JMat<D,Real> + optimization machinery (NLOPT).
    (void)n; (void)kappa; (void)points; (void)weights;
    return 0;
  }
};

} // namespace jsimplex

#endif // JQUAD_OPTIM_H
