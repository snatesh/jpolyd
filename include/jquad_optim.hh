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

    detail::gauss_jacobi_unit<Real>(n, kappa, t, w);
    return static_cast<int>(n);
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
