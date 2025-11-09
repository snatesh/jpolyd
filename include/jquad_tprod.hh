#ifndef JQUAD_TPROD_H
#define JQUAD_TPROD_H

#include <jmat.hh>
#include <cstdlib>
#include <cmath>
#include <jdetail.hh>

/* Mapped tensor-product quadrature on the D-simplex
   for the normalized unit weight function

   Construction:
   - Use stick-breaking map t in [0,1]^D -> x in simplex T^D.
   - On each axis j = 0..D-1, we build a 1D Gauss-Legendre rule
     on [0,1] and form the tensor product rule, mapped to the simplex. */

namespace jsimplex
{

template<int D, class Real>
struct QuadMapped
{


  /* Number of tensor points for order n on each axis: n^D. */
  static unsigned int npoints(unsigned int n)
  {
    unsigned int res = 1;
    for (int j = 0; j < D; ++j)
    {
      res *= n;
    }
    return res;
  }

  /* Build a mapped tensor-product quadrature rule on the
     D-simplex for Lebesgue measure.

     Inputs:
       n       : number of 1D points per axis

     Outputs:
       points  : length npts * D, where npts = n^D.
                 Layout: points[p*D + j] is j-th coord of p-th point
       weights : length npts, Lebesgue weights on the simplex

     The construction:
       1) Build a single 1D Legendre rule on [0,1] (kappa \equiv 1/2).
       2) Form the tensor product over D axes.
       3) Map (t_1,...,t_D) in [0,1]^D to simplex coords via
          x_1 = t_1
          x_2 = (1 - t_1) t_2
          ...
          x_D = (1 - t_1) ... (1 - t_{D-1}) t_D
          and include the Jacobian
          J(t) = prod_{j=1}^{D-1} (1 - t_j)^{D-j}.

     The resulting rule approximates
       \int_{T^D} f(x) dx \approx \sum_{p=0}^{npts-1} weights[p] f(points_p).
  */
  static int build(unsigned int n, Real* points, Real* weights)
  {
    if (!points || !weights) { return 0; }
    if (n == 0) { return 0; }

    // 1D Legendre rule on [0,1] corresponds to kappa = (1/2, 1/2). 
    Real kappa_unit[2];
    kappa_unit[0] = Real(0.5);
    kappa_unit[1] = Real(0.5);

    Real* t1 = (Real*) std::malloc(n * sizeof(Real));
    Real* w1 = (Real*) std::malloc(n * sizeof(Real));
    if (!t1 || !w1)
    {
      if (t1) { std::free(t1); }
      if (w1) { std::free(w1); }
      return 0;
    }

    detail::legendre_unit<Real>(n, t1, w1);

    // Total number of points: n^D. 
    unsigned int npts = 1;
    for (int j = 0; j < D; ++j)
    {
      npts *= n;
    }

    // Multi-index idx[0..D-1] running over {0,..,n-1}^D. 
    unsigned int idx[D];
    for (int j = 0; j < D; ++j)
    {
      idx[j] = 0;
    }

    for (unsigned int p = 0; p < npts; ++p)
    {
      Real t[D];
      Real wprod = Real(1.0);

      // gather 1D nodes / weights for this tensor index
      for (int j = 0; j < D; ++j)
      {
        t[j]     = t1[idx[j]];
        wprod   *= w1[idx[j]];
      }

      // Duffy / stick-breaking map and Jacobian
      Real rem = Real(1.0);
      Real J   = Real(1.0);

      for (int j = 0; j < D; ++j)
      {
        Real tj = t[j];

        // x_j = rem * t_j
        points[p * D + j] = rem * tj;

        // Jacobian factor: for j = 0..D-2, multiply (1 - t_j)^(D - j - 1)
        if (j < D - 1)
        {
          J *= std::pow(Real(1.0) - tj, Real(D - j - 1));
        }

        rem *= (Real(1.0) - tj);
      }

      weights[p] = wprod * J;

      // increment multi-index idx (lexicographic)
      for (int j = D - 1; j >= 0; --j)
      {
        idx[j] += 1;
        if (idx[j] < n)
        {
          break;
        }
        idx[j] = 0;
      }
    }

    std::free(t1);
    std::free(w1);

    return (int)npts;
  }


  /* Integrate with a prebuilt rule; function-pointer overload. */
  static Real integrate_with_rule(const Real* points,
                                  const Real* weights,
                                  unsigned int npts,
                                  Real (*f)(const Real* x))
  {
    Real sum = Real(0.0);

    for (unsigned int p = 0; p < npts; ++p)
    {
      const Real* xp = points + p * D;
      sum += weights[p] * f(xp);
    }

    return sum;
  }

  /* Integrate with a prebuilt rule; templated functor overload. */
  template<class F>
  static Real integrate_with_rule(const Real* points,
                                  const Real* weights,
                                  unsigned int npts, F&& f)
  {
    Real sum = Real(0.0);

    for (unsigned int p = 0; p < npts; ++p)
    {
      const Real* xp = points + p * D;
      sum += weights[p] * f(xp);
    }

    return sum;
  }

}; // struct QuadMapped

} // namespace jsimplex




#endif //JQUAD_TPROD_H
