#ifndef JWEIGHT_H
#define JWEIGHT_H

#include <cmath>    // pow, log, exp, lgamma

namespace jsimplex
{

template<int D, class Real>
struct Weight
{
  static_assert(D >= 1, "Dimension D must be >= 1");

  // Returns x_{D+1} = 1 - sum_{i=1}^D x_i
  static inline Real x_last(const Real* x)
  {
    Real s = Real(0);
    for (int i = 0; i < D; ++i)
    {
      s += x[i];
    }
    return Real(1) - s;
  }

  // Monomial part (no gamma ratio):
  // Π_{i=1}^D x_i^{kappa_i - 1/2} * (1 - |x|)^{kappa_{D+1} - 1/2}
  static inline Real monomial(const Real* x, const Real* kappa)
  {

    Real v = Real(1);
    for (int i = 0; i < D; ++i)
    {
      v *= std::pow(x[i], kappa[i] - Real(0.5));
    }
    v *= std::pow(x_last(x), kappa[D] - Real(0.5));
    return v;
  }

  // Normalizer w_kappa:
  // w = Γ(|k| + (D+1)/2) / Π Γ(k_i + 1/2)
  // Computed in log-domain for numerical stability.
  static inline Real w_kappa(const Real* kappa)
  {
    // assert(kappa);
    Real sumK = Real(0);
    for (int i = 0; i <= D; ++i)
    {
      sumK += kappa[i];
      // assert(kappa[i] > Real(-0.5));
    }
    const Real A = sumK + Real(D + 1) * Real(0.5);

    const Real lgNum = std::lgamma(A);
    Real lgDen = Real(0);
    for (int i = 0; i <= D; ++i)
    {
      lgDen += std::lgamma(kappa[i] + Real(0.5));
    }
    return std::exp(lgNum - lgDen);
  }

  // Normalized weight W_kappa (integrates to 1 over the D-simplex):
  // W = w_kappa * monomial
  static inline Real eval(const Real* x, const Real* kappa)
  {
    return w_kappa(kappa) * monomial(x, kappa);
  }

  static inline Real monomial_tolerant(const Real* x, const Real* kappa)
  {
    const Real eps = Real(64) * std::numeric_limits<Real>::epsilon(); // tiny, type-aware
    Real v = Real(1);
    for (int i = 0; i < D; ++i)
    {
      const Real base = x[i] < eps ? eps : x[i];
      v *= std::pow(base, kappa[i] - Real(0.5));
    }
    Real xl = x_last(x);
    xl = (xl < eps) ? eps : xl;
    v *= std::pow(xl, kappa[D] - Real(0.5));
    return v;
  }

  static inline Real eval_tolerant(const Real* x, const Real* kappa)
  {
    return w_kappa(kappa) * monomial_tolerant(x, kappa);
  }


};

} // namespace jsimplex

#endif // JWEIGHT_H

