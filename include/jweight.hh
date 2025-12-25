#ifndef JWEIGHT_H
#define JWEIGHT_H

#include <cmath>    // pow, log, exp, lgamma

namespace jsimplex
{

template<int D, class Real>
struct Weight
{
  static_assert(D > 0, "Dimension D must be > 0");

  /* Returns x_{D+1} = 1 - sum_{i=1}^D x_i */
  static inline Real x_last(const Real* x)
  {
    Real s = Real(0);
    for (int i = 0; i < D; ++i)
    {
      s += x[i];
    }
    return Real(1) - s;
  }

  /* Monomial part (no gamma ratio):
     gamma_{i=1}^D x_i^{kappa_i - 1/2} * (1 - |x|)^{kappa_{D+1} - 1/2} */
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

  /* Normalizer w_kappa:
     w = gamma(|k| + (D+1)/2) /prod_i^{D+1}( gamma(k_i + 1/2))
     Computed in log-domain for numerical stability. */
  static inline Real w_kappa(const Real* kappa)
  {
    // assert(kappa);
    Real sumK = Real(0);
    for (int i = 0; i <= D; ++i)
    {
      sumK += kappa[i];
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

  /* Normalized weight W_kappa (integrates to 1 over the D-simplex):
     W = w_kappa * monomial */
  static inline Real eval(const Real* x, const Real* kappa)
  {
    return w_kappa(kappa) * monomial(x, kappa);
  }


  /* Evaluate at npts points from a single flat buffer X.
     Indexing: X[p*ld_point + j*ld_dim],  p=0..npts-1, j=0..D-1.
     Works for AoS (ld_point=D, ld_dim=1) and SoA/column-major (ld_point=1, ld_dim=npts). */
  static inline void eval(const Real* X, int ld_point, int ld_dim,
                          int npts, const Real* kappa, Real* out)
  {
    const Real wk   = w_kappa(kappa);
    const Real half = Real(0.5);
 
    for (int p = 0; p < npts; ++p)
    {
      const int base = p * ld_point;
  
      Real s = Real(0);
      Real v = Real(1);
  
      for (int j = 0; j < D; ++j)
      {
        const Real xpj = X[base + j * ld_dim];
        v *= std::pow(xpj, kappa[j] - half);
        s += xpj;
      }
  
      const Real xl = Real(1) - s;
      v *= std::pow(xl, kappa[D] - half);
  
      out[p] = wk * v;
    }
  }

};

} // namespace jsimplex

#endif // JWEIGHT_H

