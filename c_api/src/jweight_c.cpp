#include <jweight.hh>
#include <jweight_c.h>

using namespace jsimplex;

// Internal helpers: template dispatch by D
template<int D>
static inline double w_kappa_D(const double* kappa)
{
  return Weight<D, double>::w_kappa(kappa);
}

template<int D>
static inline double eval_D(const double* x, const double* kappa)
{
  return Weight<D, double>::eval(x, kappa);
}

template<int D>
static inline double monomial_D(const double* x, const double* kappa)
{
  return Weight<D, double>::monomial(x, kappa);
}

template<int D>
static inline double x_last_D(const double* x)
{
  return Weight<D, double>::x_last(x);
}

// Public C API (runtime dispatch). Extend the switch as needed.
extern "C"
{

double jweight_w_kappa(const double* kappa, int D)
{
  switch (D)
  {
    case 1: return w_kappa_D<1>(kappa);
    case 2: return w_kappa_D<2>(kappa);
    case 3: return w_kappa_D<3>(kappa);
    case 4: return w_kappa_D<4>(kappa);
    default: return 0.0;
  }
}

double jweight_eval(const double* x, const double* kappa, int D)
{
  switch (D)
  {
    case 1: return eval_D<1>(x, kappa);
    case 2: return eval_D<2>(x, kappa);
    case 3: return eval_D<3>(x, kappa);
    case 4: return eval_D<4>(x, kappa);
    default: return 0.0;
  }
}

double jweight_monomial(const double* x, const double* kappa, int D)
{
  switch (D)
  {
    case 1: return monomial_D<1>(x, kappa);
    case 2: return monomial_D<2>(x, kappa);
    case 3: return monomial_D<3>(x, kappa);
    case 4: return monomial_D<4>(x, kappa);
    default: return 0.0;
  }
}

double jweight_x_last(const double* x, int D)
{
  switch (D)
  {
    case 1: return x_last_D<1>(x);
    case 2: return x_last_D<2>(x);
    case 3: return x_last_D<3>(x);
    case 4: return x_last_D<4>(x);
    default: return 0.0;
  }
}

} // extern "C"
