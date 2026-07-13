#include <jweight.hh>
#include <jweight_c.h>

using namespace jsimplex;

/* Internal helpers: template dispatch by D */
template<int D>
static inline double w_kappa_D(const double* kappa)
{
  return Weight<D, double>::w_kappa(kappa);
}

template<int D>
static inline void eval_D(const double* X, int ld_point, int ld_dim,
                          int npts, const double* kappa, double* out)
{
  return Weight<D, double>::eval(X, ld_point, ld_dim, npts, kappa, out);
}


/* Public C API (runtime dispatch). Extend the switch as needed. */
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
    case 5: return w_kappa_D<5>(kappa);
    default: return 0.0;
  }
}

void jweight_eval(const double* X, int ld_point, int ld_dim,
                    int npts, const double* kappa, double* out, int D)
{
  switch (D)
  {
    case 1: eval_D<1>(X, ld_point, ld_dim, npts, kappa, out); return;
    case 2: eval_D<2>(X, ld_point, ld_dim, npts, kappa, out); return;
    case 3: eval_D<3>(X, ld_point, ld_dim, npts, kappa, out); return;
    case 4: eval_D<4>(X, ld_point, ld_dim, npts, kappa, out); return;
    case 5: eval_D<5>(X, ld_point, ld_dim, npts, kappa, out); return;
    default: return; // no-op
  }
}

} // extern "C"
