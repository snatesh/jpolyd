#include <jquad_tprod.hh>
#include <jquad_tprod_c.h>

using namespace jsimplex;

/* Internal: safe integer power n^D for small D, returns 0 on overflow-ish */
static inline int ipow_u32(unsigned int n, int D)
{
  if (D <= 0 || n == 0) { return 0; }
  unsigned long long acc = 1ULL;
  for (int i = 0; i < D; ++i)
  {
    acc *= (unsigned long long) n;
    if (acc > 0x7fffffffULL) { return 0; } // clamp to 32-bit int range
  }
  return (int)acc;
}

/* Internal helper: template dispatch by D. */
template<int D>
static inline int quad_mapped_build_D(int n, double* X, double* W)
{
  if (n <= 0 || !X || !W)
  {
    return 0;
  }
  return QuadMapped<D, double>::build(static_cast<unsigned int>(n), X, W);
}

template<int D>
static inline int quad_mapped_build_kappa_D(const double* kappa, unsigned int n,
                                            double* points, double* weights)
{
  return QuadMapped<D, double>::build_kappa(n, kappa, points, weights);
}


/* Public C API (runtime dispatch on D). */
extern "C"
{

int jquad_mapped_npoints(int D, unsigned int n)
{
  return ipow_u32(n, D);
}


int jquad_mapped_build(int D, int n, double* X, double* W)
{
  if (!X || !W) { return 0; }
  if (n <= 0)   { return 0; }

  switch (D)
  {
    case 1: return quad_mapped_build_D<1>(n, X, W);
    case 2: return quad_mapped_build_D<2>(n, X, W);
    case 3: return quad_mapped_build_D<3>(n, X, W);
    case 4: return quad_mapped_build_D<4>(n, X, W);
    case 5: return quad_mapped_build_D<5>(n, X, W);
    default:
      return 0; // unsupported D
  }
}

int jquad_mapped_build_kappa(const double* kappa, int D, unsigned int n,
                            double* points, double* weights)
{
  if (!kappa || !points || !weights || D <= 0 || n == 0)
  {
    return 0;
  }

  // Runtime dispatch on D
  switch (D)
  {
    case 1: return quad_mapped_build_kappa_D<1>(kappa, n, points, weights);
    case 2: return quad_mapped_build_kappa_D<2>(kappa, n, points, weights);
    case 3: return quad_mapped_build_kappa_D<3>(kappa, n, points, weights);
    case 4: return quad_mapped_build_kappa_D<4>(kappa, n, points, weights);
    case 5: return quad_mapped_build_kappa_D<5>(kappa, n, points, weights);
    default:
      // no-op for unsupported D
      return 0;
  }
}


} // extern "C"
