#include <jquad_tprod.hh>
#include <jquad_tprod_c.h>

using namespace jsimplex;

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

/* Public C API (runtime dispatch on D). */
extern "C"
{

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
    case 5: return quad_mapped_build_D<4>(n, X, W);
    default:
      return 0; // unsupported D
  }
}

} // extern "C"
