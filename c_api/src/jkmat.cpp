#include <jkmat.h>

#include <cstdlib>
#include <iostream>

#include <jkmat.hh>

// Change this to whatever max dimension you want to compile in.
static constexpr int JS_KMAT_MAX_D = 4;

template<int D>
static inline void js_kmat_build_tprod_dispatch(int n,
                                                unsigned int q,
                                                const double* kappa_src,
                                                const double* kappa_tgt,
                                                double* K_out)
{
  jsimplex::KMat<D,double>::build_tprod(n, q, kappa_src, kappa_tgt, K_out);
}

int js_kmat_build_tprod(int D,
                        int n,
                        unsigned int q,
                        const double* kappa_src,
                        const double* kappa_tgt,
                        double* K_out)
{
  if (!kappa_src || !kappa_tgt || !K_out || n < 0 || q == 0)
  {
    return 1;
  }

  if (D < 1 || D > JS_KMAT_MAX_D)
  {
    std::cerr << "js_kmat_build_tprod: unsupported D=" << D
              << " (compiled JS_KMAT_MAX_D=" << JS_KMAT_MAX_D << ")\n";
    return 2;
  }

  switch (D)
  {
    case 1:
      js_kmat_build_tprod_dispatch<1>(n, q, kappa_src, kappa_tgt, K_out);
      return 0;

    case 2:
      js_kmat_build_tprod_dispatch<2>(n, q, kappa_src, kappa_tgt, K_out);
      return 0;

    case 3:
      js_kmat_build_tprod_dispatch<3>(n, q, kappa_src, kappa_tgt, K_out);
      return 0;

    case 4:
      js_kmat_build_tprod_dispatch<4>(n, q, kappa_src, kappa_tgt, K_out);
      return 0;

    default:
      // Should never hit due to range check above.
      return 3;
  }
}

