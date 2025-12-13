#include <cstdlib>
#include <iostream>

#include "jdmat.h"
#include "jdmat.hh"

// Change this to whatever max dimension you want compiled in.
static constexpr int JS_DMAT_MAX_D = 4;

template<int D>
static inline void js_dmat_build_tprod_dispatch(int n,
                                                unsigned int q,
                                                const double* kappa_src,
                                                const double* kappa_rng,
                                                int axis,
                                                double* D_out)
{
  jsimplex::DMat<D,double>::build_tprod(n, q, kappa_src, kappa_rng, axis, D_out);
}

int js_dmat_build_tprod(int D,
                        int n,
                        unsigned int q,
                        const double* kappa_src,
                        const double* kappa_rng,
                        int axis,
                        double* D_out)
{
  if (!kappa_src || !kappa_rng || !D_out || n < 1 || q == 0)
  {
    return 1;
  }

  if (D < 1 || D > JS_DMAT_MAX_D)
  {
    std::cerr << "js_dmat_build_tprod: unsupported D=" << D
              << " (compiled JS_DMAT_MAX_D=" << JS_DMAT_MAX_D << ")\n";
    return 2;
  }

  switch (D)
  {
    case 1:
      js_dmat_build_tprod_dispatch<1>(n, q, kappa_src, kappa_rng, axis, D_out);
      return 0;

    case 2:
      js_dmat_build_tprod_dispatch<2>(n, q, kappa_src, kappa_rng, axis, D_out);
      return 0;

    case 3:
      js_dmat_build_tprod_dispatch<3>(n, q, kappa_src, kappa_rng, axis, D_out);
      return 0;

    case 4:
      js_dmat_build_tprod_dispatch<4>(n, q, kappa_src, kappa_rng, axis, D_out);
      return 0;

    default:
      return 3;
  }
}


template<int D>
static inline void js_dmat_build_tprod_natural_pruned_dispatch(int n,
                                                               unsigned int q,
                                                               const double* kappa_src,
                                                               int axis,
                                                               double* D_out)
{
  jsimplex::DMat<D,double>::build_tprod_natural_pruned(n, q, kappa_src, axis, D_out);
}

int js_dmat_build_tprod_natural_pruned(int D,
                                       int n,
                                       unsigned int q,
                                       const double* kappa_src,
                                       int axis,
                                       double* D_out)
{
  if (!kappa_src || !D_out || n < 1 || q == 0)
  {
    return 1;
  }
  if (D < 1 || D > JS_DMAT_MAX_D)
  {
    return 2;
  }

  switch (D)
  {
    case 1:
      js_dmat_build_tprod_natural_pruned_dispatch<1>(n, q, kappa_src, axis, D_out);
      return 0;
    case 2:
      js_dmat_build_tprod_natural_pruned_dispatch<2>(n, q, kappa_src, axis, D_out);
      return 0;
    case 3:
      js_dmat_build_tprod_natural_pruned_dispatch<3>(n, q, kappa_src, axis, D_out);
      return 0;
    case 4:
      js_dmat_build_tprod_natural_pruned_dispatch<4>(n, q, kappa_src, axis, D_out);
      return 0;
    default:
      return 3;
  }
}


