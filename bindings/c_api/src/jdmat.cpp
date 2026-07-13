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

template<int D>
static inline void js_dmat_build_tprod_natural_pruned_csc_dispatch(int n,
                                                                   unsigned int q,
                                                                   const double* kappa_src,
                                                                   int axis,
                                                                   int* nrow_out,
                                                                   int* ncol_out,
                                                                   int* nnz_out,
                                                                   int** colptr_out,
                                                                   int** rowind_out,
                                                                   double** x_out)
{
  int* colptr = nullptr;
  int* rowind = nullptr;
  double* x = nullptr;

  const std::size_t nnz = jsimplex::DMat<D,double>::build_tprod_natural_pruned_csc(
    n, q, kappa_src, axis, &colptr, &rowind, &x);

  const int ncol = jsimplex::Basis<D,double>::dim_Pi(n);
  const int nrow = (n > 0) ? jsimplex::Basis<D,double>::dim_Pi(n - 1) : 0;

  *nrow_out = nrow;
  *ncol_out = ncol;
  *nnz_out = (nnz > (std::size_t)std::numeric_limits<int>::max())
             ? -1
             : (int)nnz;

  *colptr_out = colptr;
  *rowind_out = rowind;
  *x_out = x;
}

int js_dmat_build_tprod_natural_pruned_csc(int D,
                                          int n,
                                          unsigned int q,
                                          const double* kappa_src,
                                          int axis,
                                          int* nrow_out,
                                          int* ncol_out,
                                          int* nnz_out,
                                          int** colptr_out,
                                          int** rowind_out,
                                          double** x_out)
{
  if (!kappa_src || !nrow_out || !ncol_out || !nnz_out ||
      !colptr_out || !rowind_out || !x_out || q == 0)
  {
    return 1;
  }

  // Policy you stated: error only for n < 0; allow n==0 (returns empty range)
  if (n < 0)
  {
    return 1;
  }

  if (D < 1 || D > JS_DMAT_MAX_D)
  {
    std::cerr << "js_dmat_build_tprod_natural_pruned_csc: unsupported D=" << D
              << " (compiled JS_DMAT_MAX_D=" << JS_DMAT_MAX_D << ")\n";
    return 2;
  }

  // initialize outputs to null on entry
  *nrow_out = 0;
  *ncol_out = 0;
  *nnz_out = 0;
  *colptr_out = nullptr;
  *rowind_out = nullptr;
  *x_out = nullptr;

  switch (D)
  {
    case 1:
      js_dmat_build_tprod_natural_pruned_csc_dispatch<1>(n, q, kappa_src, axis,
                                                         nrow_out, ncol_out, nnz_out,
                                                         colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 2:
      js_dmat_build_tprod_natural_pruned_csc_dispatch<2>(n, q, kappa_src, axis,
                                                         nrow_out, ncol_out, nnz_out,
                                                         colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 3:
      js_dmat_build_tprod_natural_pruned_csc_dispatch<3>(n, q, kappa_src, axis,
                                                         nrow_out, ncol_out, nnz_out,
                                                         colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 4:
      js_dmat_build_tprod_natural_pruned_csc_dispatch<4>(n, q, kappa_src, axis,
                                                         nrow_out, ncol_out, nnz_out,
                                                         colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    default:
      return 3;
  }
}

void js_dmat_csc_free(int* colptr, int* rowind, double* x)
{
  if (colptr) { std::free(colptr); }
  if (rowind) { std::free(rowind); }
  if (x)      { std::free(x); }
}
