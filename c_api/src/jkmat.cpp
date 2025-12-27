#include <jkmat.h>

#include <cstdlib>
#include <iostream>
#include <limits>
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
  jsimplex::KMat<D,double>::build_tprod_pruned_dense(n, q, kappa_src, kappa_tgt, K_out);
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

template<int D>
static inline void js_kmat_build_tprod_pruned_csc_dispatch(int n,
                                                           unsigned int q,
                                                           const double* kappa_src,
                                                           const double* kappa_tgt,
                                                           int* nrow_out,
                                                           int* ncol_out,
                                                           int* nnz_out,
                                                           int** colptr_out,
                                                           int** rowind_out,
                                                           double** x_out)
{
  int* colptr = nullptr;
  int* rowind = nullptr;
  double* x   = nullptr;

  const std::size_t nnz = jsimplex::KMat<D,double>::build_tprod_pruned_csc(
    n, q, kappa_src, kappa_tgt, &colptr, &rowind, &x);

  const int M = jsimplex::Basis<D,double>::dim_Pi(n);

  *nrow_out = M;
  *ncol_out = M;

  if (nnz > (std::size_t)std::numeric_limits<int>::max())
  {
    // signal overflow; free and return through caller via nnz_out<0
    std::free(colptr);
    std::free(rowind);
    std::free(x);
    *nnz_out = -1;
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *x_out = nullptr;
    return;
  }

  *nnz_out = (int)nnz;
  *colptr_out = colptr;
  *rowind_out = rowind;
  *x_out = x;
}

int js_kmat_build_tprod_pruned_csc(int D,
                                  int n,
                                  unsigned int q,
                                  const double* kappa_src,
                                  const double* kappa_tgt,
                                  int* nrow_out,
                                  int* ncol_out,
                                  int* nnz_out,
                                  int** colptr_out,
                                  int** rowind_out,
                                  double** x_out)
{
  if (!kappa_src || !kappa_tgt ||
      !nrow_out || !ncol_out || !nnz_out ||
      !colptr_out || !rowind_out || !x_out ||
      n < 0 || q == 0)
  {
    return 1;
  }

  if (D < 1 || D > JS_KMAT_MAX_D)
  {
    std::cerr << "js_kmat_build_tprod_pruned_csc: unsupported D=" << D
              << " (compiled JS_KMAT_MAX_D=" << JS_KMAT_MAX_D << ")\n";
    return 2;
  }

  // Initialize outputs
  *nrow_out = 0;
  *ncol_out = 0;
  *nnz_out = 0;
  *colptr_out = nullptr;
  *rowind_out = nullptr;
  *x_out = nullptr;

  switch (D)
  {
    case 1:
      js_kmat_build_tprod_pruned_csc_dispatch<1>(n, q, kappa_src, kappa_tgt,
                                                 nrow_out, ncol_out, nnz_out,
                                                 colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 2:
      js_kmat_build_tprod_pruned_csc_dispatch<2>(n, q, kappa_src, kappa_tgt,
                                                 nrow_out, ncol_out, nnz_out,
                                                 colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 3:
      js_kmat_build_tprod_pruned_csc_dispatch<3>(n, q, kappa_src, kappa_tgt,
                                                 nrow_out, ncol_out, nnz_out,
                                                 colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    case 4:
      js_kmat_build_tprod_pruned_csc_dispatch<4>(n, q, kappa_src, kappa_tgt,
                                                 nrow_out, ncol_out, nnz_out,
                                                 colptr_out, rowind_out, x_out);
      return (*nnz_out < 0) ? 4 : 0;

    default:
      return 3;
  }
}

void js_kmat_csc_free(int* colptr, int* rowind, double* x)
{
  if (colptr) { std::free(colptr); }
  if (rowind) { std::free(rowind); }
  if (x)      { std::free(x); }
}

