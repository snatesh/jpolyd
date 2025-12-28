#include <jmat.hh>
#include <jbasis.hh>
#include <jdetail.hh>
#include <jquad_tprod.hh>
#include <jmat_c.h>

using namespace jsimplex;

/* Internal helpers: template dispatch by D */

template<int D>
static inline int jmat_dim_Pi_D(int n)
{
  if (n < 0)
  {
    return 0;
  }
  return Basis<D,double>::dim_Pi(n);
}

template<int D>
static inline int jmat_build_D(const double* kappa,
                               int n,
                               double* J_all)
{
  if (!kappa || !J_all || n < 0)
  {
    return 0;
  }

  JMat<D,double>::build_pruned(n, kappa, n+1, J_all);
  return Basis<D,double>::dim_Pi(n);
}


template<int D>
static inline int jmat_build_coord_pruned_csc_D(const double* kappa,
                                                int n,
                                                unsigned int nquad,
                                                int coord,
                                                int** colptr_out,
                                                int** rowind_out,
                                                double** x_out,
                                                int* N_out,
                                                int* nnz_out)
{
  if (!kappa || !colptr_out || !rowind_out || !x_out || !N_out || !nnz_out)
  {
    return 1;
  }
  if (n < 0 || nquad == 0)
  {
    return 2;
  }
  if (coord < 0 || coord >= D)
  {
    return 3;
  }

  const int N = Basis<D,double>::dim_Pi(n);

  int* colptr = nullptr;
  int* rowind = nullptr;
  double* x   = nullptr;

  const std::size_t nnz_s =
    JMat<D,double>::build_pruned_csc(n, kappa, nquad, coord, &colptr, &rowind, &x);

  // Defensive: nnz fits in int?
  if (nnz_s > (std::size_t)std::numeric_limits<int>::max())
  {
    // avoid leaking
    if (colptr) std::free(colptr);
    if (rowind) std::free(rowind);
    if (x)      std::free(x);
    return 4;
  }

  *colptr_out = colptr;
  *rowind_out = rowind;
  *x_out      = x;
  *N_out      = N;
  *nnz_out    = (int)nnz_s;

  return 0;
}

/* Public C API */

extern "C"
{

int jmat_dim_Pi(int D, int n)
{
  switch (D)
  {
    case 1: return jmat_dim_Pi_D<1>(n);
    case 2: return jmat_dim_Pi_D<2>(n);
    case 3: return jmat_dim_Pi_D<3>(n);
    case 4: return jmat_dim_Pi_D<4>(n);
    case 5: return jmat_dim_Pi_D<5>(n);
    default: return 0;
  }
}

int jmat_build(const double* kappa,
               int D,
               int n,
               double* J_all)
{
  switch (D)
  {
    case 1: return jmat_build_D<1>(kappa, n, J_all);
    case 2: return jmat_build_D<2>(kappa, n, J_all);
    case 3: return jmat_build_D<3>(kappa, n, J_all);
    case 4: return jmat_build_D<4>(kappa, n, J_all);
    case 5: return jmat_build_D<5>(kappa, n, J_all);
    default: return 0;
  }
}

int jmat_build_coord_pruned_csc(const double* kappa,
                                int D,
                                int n,
                                unsigned int nquad,
                                int coord,
                                int** colptr_out,
                                int** rowind_out,
                                double** x_out,
                                int* N_out,
                                int* nnz_out)
{
  // Basic validation at the C boundary
  if (!kappa || !colptr_out || !rowind_out || !x_out || !N_out || !nnz_out)
  {
    return 1;
  }
  *colptr_out = nullptr;
  *rowind_out = nullptr;
  *x_out      = nullptr;
  *N_out      = 0;
  *nnz_out    = 0;

  if (n < 0 || nquad == 0)
  {
    return 2;
  }
  if (D < 1 || D > 5)
  {
    return 5;
  }

  switch (D)
  {
    case 1: return jmat_build_coord_pruned_csc_D<1>(kappa, n, nquad, coord,
                                                    colptr_out, rowind_out, x_out, N_out, nnz_out);
    case 2: return jmat_build_coord_pruned_csc_D<2>(kappa, n, nquad, coord,
                                                    colptr_out, rowind_out, x_out, N_out, nnz_out);
    case 3: return jmat_build_coord_pruned_csc_D<3>(kappa, n, nquad, coord,
                                                    colptr_out, rowind_out, x_out, N_out, nnz_out);
    case 4: return jmat_build_coord_pruned_csc_D<4>(kappa, n, nquad, coord,
                                                    colptr_out, rowind_out, x_out, N_out, nnz_out);
    case 5: return jmat_build_coord_pruned_csc_D<5>(kappa, n, nquad, coord,
                                                    colptr_out, rowind_out, x_out, N_out, nnz_out);
    default:
      return 5;
  }
}

void jmat_free(void* p)
{
  std::free(p);
}

} // extern "C"
