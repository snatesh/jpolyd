#include "jbasis.hh"
#include "jbasis_c.h"

using namespace jsimplex;

/* Internal template dispatchers */
template<int D>
static inline void jbasis_eval_all_D(const double* X,
                                     int ld_point,
                                     int ld_dim,
                                     int npts,
                                     const double* kappa,
                                     int n,
                                     const int* alpha_table,
                                     const int* tail_deg,
                                     const double* inv_h,
                                     double* V,
                                     int ld_V,
                                     double* dV)
{
  Basis<D,double>::eval_all(X,
                            ld_point,
                            ld_dim,
                            npts,
                            kappa,
                            n,
                            alpha_table,
                            tail_deg,
                            inv_h,
                            V,
                            ld_V,
                            dV);
}

template<int D>
static inline int jbasis_dim_Pi_D(int n)
{
  return Basis<D,double>::dim_Pi(n);
}

template<int D>
static inline void jbasis_build_structures_D(const double* kappa,
                                             int n,
                                             int* alpha_table,
                                             int* tail_deg,
                                             double* inv_h)
{
  int M = Basis<D,double>::dim_Pi(n);

  Basis<D,double>::build_alpha_table(n, alpha_table);
  Basis<D,double>::build_tail_deg(n, alpha_table, tail_deg);

  for (int m = 0; m < M; ++m)
  {
    const int* alpha = alpha_table + m * D;
    inv_h[m] = Basis<D,double>::inv_h_alpha(alpha, kappa);
  }
}


/* Public C API */
extern "C"
{

void jbasis_eval_all(const double* X,
                     int ld_point,
                     int ld_dim,
                     int npts,
                     const double* kappa,
                     int D,
                     int n,
                     const int* alpha_table,
                     const int* tail_deg,
                     const double* inv_h,
                     double* V,
                     int ld_V,
                     double* dV)
{
  switch (D)
  {
    case 1:
      jbasis_eval_all_D<1>(X, ld_point, ld_dim,
                           npts, kappa, n,
                           alpha_table, tail_deg,
                           inv_h, V, ld_V, dV);
      break;
    case 2:
      jbasis_eval_all_D<2>(X, ld_point, ld_dim,
                           npts, kappa, n,
                           alpha_table, tail_deg,
                           inv_h, V, ld_V, dV);
      break;
    case 3:
      jbasis_eval_all_D<3>(X, ld_point, ld_dim,
                           npts, kappa, n,
                           alpha_table, tail_deg,
                           inv_h, V, ld_V, dV);
      break;
    case 4:
      jbasis_eval_all_D<4>(X, ld_point, ld_dim,
                           npts, kappa, n,
                           alpha_table, tail_deg,
                           inv_h, V, ld_V, dV);
      break;
    case 5:
      jbasis_eval_all_D<5>(X, ld_point, ld_dim,
                           npts, kappa, n,
                           alpha_table, tail_deg,
                           inv_h, V, ld_V, dV);
      break;
    default:
      // Unsupported D: no-op
      break;
  }
}

int jbasis_dim_Pi(int D, int n)
{
  switch (D)
  {
    case 1: return jbasis_dim_Pi_D<1>(n);
    case 2: return jbasis_dim_Pi_D<2>(n);
    case 3: return jbasis_dim_Pi_D<3>(n);
    case 4: return jbasis_dim_Pi_D<4>(n);
    case 5: return jbasis_dim_Pi_D<5>(n);
    default:
      return 0;
  }
}

void jbasis_build_structures(const double* kappa,
                             int D,
                             int n,
                             int* alpha_table,
                             int* tail_deg,
                             double* inv_h)
{
  switch (D)
  {
    case 1:
      jbasis_build_structures_D<1>(kappa, n,
                                   alpha_table, tail_deg,
                                   inv_h);
      break;
    case 2:
      jbasis_build_structures_D<2>(kappa, n,
                                   alpha_table, tail_deg,
                                   inv_h);
      break;
    case 3:
      jbasis_build_structures_D<3>(kappa, n,
                                   alpha_table, tail_deg,
                                   inv_h);
      break;
    case 4:
      jbasis_build_structures_D<4>(kappa, n,
                                   alpha_table, tail_deg,
                                   inv_h);
      break;
    case 5:
      jbasis_build_structures_D<5>(kappa, n,
                                   alpha_table, tail_deg,
                                   inv_h);
      break;
    default:
      // unsupported D
      break;
  }
}

} // extern "C"
