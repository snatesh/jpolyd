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

  JMat<D,double>::build(n, kappa, n+1, J_all);
  return Basis<D,double>::dim_Pi(n);
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

} // extern "C"
