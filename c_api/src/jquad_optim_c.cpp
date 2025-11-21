#include <jquad_optim_c.h>
#include <jquad_optim.hh>   // QuadOptimizer<D, Real>, QuadOptimOptions<D,Real>

extern "C"
int jquad_optimize(int D,
                   int node_deg,
                   int m_basis,
                   const double* kappa,
                   double* z_io,
                   double* V_opt,
                   int max_nlopt_eval,
                   int max_gn_iter,
                   double gn_step,
                   double tol,
                   double tol_up,
                   int verbose)
{
  if (!kappa || !z_io)
  {
    return -1;
  }

  if (node_deg <= 0 || D <= 0)
  {
    return -2;
  }

  /* For now we support a small set of D explicitly.
     You can extend this switch with more dimensions as needed. */

  int status = -99;

  switch (D)
  {
    case 1:
    {
      jsimplex::QuadOptimOptions<1,double> opts;
      /* For D=1 these are ignored, but set them anyway. */
      opts.max_nlopt_eval = max_nlopt_eval;
      opts.max_gn_iter    = max_gn_iter;
      opts.gn_step        = static_cast<double>(gn_step);
      opts.tol            = static_cast<double>(tol);
      opts.tol_up         = static_cast<double>(tol_up);
      opts.verbose        = (verbose != 0);

      status = jsimplex::QuadOptimizer<1,double>::optimize(
                  node_deg,
                  m_basis,
                  kappa,
                  z_io,
                  V_opt,
                  opts
               );
      break;
    }

    case 2:
    {
      jsimplex::QuadOptimOptions<2,double> opts;
      opts.max_nlopt_eval = max_nlopt_eval;
      opts.max_gn_iter    = max_gn_iter;
      opts.gn_step        = static_cast<double>(gn_step);
      opts.tol            = static_cast<double>(tol);
      opts.tol_up         = static_cast<double>(tol_up);
      opts.verbose        = (verbose != 0);

      status = jsimplex::QuadOptimizer<2,double>::optimize(
                  node_deg,
                  m_basis,
                  kappa,
                  z_io,
                  V_opt,
                  opts
               );
      break;
    }

    case 3:
    {
      jsimplex::QuadOptimOptions<3,double> opts;
      opts.max_nlopt_eval = max_nlopt_eval;
      opts.max_gn_iter    = max_gn_iter;
      opts.gn_step        = static_cast<double>(gn_step);
      opts.tol            = static_cast<double>(tol);
      opts.tol_up         = static_cast<double>(tol_up);
      opts.verbose        = (verbose != 0);

      status = jsimplex::QuadOptimizer<3,double>::optimize(
                  node_deg,
                  m_basis,
                  kappa,
                  z_io,
                  V_opt,
                  opts
               );
      break;
    }

    case 4:
    {
      jsimplex::QuadOptimOptions<4,double> opts;
      opts.max_nlopt_eval = max_nlopt_eval;
      opts.max_gn_iter    = max_gn_iter;
      opts.gn_step        = static_cast<double>(gn_step);
      opts.tol            = static_cast<double>(tol);
      opts.tol_up         = static_cast<double>(tol_up);
      opts.verbose        = (verbose != 0);

      status = jsimplex::QuadOptimizer<4,double>::optimize(
                  node_deg,
                  m_basis,
                  kappa,
                  z_io,
                  V_opt,
                  opts
               );
      break;
    }
    case 5:
    {
      jsimplex::QuadOptimOptions<5,double> opts;
      opts.max_nlopt_eval = max_nlopt_eval;
      opts.max_gn_iter    = max_gn_iter;
      opts.gn_step        = static_cast<double>(gn_step);
      opts.tol            = static_cast<double>(tol);
      opts.tol_up         = static_cast<double>(tol_up);
      opts.verbose        = (verbose != 0);

      status = jsimplex::QuadOptimizer<5,double>::optimize(
                  node_deg,
                  m_basis,
                  kappa,
                  z_io,
                  V_opt,
                  opts
               );
      break;
    }

    default:
      /* D not supported in this C API */
      status = -10;
      break;
  }

  return status;
}
