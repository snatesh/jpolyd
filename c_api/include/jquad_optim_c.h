#ifndef JQUAD_OPTIM_C_H
#define JQUAD_OPTIM_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* C API for Jacobi simplex quadrature optimization.
 *
 * Parameters:
 *   D             : spatial dimension (1, 2, 3, ...)
 *   node_deg      : node polynomial degree n
 *   m_basis       : exactness degree m (ignored for D=1)
 *   kappa         : array of length D+1 (Jacobi parameters)
 *   z_io          : in/out array of length (D+1)*N, where
 *                   N = dim Pi_n^D = C(n + D, D).
 *                   Layout: [x_0,...,x_{N-1}, w_0,...,w_{N-1}],
 *                   where each x_p has D components in simplex barycentric coords.
 *                   Must be non-null.
 *   V_opt         : optional, if non-null, on return contains V_opt[p + m*N]
 *                   = P_m(x_p_opt) with dimension N*M, column-major in polynomial index.
 *   max_nlopt_eval: max evaluations for NLopt (ignored for D=1)
 *   max_gn_iter   : max Gauss–Newton iterations (ignored for D=1)
 *   gn_step       : Gauss–Newton step length alpha (ignored for D=1)
 *   tol           : residual tolerance for GN and NLopt ftol (ignored for D=1)
 *   tol_up        : upper bound on residual (ignored for D=1)
 *   verbose       : nonzero for verbose output
 *
 * Returns:
 *   0  on (nominal) success
 *  <0  on error (invalid D, allocation failure, NLopt error, etc.)
 */
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
                   int verbose);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JQUAD_OPTIM_C_H */
