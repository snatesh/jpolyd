#ifndef JBASIS_C_H
#define JBASIS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate all orthonormal Jacobi basis functions of total degree
   <= n on the D-simplex at npts points.

   Inputs:
     X:          point coordinates, X[p*ld_point + j*ld_dim],
                 p = 0..npts-1, j = 0..D-1
     ld_point,
     ld_dim:     leading dimensions for X
     npts:       number of points
     kappa:      parameters array of length D+1
     D:          dimension of simplex (1..4, as compiled)
     n:          maximum total degree
     alpha_table: M x D multi-index table (row-major)
     tail_deg:    M x D tail-degree table (row-major)
     inv_h:       length M array of 1 / h_alpha
     ld_V:        leading dimension for V in point index

   Output:
     V:          values V[p + m*ld_V] for m=0..M-1, p=0..npts-1 */
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
                     double* dV);

/* Return M = dim Pi_n^D, the number of basis functions of total
   degree <= n in D variables.

   This is a thin C wrapper around Basis<D,double>::dim_Pi(n). */
int jbasis_dim_Pi(int D, int n);

/* Build basis structure for total degree <= n in dimension D:

   Inputs:
     kappa:       parameters of length D+1
     D:           dimension of simplex
     n:           maximum total degree

   Outputs (caller allocated):
     alpha_table: M x D multi-index table (row-major),
                  where M = jbasis_dim_Pi(D,n)
     tail_deg:    M x D tail-degree table (row-major)
     inv_h:       length M array of 1 / h_alpha, in the same
                  graded lex order as alpha_table.

   Arrays must be preallocated by the caller. */
void jbasis_build_structures(const double* kappa,
                             int D,
                             int n,
                             int* alpha_table,
                             int* tail_deg,
                             double* inv_h);



#ifdef __cplusplus
} // extern "C"
#endif

#endif // JBASIS_C_H
