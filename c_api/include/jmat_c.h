#ifndef JMAT_C_H
#define JMAT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Return dim_Pi(n) for given dimension D (double precision).
   Returns 0 on error or unsupported D. */
int jmat_dim_Pi(int D, int n);

/* Build Jacobi matrices for given dimension D, maximum degree n,
   and Jacobi parameters kappa[0..D].

   On success:
     - returns N = dim_Pi(n) (size of each matrix block),
     - writes D*N*N doubles into J_all in row-major order:

         J_all[ i*N*N + row*N + col ] = (J_i)_{row,col}

       for i = 0..D-1, row,col = 0..N-1.

   Returns 0 on failure (bad D, n < 0, null pointers, etc.). */
int jmat_build(const double* kappa, int D, int n, double* J_all);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JMAT_C_H
