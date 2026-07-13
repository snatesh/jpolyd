#ifndef JQUAD_TPROD_C_H
#define JQUAD_TPROD_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Return total number of nodes for tensor grid = n^D.
   Returns 0 if D<=0 or n==0 or overflow (very large n^D). */
int jquad_mapped_npoints(int D, unsigned int n);

/* Build a mapped tensor-product quadrature rule for Lebesgue
   measure on the D-simplex.

   Simplex:
     T^D = { x in R^D : x_j >= 0, sum_j x_j <= 1 }.

   Inputs:
     D      : dimension of the simplex (1 <= D <= supported max)
     n      : number of 1D nodes per axis

   Outputs:
     X      : array of length npts * D, where npts = n^D.
              Layout is array-of-struct:
                X[p*D + j] = j-th coordinate of p-th point,
                p = 0..npts-1, j = 0..D-1
     W      : array of length npts with quadrature weights.

   The rule approximates
     ∫_{T^D} f(x) dx ≈ ∑_{p=0}^{npts-1} W[p] f(X_p),
   where sum(W) → vol(T^D) = 1 / D!.

   Return:
     npts = n^D on success, 0 on failure or unsupported D. */
int jquad_mapped_build(int D, int n, double* X, double* W);



/* Build κ-aware mapped tensor-product quadrature on the D-simplex.
   Inputs:
     kappa   : length D+1, Jacobi params (> -1/2)
     D       : dimension (1..)
     n       : nodes per axis
   Outputs:
     points  : length (n^D * D), AoS: points[p*D + j]
     weights : length (n^D), product weights (measure is already W_kappa dx)
   Returns:
     N = n^D on success, 0 on failure (bad args / alloc fail / D unsupported). */
int jquad_mapped_build_kappa(const double* kappa, int D, unsigned int n,
                             double* points, double* weights);



#ifdef __cplusplus
} // extern "C"
#endif

#endif // JQUAD_TPROD_C_H
