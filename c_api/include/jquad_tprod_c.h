#ifndef JQUAD_TPROD_C_H
#define JQUAD_TPROD_C_H

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JQUAD_TPROD_C_H
