#ifndef JEVD_C_H
#define JEVD_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * C wrapper around the jevd_serial::joint_evd_symmetric routine.
 *
 * Layout of J:
 *   - m: size of each matrix
 *   - n: number of matrices
 *   - J: pointer to m*m*n doubles stored in the same layout as the
 *        original serial code:
 *
 *          J[i + m * ( j + nn * m )]
 *
 *     i = 0..m-1 (row), j = 0..m-1 (column), nn = 0..n-1 (matrix index)
 *     i.e. each matrix is column-major, and matrices are stacked.
 *
 * V:
 *   - optional joint diagonalizer, column-major m x m.
 *   - Only used if accumulate_V != 0.
 *
 * Arguments:
 *   J            [in/out]  m*m*n array as above.
 *   m            [in]      matrix size
 *   n            [in]      number of matrices
 *   tol          [in]      threshold on |s| (like "thresh" in serial code)
 *   max_sweeps   [in]      maximum sweeps over (p,q)
 *   accumulate_V [in]      nonzero => update V, zero => ignore V
 *   V            [in/out]  m*m column-major, may be NULL if accumulate_V == 0
 *   sweeps_out   [out]     number of sweeps actually performed
 *   max_offdiag_out [out]  max |s| in last sweep
 *
 * Returns:
 *   0  on convergence,
 *   1  if not converged in max_sweeps,
 *  <0  on invalid input.
 */
int jjevd_serial(double* J,
                 int m,
                 int n,
                 double tol,
                 int max_sweeps,
                 int accumulate_V,
                 double* V,
                 int* sweeps_out,
                 double* max_offdiag_out);

#ifdef __cplusplus
}
#endif

#endif /* JEVD_C_H */
