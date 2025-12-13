#ifndef JSIMPLEX_JDMAT_H
#define JSIMPLEX_JDMAT_H

#ifdef __cplusplus
extern "C" {
#endif

// Build dense derivative projection matrix using internally generated κ-aware mapped
// tensor-product quadrature (q points per axis) for the range weight kappa_rng.
//
//   D_out[i,j] = < phi_i(kappa_rng), d/dx_axis phi_j(kappa_src) >_{L2(w_kappa_rng)}
//
// Output is row-major MxM, where M = dim_Pi(D,n).
//
// Returns 0 on success, nonzero on error.
int js_dmat_build_tprod(int D,
                        int n,
                        unsigned int q,
                        const double* kappa_src,  // length D+1
                        const double* kappa_rng,  // length D+1
                        int axis,                 // 0..D-1
                        double* D_out);

// Build the "natural" differentiation matrix for axis using the rule:
//   kappa_rng = kappa_src + e_axis + e_last
// and apply row-relative pruning with factor = 100*eps (double).
//
// Output is row-major MxM, where M = dim_Pi(D,n).
//
// Returns 0 on success, nonzero on error.
int js_dmat_build_tprod_natural_pruned(int D,
                                       int n,
                                       unsigned int q,
                                       const double* kappa_src,  // length D+1
                                       int axis,                 // 0..D-1
                                       double* D_out);


#ifdef __cplusplus
}
#endif

#endif

