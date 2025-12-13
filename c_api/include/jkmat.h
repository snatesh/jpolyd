#ifndef JSIMPLEX_JKMAT_H
#define JSIMPLEX_JKMAT_H

#ifdef __cplusplus
extern "C" {
#endif

// Build dense promotion/connection matrix K (row-major MxM) using internally
// generated κ-aware mapped tensor-product quadrature.
//
// Inputs:
//   D          simplex dimension (supported up to the MAX_D compiled in jkmat.cpp)
//   n          max total degree for basis (Pi_n)
//   q          1D Gauss-Jacobi points per axis for mapped quadrature (rule has ~2q-1 exactness)
//   kappa_src  length D+1 (Dirichlet parameters for source orthonormal basis)
//   kappa_tgt  length D+1 (Dirichlet parameters for target orthonormal basis + quadrature weight)
//   K_out      output dense matrix, size M*M, M = dim_Pi(D,n)
//
// Returns:
//   0 on success, nonzero on error.
int js_kmat_build_tprod(int D,
                        int n,
                        unsigned int q,
                        const double* kappa_src,
                        const double* kappa_tgt,
                        double* K_out);

#ifdef __cplusplus
}
#endif

#endif

