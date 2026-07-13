#ifndef JSIMPLEX_JKMAT_H
#define JSIMPLEX_JKMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 Build dense promotion/connection matrix K (row-major MxM) using internally
 generated κ-aware mapped tensor-product quadrature.

 Inputs:
   D          simplex dimension (supported up to the MAX_D compiled in jkmat.cpp)
   n          max total degree for basis (Pi_n)
   q          1D Gauss-Jacobi points per axis for mapped quadrature (rule has ~2q-1 exactness)
   kappa_src  length D+1 (Dirichlet parameters for source orthonormal basis)
   kappa_tgt  length D+1 (Dirichlet parameters for target orthonormal basis + quadrature weight)
   K_out      output dense matrix, size M*M, M = dim_Pi(D,n)

 Returns:
   0 on success, nonzero on error. */
int js_kmat_build_tprod(int D,
                        int n,
                        unsigned int q,
                        const double* kappa_src,
                        const double* kappa_tgt,
                        double* K_out);

/* Build sparse promotion/connection matrix K in CSC form using the pruned/stencil path.

 Output is CSC for an MxM matrix, M = dim_Pi(D,n).
 Arrays are allocated with malloc inside the library; caller must free them
 with js_kmat_csc_free.

 Returns 0 on success, nonzero on error. */
int js_kmat_build_tprod_pruned_csc(int D,
                                   int n,
                                   unsigned int q,
                                   const double* kappa_src,  // length D+1
                                   const double* kappa_tgt,  // length D+1
                                   int* nrow_out,
                                   int* ncol_out,
                                   int* nnz_out,
                                   int** colptr_out,         // length ncol+1
                                   int** rowind_out,         // length nnz
                                   double** x_out);          // length nnz

// Free CSC arrays returned by js_kmat_build_tprod_pruned_csc. Safe on NULL.
void js_kmat_csc_free(int* colptr, int* rowind, double* x);


#ifdef __cplusplus
}
#endif

#endif

