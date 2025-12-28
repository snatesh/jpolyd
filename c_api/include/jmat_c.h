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

/* Build ONE Jacobi coordinate multiplication matrix J_coord in CSC format.

   This builds the pruned (sparse) matrix for multiplication by x_coord in the
   same kappa space:
     (J_coord)_{row,col} = < phi_row, x_coord * phi_col >_{w_kappa}

   Inputs:
     kappa      length D+1
     D          dimension
     n          max total degree
     nquad      1D points per axis for mapped quadrature (you’ve used n+1)
     coord      which coordinate (0..D-1)

   Outputs (allocated with malloc; caller must free):
     colptr_out length N+1
     rowind_out length nnz
     x_out      length nnz
     N_out      N = dim_Pi(D,n)
     nnz_out    number of nonzeros

   Returns:
     0 on success, nonzero on error.
*/
int jmat_build_coord_pruned_csc(const double* kappa,
                                int D,
                                int n,
                                unsigned int nquad,
                                int coord,
                                int** colptr_out,
                                int** rowind_out,
                                double** x_out,
                                int* N_out,
                                int* nnz_out);


/* Free memory allocated by CSC builders (colptr/rowind/x). */
void jmat_free(void* p);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JMAT_C_H
