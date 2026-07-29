#ifndef JPRECOMP_C_H
#define JPRECOMP_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque reference-simplex precompute handle.

  Error codes:
    0 success
    1 null pointer
    2 invalid dimensions/arguments
    3 unsupported D
    4 exception during construction/copy
*/

int jprecomp_create(int D,
                    int n,
                    int q_pad,
                    int q_vol,
                    int q_face,
                    const double* kappa,
                    void** handle_out);

void jprecomp_destroy(void* handle);

int jprecomp_dims(void* handle,
                  int* D_out,
                  int* n_out,
                  int* q_vol_out,
                  int* q_face_out,
                  int* M_out,
                  int* m_int_out,
                  int* kf_out,
                  int* nface_out,
                  int* nsigma_out,
                  int* nq_vol_out,
                  int* nq_face_out);

int jprecomp_get_face_ref_scale(void* handle, double* scale_out);

/* PDE residual Jacobi parameters, length D+1: kappa_res = kappa + 2. */
int jprecomp_get_kappa_res(void* handle, double* kappa_res_out);

/* Lij_ref shape: (M,M,D,D), Fortran order.
   Rows are coefficients in the kappa_res = kappa+2 residual basis. */
int jprecomp_get_Lij_ref(void* handle, double* Lij_out);

/* T_ref shape: (kf,M,nsigma,nface), Fortran order. */
int jprecomp_get_T_ref(void* handle, double* T_out);

/* Fgrad_ref shape: (kf,M,D,nsigma,nface), Fortran order. */
int jprecomp_get_Fgrad_ref(void* handle, double* Fg_out);

/* Mface_ref shape: (kf,kf,nface), Fortran order. */
int jprecomp_get_Mface_ref(void* handle, double* Mface_out);

/* Optional quadrature/basis accessors for RHS tests.
   X_vol: row-major (nq_vol,D), W_vol length nq_vol, V_vol Fortran (nq_vol,M). */
int jprecomp_get_volume_quad(void* handle, double* X_out, double* W_out);
int jprecomp_get_volume_basis(void* handle, double* V_out);

/* PDE-residual quadrature/basis accessors in the kappa_res=kappa+2 family.
   X_res: row-major (nq_vol,D), W_res length nq_vol,
   V_res: Fortran/column-major (nq_vol,M). */
int jprecomp_get_residual_quad(void* handle, double* X_out, double* W_out);
int jprecomp_get_residual_basis(void* handle, double* V_out);

/* Face quadrature/basis accessors for RHS boundary projection.
   Y_face: row-major (nq_face,D-1). For D=1 this has zero columns.
   W_face: length nq_face.
   V_face: Fortran/column-major (nq_face,kf). */
int jprecomp_get_face_quad(void* handle, double* Y_out, double* W_out);
int jprecomp_get_face_basis(void* handle, double* V_out);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPRECOMP_C_H
