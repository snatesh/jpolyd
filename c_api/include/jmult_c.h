#ifndef JMULT_C_H
#define JMULT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for MultByQClenshaw plan */
typedef void* jmult_handle_t;

/* Create a Clenshaw plan.
   Inputs:
     kappa: length D+1
     D: simplex dimension (1..5)
     p: degree of q expansion (Pi_p)
     K: degree of c expansion / output space (Pi_K)
     alpha_p: alpha-table for Pi_p in row-major (Mp x D), same ordering as your basis
     Mp: number of rows in alpha_p = dim_Pi(p)
     assume_symmetric: if nonzero, uses symmetric shortcut for C blocks (matches python default)

   Output:
     *handle_out is a valid handle, must be destroyed by jmult_clenshaw_destroy.

   Returns 0 on success, nonzero error code on failure.
*/
int jmult_clenshaw_create(const double* kappa,
                          int D,
                          int p,
                          int K,
                          const int* alpha_p,
                          int Mp,
                          int assume_symmetric,
                          jmult_handle_t* handle_out);

/* Apply y = M_q c (de-aliased / lifted to Pi_K according to your algorithm).
   q: length Mp (dim_Pi(p))
   c: length MK (dim_Pi(K))
   y_out: length MK
*/
int jmult_clenshaw_apply(jmult_handle_t handle,
                         const double* q,
                         const double* c,
                         double* y_out);

/* Destroy and free all memory owned by handle. */
void jmult_clenshaw_destroy(jmult_handle_t handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JMULT_C_H




