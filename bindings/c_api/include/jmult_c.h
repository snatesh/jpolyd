#ifndef JMULT_C_H
#define JMULT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for MultByQClenshaw plan */
typedef void* jmult_handle_t;

/* Opaque handle for caller-owned MultByQClenshaw workspace */
typedef void* jmult_workspace_t;

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

/* Apply y = M_q c using the plan-owned compatibility workspace.
   This preserves the original serial API. Simultaneous calls using the same
   plan handle are not thread-safe; use jmult_clenshaw_apply_workspace with
   one independent workspace per concurrent caller instead.

   q: length Mp (dim_Pi(p))
   c: length MK (dim_Pi(K))
   y_out: length MK
*/
int jmult_clenshaw_apply(jmult_handle_t handle,
                         const double* q,
                         const double* c,
                         double* y_out);

/* Create an independent mutable workspace compatible with plan.
   The plan remains the owner of the immutable Clenshaw data; the workspace
   owns only per-application scratch storage. A workspace may be reused across
   serial calls, but it must not be used by two active calls simultaneously.
*/
int jmult_clenshaw_workspace_create(jmult_handle_t plan,
                                    jmult_workspace_t* workspace_out);

/* Apply y = M_q c using an explicitly supplied workspace.
   Concurrent calls sharing one plan are thread-safe when every active call
   uses a different workspace and a different y_out buffer.
*/
int jmult_clenshaw_apply_workspace(jmult_handle_t plan,
                                   jmult_workspace_t workspace,
                                   const double* q,
                                   const double* c,
                                   double* y_out);

/* Destroy and free an independent workspace. */
void jmult_clenshaw_workspace_destroy(jmult_workspace_t workspace);

/* OpenMP regression test for shared-plan / independent-workspace execution.

   The routine deterministically generates ntrials input pairs from q and c,
   computes serial reference results with one explicit workspace, then computes
   the same trials in an OpenMP region using one workspace per worker thread.

   nthreads <= 0 requests the OpenMP maximum thread count.
   Success requires, for every trial,

     max_abs_error <= atol + rtol * ||serial_result||_inf.

   Output pointers may be null. Returns:
     0  success
     1  invalid pointer/handle
     2  invalid opaque state
     3  invalid numeric argument
     4  allocation/internal failure
     6  serial/parallel mismatch
     7  library was compiled without OpenMP support
*/
int jmult_clenshaw_test_concurrency(jmult_handle_t plan,
                                    const double* q,
                                    const double* c,
                                    int ntrials,
                                    int nthreads,
                                    double rtol,
                                    double atol,
                                    double* max_abs_error_out,
                                    double* max_rel_error_out,
                                    int* threads_used_out);

/* Destroy and free all memory owned by handle. */
void jmult_clenshaw_destroy(jmult_handle_t handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JMULT_C_H
