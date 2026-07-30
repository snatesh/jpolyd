#ifndef JELLIPTIC_C_H
#define JELLIPTIC_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* jelliptic_plan_t;
typedef void* jelliptic_workspace_t;

/*
  Create a reusable affine-element elliptic assembly plan.

  kappa_res has length D+1 and is the common PDE-residual Jacobi family.
  p2, p1, p0 are the common coefficient degrees for the second-, first-,
  and zero-order terms.  Use -1 to disable an order.
*/
int jelliptic_plan_create(int D,
                          int n,
                          const double* kappa_res,
                          int p2,
                          int p1,
                          int p0,
                          int assume_symmetric,
                          jelliptic_plan_t* plan_out);

void jelliptic_plan_destroy(jelliptic_plan_t plan);

/* Return dimensions and coefficient-vector lengths. */
int jelliptic_plan_dims(jelliptic_plan_t plan,
                        int* D_out,
                        int* n_out,
                        int* M_out,
                        int* m2_out,
                        int* m1_out,
                        int* Mp2_out,
                        int* Mp1_out,
                        int* Mp0_out);

/* One independent mutable workspace per concurrent caller. */
int jelliptic_workspace_create(jelliptic_plan_t plan,
                               jelliptic_workspace_t* workspace_out);

void jelliptic_workspace_destroy(jelliptic_workspace_t workspace);

/*
  Assemble L_int, shape (m2,M), column-major.

  BinvT:   B^{-T}, shape (D,D), column-major.
  Lij_ref: shape (M,M,D,D), Fortran order.
  Li_ref:  shape (M,M,D), Fortran order.
  L0_ref:  shape (M,M), column-major.

  Coefficient storage is physical-component-major with modal index fastest:
    A[(r*D+s)*Mp2 + alpha]
    b[r*Mp1 + alpha]
    c[alpha].

  Pointers for disabled orders may be null.
*/
int jelliptic_assemble_L_int(jelliptic_plan_t plan,
                             jelliptic_workspace_t workspace,
                             const double* BinvT,
                             double detBabs,
                             const double* Lij_ref,
                             const double* Li_ref,
                             const double* L0_ref,
                             const double* A,
                             const double* b,
                             const double* c,
                             double* L_int_out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JELLIPTIC_C_H
