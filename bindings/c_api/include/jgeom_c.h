#ifndef JGEOM_C_H
#define JGEOM_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum simplex dimension compiled into this C wrapper. */
int jgeom_c_max_D(void);

/*
  Invert a D x D dense matrix stored column-major:

    A(i,j) = A[i + D*j].

  Inputs:
    D              simplex/ambient dimension
    A              length D*D, column-major
    rel_pivot_tol  relative pivot tolerance; if <= 0, C++ default is used

  Outputs:
    Ainv           length D*D, column-major inverse
    det_out        determinant

  Returns:
    0 success
    1 null pointer
    2 unsupported or invalid D
    3 singular / numerically defective matrix
*/
int jgeom_invert_colmajor(int D,
                          const double* A,
                          double rel_pivot_tol,
                          double* Ainv,
                          double* det_out);

/*
  Build affine geometry from a column-major vertex matrix V of shape D x (D+1):

    V(coord, vertex) = V[coord + D*vertex].

  The affine map is x = v0 + B*xhat, with B(:,j) = v_{j+1} - v0.

  Outputs:
    B_out       length D*D, column-major
    BinvT_out   length D*D, column-major, equal to B^{-T}
    detB_out    determinant of B
    detBabs_out abs(detB)

  Returns:
    0 success
    1 null pointer
    2 unsupported or invalid D
    3 singular / numerically defective simplex
*/
int jgeom_affine_from_verts(int D,
                            const double* V,
                            double* B_out,
                            double* BinvT_out,
                            double* detB_out,
                            double* detBabs_out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JGEOM_C_H
