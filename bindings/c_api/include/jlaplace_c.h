#ifndef JLAPLACE_C_H
#define JLAPLACE_C_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  Assemble L_int = |detB| * (sum_ij G_ij Lij_ref[:,:,i,j])[:m_int,:].

  Layout conventions:
    G             shape (D,D), column-major
    Lij_ref       shape (M,M,D,D), Fortran order
    L_int_out     shape (m_int,M), column-major

  Return codes:
    0 success
    1 null pointer
    2 invalid dimensions
    3 unsupported D
*/
int jlaplace_assemble_L_int(int D,
                             int M,
                             int m_int,
                             const double* G,
                             double detBabs,
                             const double* Lij_ref,
                             double* L_int_out);

#ifdef __cplusplus
}
#endif

#endif // JLAPLACE_C_H
