#ifndef JFLUX_C_H
#define JFLUX_C_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  Assemble F_full using one common canonical face basis for all faces.

  Matrix/array layout conventions:
    normal_scaled       shape (D+1,D), row-major; normal_scaled[f,:] = s_f n_f
    BinvT               shape (D,D), column-major; grad_x = BinvT grad_hat
    Vt_common           shape (nq,kf), column-major
    dVv_hat_sigma_face  shape (nq,M,D,nsigma,D+1), Fortran order
    F_full_out          shape ((D+1)*kf,M), column-major

  Return codes:
    0 success
    1 null pointer
    2 invalid dimensions
    3 unsupported D
    4 invalid sigma index
*/
int jflux_assemble_F_full_common(int D,
                                  int M,
                                  int kf,
                                  int nq,
                                  const int* face_sigma_index,
                                  const double* normal_scaled,
                                  const double* BinvT,
                                  const double* Vt_common,
                                  const double* wS_hat_common,
                                  const double* dVv_hat_sigma_face,
                                  double* F_full_out);

/*
  Compatibility variant with face-packed face basis/weights.

  Vt_face      packed face -> shape (nq,kf), column-major
  wS_hat_face  packed face -> length nq
  Output remains stacked shape ((D+1)*kf,M), column-major.
*/
int jflux_assemble_F_full_facepacked(int D,
                                      int M,
                                      int kf,
                                      int nq,
                                      const int* face_sigma_index,
                                      const double* normal_scaled,
                                      const double* BinvT,
                                      const double* Vt_face,
                                      const double* wS_hat_face,
                                      const double* dVv_hat_sigma_face,
                                      double* F_full_out);

#ifdef __cplusplus
}
#endif

#endif // JFLUX_C_H
