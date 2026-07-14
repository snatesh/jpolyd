#ifndef JTRACE_C_H
#define JTRACE_C_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  Assemble T_full using one common canonical face basis for all faces.

  Matrix layout conventions:
    Vt_common      shape (nq, kf), column-major
    Vv_sigma_face  packed as blocks (face, sigma), each block shape (nq, M), column-major
    T_full_out     shape ((D+1)*kf, M), column-major

  Return codes:
    0 success
    1 null pointer
    2 invalid dimensions
    3 unsupported D
    4 invalid sigma index
*/
int jtrace_assemble_T_full_common(int D,
                                  int M,
                                  int kf,
                                  int nq,
                                  const int* face_sigma_index,
                                  const double* face_scale,
                                  const double* Vt_common,
                                  const double* wS_hat_common,
                                  const double* Vv_sigma_face,
                                  double* T_full_out);

/*
  Compatibility variant with face-packed face basis/weights.

  Vt_face      packed face -> shape (nq, kf), column-major
  wS_hat_face  packed face -> length nq
  Output remains stacked shape ((D+1)*kf, M), column-major.
*/
int jtrace_assemble_T_full_facepacked(int D,
                                      int M,
                                      int kf,
                                      int nq,
                                      const int* face_sigma_index,
                                      const double* face_scale,
                                      const double* Vt_face,
                                      const double* wS_hat_face,
                                      const double* Vv_sigma_face,
                                      double* T_full_out);

#ifdef __cplusplus
}
#endif

#endif // JTRACE_C_H
