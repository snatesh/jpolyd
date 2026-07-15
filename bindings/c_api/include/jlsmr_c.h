#ifndef JLSMR_C_H
#define JLSMR_C_H

#ifdef __cplusplus
extern "C" {
#endif

int lsmr_dense_solve_colmajor(
  int m,
  int n,
  const double* A_colmajor,
  const double* b,
  double* x,
  double damp,
  double atol,
  double btol,
  double conlim,
  int itnlim,
  int nout,
  int localsize,
  int ctest,
  int* istop_out,
  int* itn_out,
  int* stat_out,
  double* normr_out,
  double* normA_out,
  double* condA_out,
  double* normb_out,
  double* normx_out,
  double* normAr_out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
