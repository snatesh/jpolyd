#ifndef JTRACE_H
#define JTRACE_H

#include <cstddef>
#include <jdetail.hh>   // BlasGemm<Real>
#include <jperms.hh>    // factorial_int

namespace jsimplex {

/*
  Assemble the full stacked trace matrix T_full for one affine D-simplex.

  Current HPS/common-face-basis convention:
    - one common canonical face basis is used for every face;
    - Vt_common is the common face basis evaluation matrix, shape nq x kf;
    - wS_hat_common is the common reference/canonical face quadrature weight vector.

  Inputs:
    M                  volume basis dimension
    kf                 face basis dimension
    nq                 number of face quadrature points
    face_sigma_index   length D+1, one orientation index per face
    face_scale         length D+1, physical/reference surface scaling per face
    Vt_common          nq x kf, column-major, ld=nq
    wS_hat_common      length nq
    Vv_sigma_face      packed (face, sigma) blocks, each nq x M column-major, ld=nq

  Output:
    T_full_out         stacked ((D+1)*kf) x M column-major, ld=(D+1)*kf.
                       Face f occupies rows [f*kf, (f+1)*kf).

  Workspace:
    work_nqM           nq x M column-major, ld=nq.
*/
template<int D, class Real>
inline void jdsimplex_assemble_T_full_common_face(
  int M, int kf, int nq,
  const int* face_sigma_index,
  const Real* face_scale,
  const Real* Vt_common,
  const Real* wS_hat_common,
  const Real* Vv_sigma_face,
  Real* T_full_out,
  Real* work_nqM
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);
  const int ldT    = nface * kf;

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* Vv = Vv_sigma_face
      + (std::size_t)(face_id * nsigma + sigma) * (std::size_t)(nq * M);

    const Real s = face_scale[face_id];

    // work_nqM = diag(wS_hat_common*s) * Vv.
    // ColMajor (nq x M): index q + a*nq.
    for (int a = 0; a < M; ++a)
    {
      const Real* colV = Vv + (std::size_t)a * (std::size_t)nq;
      Real* colB       = work_nqM + (std::size_t)a * (std::size_t)nq;
      for (int q = 0; q < nq; ++q)
        colB[q] = (wS_hat_common[q] * s) * colV[q];
    }

    // Tf is the face block inside the stacked matrix. The leading dimension is
    // the full stacked row count, not kf.
    Real* Tf = T_full_out + (std::size_t)face_id * (std::size_t)kf;

    // Tf (kf x M) = Vt_common^T (kf x nq) * work_nqM (nq x M).
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt_common, nq,
      work_nqM, nq,
      (Real)0,
      Tf, ldT
    );
  }
}

/*
  Compatibility variant for face-packed face basis/weights.

  Vt_face and wS_hat_face are packed by face:
    Vt_face      face -> nq x kf column-major
    wS_hat_face  face -> nq

  Output is still the corrected stacked ((D+1)*kf) x M column-major matrix.
*/
template<int D, class Real>
inline void jdsimplex_assemble_T_full_sigma(
  int M, int kf, int nq,
  const int* face_sigma_index,
  const Real* face_scale,
  const Real* Vt_face,
  const Real* wS_hat_face,
  const Real* Vv_sigma_face,
  Real* T_full_out,
  Real* work_nqM
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);
  const int ldT    = nface * kf;

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* Vt = Vt_face + (std::size_t)face_id * (std::size_t)(nq * kf);
    const Real* wS = wS_hat_face + (std::size_t)face_id * (std::size_t)nq;
    const Real* Vv = Vv_sigma_face
      + (std::size_t)(face_id * nsigma + sigma) * (std::size_t)(nq * M);

    const Real s = face_scale[face_id];

    for (int a = 0; a < M; ++a)
    {
      const Real* colV = Vv + (std::size_t)a * (std::size_t)nq;
      Real* colB       = work_nqM + (std::size_t)a * (std::size_t)nq;
      for (int q = 0; q < nq; ++q)
        colB[q] = (wS[q] * s) * colV[q];
    }

    Real* Tf = T_full_out + (std::size_t)face_id * (std::size_t)kf;

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt, nq,
      work_nqM, nq,
      (Real)0,
      Tf, ldT
    );
  }
}

} // namespace jsimplex

#endif // JTRACE_H
