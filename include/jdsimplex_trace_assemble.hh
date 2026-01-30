#ifndef JDSIMPLEX_TRACE_ASSEMBLE_H
#define JDSIMPLEX_TRACE_ASSEMBLE_H

#include <cstddef>
#include <jdetail.hh>             // BlasGemm<Real>
#include <jdsimplex_perms.hh>     // factorial_int(D)

template<int D, class Real>
inline void jdsimplex_assemble_T_full_sigma(
  int M, int kf, int nq,
  const int* face_sigma_index,       // length D+1
  const Real* face_scale,            // length D+1
  const Real* Vt_face,               // packed: face -> (nq x kf) ColMajor, ld=nq
  const Real* wS_hat_face,           // packed: face -> (nq)
  const Real* Vv_sigma_face,         // packed: (face,sigma) -> (nq x M) ColMajor, ld=nq
  Real* T_full_out,                  // packed: face -> (kf x M) ColMajor, ld=kf
  Real* work_nqM                      // workspace (nq x M) ColMajor, ld=nq
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* Vt = Vt_face + (std::size_t)face_id * (std::size_t)(nq*kf);
    const Real* wS = wS_hat_face + (std::size_t)face_id * (std::size_t)nq;
    const Real* Vv = Vv_sigma_face
      + (std::size_t)(face_id*nsigma + sigma) * (std::size_t)(nq*M);

    const Real s = face_scale[face_id];

    // work_nqM = diag(wS*s) * Vv
    // ColMajor (nq x M): index q + a*nq
    for (int a = 0; a < M; ++a)
    {
      const Real* colV = Vv + (std::size_t)a * (std::size_t)nq;
      Real* colB       = work_nqM + (std::size_t)a * (std::size_t)nq;
      for (int q = 0; q < nq; ++q)
        colB[q] = (wS[q] * s) * colV[q];
    }

    Real* Tf = T_full_out + (std::size_t)face_id * (std::size_t)(kf*M);

    // Tf (kf x M) = Vt^T (kf x nq) * work (nq x M)
    BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt, nq,           // lda = nq
      work_nqM, nq,     // ldb = nq
      (Real)0,
      Tf, kf            // ldc = kf
    );
  }
}


#endif // JDSIMPLEX_TRACE_ASSEMBLE_H
