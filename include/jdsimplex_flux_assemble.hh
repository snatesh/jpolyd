#ifndef JDSIMPLEX_FLUX_ASSEMBLE_H
#define JDSIMPLEX_FLUX_ASSEMBLE_H

#include <cstddef>

#include <jdetail.hh>
#include <jdsimplex_geom.hh>
#include <jdsimplex_perms.hh>

template<int D, class Real>
inline void jdsimplex_assemble_F_full_sigma_poisson(
  int M, int kf, int nq,
  const JDSimplexGeom<D,Real>& geom,  // provides JinvT (DxD col-major) and detJabs if you need it
  const int* face_sigma_index,        // length D+1
  const Real* face_scale,             // length D+1
  const Real* face_n_unit,            // (D+1)*D
  const Real* Vt_face,                // face -> (nq x kf) ColMajor, ld=nq
  const Real* wS_hat_face,            // face -> (nq)
  const Real* dVv_hat_sigma_face,     // (face,sigma) -> (nq x M x D), stored as ((q) + a*nq) per component? see note below
  Real* F_full_out,                   // face -> (kf x M) ColMajor, ld=kf
  Real* work_nqM                       // (nq x M) ColMajor, ld=nq
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* n = face_n_unit + (std::size_t)face_id * (std::size_t)D;
    const Real* Vt = Vt_face + (std::size_t)face_id * (std::size_t)(nq*kf);
    const Real* wS = wS_hat_face + (std::size_t)face_id * (std::size_t)nq;

    const Real* dVhat = dVv_hat_sigma_face
      + (std::size_t)(face_id*nsigma + sigma) * (std::size_t)(nq*M*D);

    const Real s = face_scale[face_id];

    // work_nqM(q,a) = (wS_hat[q]*s) * (n · (JinvT * grad_hat(q,a)))
    // We assume grad_hat components are packed with q fastest, then a, then component j:
    //   grad_hat(j) for basis a is at dVhat[ (j*nq*M) + (a*nq) + q ]
    // This matches "ColMajor per (nq x M)" slice for each component.
    for (int a = 0; a < M; ++a)
    {
      Real* colB = work_nqM + (std::size_t)a * (std::size_t)nq;

      for (int q = 0; q < nq; ++q)
      {
        // grad_x = JinvT * grad_hat
        Real dot = (Real)0;
        for (int i = 0; i < D; ++i)
        {
          Real gx_i = (Real)0;
          for (int j = 0; j < D; ++j)
          {
            const Real grad_hat_j = dVhat[(std::size_t)j*(std::size_t)(nq*M) + (std::size_t)a*(std::size_t)nq + (std::size_t)q];
            gx_i += geom.JinvT[i + j*D] * grad_hat_j;
          }
          dot += n[i] * gx_i;
        }
        colB[q] = (wS[q] * s) * dot;
      }
    }

    Real* Ff = F_full_out + (std::size_t)face_id * (std::size_t)(kf*M);

    BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt, nq,
      work_nqM, nq,
      (Real)0,
      Ff, kf
    );
  }
}


#endif //JDSIMPLEX_FLUX_ASSEMBLE_H
