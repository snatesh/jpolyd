#ifndef JFLUX_H
#define JFLUX_H

#include <cstddef>
#include <jdetail.hh>   // BlasGemm<Real>
#include <jperms.hh>    // factorial_int

namespace jsimplex {

/*
  Assemble the full stacked physical normal-flux matrix F_full for one affine
  D-simplex.

  Convention:
    F_full maps volume coefficients c to face flux moments

      mu_{f,j} = ∫_{S_f} psi_j * (n_f · grad_x u_h) * w_face dS.

  This implementation uses a common canonical face basis and expects the caller
  to supply, for each physical face, the scaled outward normal

      normal_scaled[f,:] = face_scale[f] * n_out[f]

  where face_scale[f] is the surface Jacobian from the canonical reference
  (D-1)-simplex parameter domain to the physical face. Equivalently,

      (n_out · grad_x phi) dS_phys
        = (normal_scaled · grad_x phi) dY.

  Inputs:
    M                    volume basis dimension
    kf                   face basis dimension
    nq                   number of face quadrature points
    face_sigma_index     length D+1, one orientation index per face
    normal_scaled        packed row-major shape (D+1,D): face f then coord i
    BinvT                D x D column-major, B^{-T}; grad_x = BinvT grad_hat
    Vt_common            nq x kf column-major, ld=nq
    wS_hat_common        length nq, canonical face quadrature weights
    dVv_hat_sigma_face   packed shape (nq,M,D,nsigma,D+1), Fortran order.
                         Block (sigma,face) is nq x M x D, with
                         dV[q + nq*(a + M*j)] = ∂_{hat x_j} phi_a at q.

  Output:
    F_full_out           stacked ((D+1)*kf) x M column-major, ld=(D+1)*kf.
                         Face f occupies rows [f*kf, (f+1)*kf).

  Workspace:
    work_nqM             nq x M column-major, ld=nq.
*/
template<int D, class Real>
inline void jdsimplex_assemble_F_full_common_face(
  int M, int kf, int nq,
  const int* face_sigma_index,
  const Real* normal_scaled,
  const Real* BinvT,
  const Real* Vt_common,
  const Real* wS_hat_common,
  const Real* dVv_hat_sigma_face,
  Real* F_full_out,
  Real* work_nqM
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);
  const int ldF    = nface * kf;

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* dV = dVv_hat_sigma_face
      + (std::size_t)(face_id * nsigma + sigma)
      * (std::size_t)(nq * M * D);

    // beta[j] = sum_i normal_scaled_i * BinvT(i,j).
    // Then normal_scaled · grad_x(phi)
    //   = normal_scaled^T BinvT grad_hat(phi)
    //   = beta^T grad_hat(phi).
    Real beta[D];
    for (int j = 0; j < D; ++j)
    {
      Real bj = (Real)0;
      for (int i = 0; i < D; ++i)
      {
        const Real ns_i = normal_scaled[(std::size_t)face_id * (std::size_t)D
                                        + (std::size_t)i];
        bj += ns_i * BinvT[(std::size_t)i + (std::size_t)D * (std::size_t)j];
      }
      beta[j] = bj;
    }

    // work_nqM = diag(wS_hat_common) * (normal_scaled · grad_x Vv).
    // ColMajor (nq x M): index q + a*nq.
    for (int a = 0; a < M; ++a)
    {
      Real* colB = work_nqM + (std::size_t)a * (std::size_t)nq;
      for (int q = 0; q < nq; ++q)
      {
        Real ndot = (Real)0;
        for (int j = 0; j < D; ++j)
        {
          const std::size_t idx = (std::size_t)q
            + (std::size_t)nq * ((std::size_t)a + (std::size_t)M * (std::size_t)j);
          ndot += beta[j] * dV[idx];
        }
        colB[q] = wS_hat_common[q] * ndot;
      }
    }

    Real* Ff = F_full_out + (std::size_t)face_id * (std::size_t)kf;

    // Ff (kf x M) = Vt_common^T (kf x nq) * work_nqM (nq x M).
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt_common, nq,
      work_nqM, nq,
      (Real)0,
      Ff, ldF
    );
  }
}

/*
  Compatibility variant for face-packed face basis/weights.

  Vt_face and wS_hat_face are packed by face:
    Vt_face      face -> nq x kf column-major
    wS_hat_face  face -> nq

  Output remains stacked ((D+1)*kf) x M column-major.
*/
template<int D, class Real>
inline void jdsimplex_assemble_F_full_sigma(
  int M, int kf, int nq,
  const int* face_sigma_index,
  const Real* normal_scaled,
  const Real* BinvT,
  const Real* Vt_face,
  const Real* wS_hat_face,
  const Real* dVv_hat_sigma_face,
  Real* F_full_out,
  Real* work_nqM
)
{
  const int nface  = D + 1;
  const int nsigma = factorial_int(D);
  const int ldF    = nface * kf;

  for (int face_id = 0; face_id < nface; ++face_id)
  {
    const int sigma = face_sigma_index[face_id];

    const Real* Vt = Vt_face + (std::size_t)face_id * (std::size_t)(nq * kf);
    const Real* wS = wS_hat_face + (std::size_t)face_id * (std::size_t)nq;
    const Real* dV = dVv_hat_sigma_face
      + (std::size_t)(face_id * nsigma + sigma)
      * (std::size_t)(nq * M * D);

    Real beta[D];
    for (int j = 0; j < D; ++j)
    {
      Real bj = (Real)0;
      for (int i = 0; i < D; ++i)
      {
        const Real ns_i = normal_scaled[(std::size_t)face_id * (std::size_t)D
                                        + (std::size_t)i];
        bj += ns_i * BinvT[(std::size_t)i + (std::size_t)D * (std::size_t)j];
      }
      beta[j] = bj;
    }

    for (int a = 0; a < M; ++a)
    {
      Real* colB = work_nqM + (std::size_t)a * (std::size_t)nq;
      for (int q = 0; q < nq; ++q)
      {
        Real ndot = (Real)0;
        for (int j = 0; j < D; ++j)
        {
          const std::size_t idx = (std::size_t)q
            + (std::size_t)nq * ((std::size_t)a + (std::size_t)M * (std::size_t)j);
          ndot += beta[j] * dV[idx];
        }
        colB[q] = wS[q] * ndot;
      }
    }

    Real* Ff = F_full_out + (std::size_t)face_id * (std::size_t)kf;

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasTrans, CblasNoTrans,
      kf, M, nq,
      (Real)1,
      Vt, nq,
      work_nqM, nq,
      (Real)0,
      Ff, ldF
    );
  }
}

} // namespace jsimplex

#endif // JFLUX_H
