#ifndef JPRECOMP_HH
#define JPRECOMP_HH

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <jbasis.hh>
#include <jquad_tprod.hh>
#include <jkmat.hh>
#include <jdmat.hh>
#include <jgeom.hh>
#include <jperms.hh>
#include <jdetail.hh>

namespace jsimplex {

template<int D, class Real>
class RefSimplexPrecomp
{
public:
  static_assert(D >= 1, "RefSimplexPrecomp requires D>=1");

  int n = 0;
  int q_pad = 2;
  int q_vol = 0;
  int q_face = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int nface = D + 1;
  int nsigma = factorial_int(D);
  int nq_vol = 0;
  int nq_face = 0;

  std::array<Real, D + 1> kappa{};
  std::array<Real, D + 1> kappa_res{};
  std::array<Real, D> kappa_face{};

  // Volume quadrature/basis in source-kappa convention.
  std::vector<Real> X_vol;     // row-major nq_vol x D (for basis evaluator)
  std::vector<Real> W_vol;     // nq_vol
  std::vector<Real> V_vol;     // col-major nq_vol x M
  std::vector<Real> V_vol_int; // col-major nq_vol x m_int

  // Volume quadrature/basis in PDE-residual kappa+2 convention.
  // The full degree-n basis is stored so the same data can test/build
  // second-, first-, and zero-order residual operators.
  std::vector<Real> X_res;      // row-major nq_vol x D
  std::vector<Real> W_res;      // nq_vol
  std::vector<Real> V_res;      // col-major nq_vol x M

  // Canonical face quadrature/basis. For D=1, Y_face is empty, W_face=[1], V_face=[1].
  std::vector<Real> Y_face;    // row-major nq_face x (D-1) (for basis evaluator)
  std::vector<Real> W_face;    // nq_face, intrinsic canonical face weights
  std::vector<Real> V_face;    // col-major nq_face x kf

  // Embedded reference-face measure scale from canonical (D-1)-simplex to the
  // corresponding face of the reference D-simplex.
  std::vector<Real> face_ref_scale; // nface

  // Dense reference operator blocks, all Fortran/column-major in public layout.
  // Lij_ref:    M  x M x D x D
  // Li_ref:     M  x M x D
  // L0_ref:     M  x M
  // T_ref:      kf x M x nsigma x nface
  // Fgrad_ref:  kf x M x D x nsigma x nface
  // Mface_ref:  kf x kf x nface
  std::vector<Real> Lij_ref;
  std::vector<Real> Li_ref;
  std::vector<Real> L0_ref;
  std::vector<Real> T_ref;
  std::vector<Real> Fgrad_ref;
  std::vector<Real> Mface_ref;

  RefSimplexPrecomp(int n_in,
                    int q_pad_in,
                    int q_vol_in,
                    int q_face_in,
                    const Real* kappa_in)
  {
    if (!kappa_in) { throw std::invalid_argument("RefSimplexPrecomp: null kappa"); }
    if (n_in < 2) { throw std::invalid_argument("RefSimplexPrecomp: require n>=2"); }

    n = n_in;
    q_pad = (q_pad_in > 0) ? q_pad_in : 2;
    q_vol = (q_vol_in > 0) ? q_vol_in : (n + q_pad);
    q_face = (q_face_in > 0) ? q_face_in : q_vol;
    if (q_face > q_vol) { throw std::invalid_argument("RefSimplexPrecomp: require q_face <= q_vol"); }

    for (int i = 0; i <= D; ++i)
    {
      kappa[(std::size_t)i] = kappa_in[i];
      kappa_res[(std::size_t)i] = kappa_in[i] + Real(2);
    }

    // Current common-face convention: all faces use source kappa[0:D].
    dsimplex_common_face_kappa<D,Real>(kappa.data(), kappa_face.data());

    M = Basis<D,Real>::dim_Pi(n);
    m_int = Basis<D,Real>::dim_Pi(n - 2);
    if constexpr (D == 1)
    {
      kf = 1;
      nq_face = 1;
    }
    else
    {
      kf = Basis<D - 1,Real>::dim_Pi(n);
      nq_face = (int)QuadMapped<D - 1,Real>::npoints((unsigned int)q_face);
    }
    nq_vol = (int)QuadMapped<D,Real>::npoints((unsigned int)q_vol);

    build_volume_data();
    build_residual_data();
    build_face_data();
    build_second_partials();
    build_first_partials();
    build_zero_partials();
    build_face_operator_data();
  }

private:
  void build_volume_data()
  {
    X_vol.assign((std::size_t)nq_vol * D, (Real)0);
    W_vol.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa.data(), X_vol.data(), W_vol.data());
    if (built != nq_vol) { throw std::runtime_error("RefSimplexPrecomp: volume quadrature build failed"); }

    std::vector<int> alpha, tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha, tail, invh);

    V_vol.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n,
      alpha.data(), tail.data(), invh.data(), V_vol.data(), nq_vol, nullptr);

    std::vector<int> alpha_i, tail_i;
    std::vector<Real> invh_i;
    Basis<D,Real>::build_structures(n - 2, kappa.data(), alpha_i, tail_i, invh_i);

    V_vol_int.assign((std::size_t)nq_vol * m_int, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n - 2,
      alpha_i.data(), tail_i.data(), invh_i.data(), V_vol_int.data(), nq_vol, nullptr);
  }

  void build_residual_data()
  {
    X_res.assign((std::size_t)nq_vol * D, (Real)0);
    W_res.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa_res.data(), X_res.data(), W_res.data());
    if (built != nq_vol)
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: residual quadrature build failed");
    }

    std::vector<int> alpha_res, tail_res;
    std::vector<Real> invh_res;
    Basis<D,Real>::build_structures(
      n, kappa_res.data(), alpha_res, tail_res, invh_res);

    V_res.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_res.data(), D, 1, nq_vol, kappa_res.data(), n,
      alpha_res.data(), tail_res.data(), invh_res.data(),
      V_res.data(), nq_vol, nullptr);
  }

  void build_face_data()
  {
    if constexpr (D == 1)
    {
      Y_face.clear();
      W_face.assign(1, (Real)1);
      V_face.assign(1, (Real)1);
    }
    else
    {
      Y_face.assign((std::size_t)nq_face * (D - 1), (Real)0);
      W_face.assign((std::size_t)nq_face, (Real)0);
      const int built = QuadMapped<D - 1,Real>::build_kappa(
        (unsigned int)q_face, kappa_face.data(), Y_face.data(), W_face.data());
      if (built != nq_face) { throw std::runtime_error("RefSimplexPrecomp: face quadrature build failed"); }

      std::vector<int> alpha_f, tail_f;
      std::vector<Real> invh_f;
      Basis<D - 1,Real>::build_structures(n, kappa_face.data(), alpha_f, tail_f, invh_f);

      V_face.assign((std::size_t)nq_face * kf, (Real)0);
      Basis<D - 1,Real>::eval_all(
        Y_face.data(), D - 1, 1, nq_face, kappa_face.data(), n,
        alpha_f.data(), tail_f.data(), invh_f.data(), V_face.data(), nq_face, nullptr);
    }

    face_ref_scale.assign((std::size_t)nface, (Real)0);
    for (int face_id = 0; face_id < nface; ++face_id)
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);
      face_ref_scale[(std::size_t)face_id] =
        dsimplex_reference_face_scale_from_vertex_ids<D,Real>(fv);
    }
  }

  void build_second_partials()
  {
    Lij_ref.assign((std::size_t)M * M * D * D, (Real)0);

    std::vector<Real> D1((std::size_t)D * M * M, (Real)0); // row-major per axis
    std::vector<std::array<Real, D + 1>> k1((std::size_t)D);

    for (int i = 0; i < D; ++i)
    {
      Basis<D,Real>::derivative_kappa_shift(kappa.data(), i, k1[(std::size_t)i].data());
      DMat<D,Real>::build_tprod_natural_pruned_dense(
        n, (unsigned int)q_vol, kappa.data(), i,
        D1.data() + (std::size_t)i * M * M);
    }

    for (int i = 0; i < D; ++i)
    {
      for (int j = 0; j < D; ++j)
      {
        std::array<Real, D + 1> k2{};
        Basis<D,Real>::derivative_kappa_shift(k1[(std::size_t)i].data(), j, k2.data());

        std::vector<Real> Dj((std::size_t)M * M, (Real)0);
        DMat<D,Real>::build_tprod_natural_pruned_dense(
          n, (unsigned int)q_vol, k1[(std::size_t)i].data(), j, Dj.data());

        std::vector<Real> Draw_col((std::size_t)M * M, Real(0));
        
        const Real* Di_rm =
          D1.data() + (std::size_t)i * M * M;
        
        // Draw_col = Dj * Di
        // Dj and Di are row-major actual matrices.
        // Interpreted as column-major, they represent Dj^T and Di^T.
        // Therefore trans/trans gives the actual Dj and Di.
        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasTrans,
          CblasTrans,
          M,
          M,
          M,
          Real(1),
          Dj.data(),
          M,
          Di_rm,
          M,
          Real(0),
          Draw_col.data(),
          M
        );
        
        // Output block is column-major M x M inside Lij_ref.
        Real* Lij_block =
          &Lij_ref[(std::size_t)M * M * ((std::size_t)i + (std::size_t)D * j)];
        
        std::vector<Real> K((std::size_t)M * M, (Real)0);
        KMat<D,Real>::build_tprod_pruned_dense(
          n, (unsigned int)q_vol, k2.data(), kappa_res.data(), K.data());

  
        // Lij_block = K * Draw_col
        //
        // K is row-major actual matrix, so use transA.
        // Draw_col is already column-major actual matrix, so no transB.
        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasTrans,
          CblasNoTrans,
          M,
          M,
          M,
          Real(1),
          K.data(),
          M,
          Draw_col.data(),
          M,
          Real(0),
          Lij_block,
          M
        );
      }
    }
  }

  void build_first_partials()
  {
    Li_ref.assign((std::size_t)M * M * D, (Real)0);

    for (int i = 0; i < D; ++i)
    {
      std::array<Real, D + 1> k1{};
      Basis<D,Real>::derivative_kappa_shift(
        kappa.data(), i, k1.data());

      // Natural first derivative:
      //   Pi_n(kappa) -> Pi_{n-1}(k1).
      // DMat returns the actual matrix in row-major storage, padded to M x M.
      std::vector<Real> Di((std::size_t)M * M, (Real)0);
      DMat<D,Real>::build_tprod_natural_pruned_dense(
        n, (unsigned int)q_vol, kappa.data(), i, Di.data());

      // Sparse promotion into the common PDE residual family:
      //   Pi_n(k1) -> Pi_n(kappa_res),  kappa_res = kappa + 2.
      // Only the degree <= n-1 range of Di contributes.
      // KMat returns the actual matrix in row-major storage.
      std::vector<Real> K((std::size_t)M * M, (Real)0);
      KMat<D,Real>::build_tprod_pruned_dense(
        n, (unsigned int)q_vol, k1.data(), kappa_res.data(), K.data());

      // Public output block is column-major M x M.
      Real* Li_block =
        Li_ref.data() + (std::size_t)M * M * i;

      // Li_block = K * Di.
      // Row-major actual matrices appear transposed when interpreted as
      // column-major, hence trans/trans recovers the actual operands.
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasTrans,
        CblasTrans,
        M,
        M,
        M,
        Real(1),
        K.data(),
        M,
        Di.data(),
        M,
        Real(0),
        Li_block,
        M
      );
    }
  }

  void build_zero_partials()
  {
    L0_ref.assign((std::size_t)M * M, (Real)0);

    // Zero derivatives followed by sparse promotion into the common PDE
    // residual family:
    //   Pi_n(kappa) -> Pi_n(kappa_res),  kappa_res = kappa + 2.
    // KMat returns the actual matrix in row-major storage.
    std::vector<Real> K((std::size_t)M * M, (Real)0);
    KMat<D,Real>::build_tprod_pruned_dense(
      n, (unsigned int)q_vol, kappa.data(), kappa_res.data(), K.data());

    // Convert row-major actual K into the public column-major layout.
    for (int col = 0; col < M; ++col)
    {
      for (int row = 0; row < M; ++row)
      {
        L0_ref[(std::size_t)row + (std::size_t)M * col] =
          K[(std::size_t)row * M + col];
      }
    }
  }

  void embed_face_points(int face_id, const int* sigma, std::vector<Real>& Xf) const
  {
    Xf.assign((std::size_t)nq_face * D, (Real)0);

    if constexpr (D == 1)
    {
      Xf[0] = (face_id == 0) ? (Real)1 : (Real)0;
      (void)sigma;
    }
    else
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);

      for (int q = 0; q < nq_face; ++q)
      {
        Real mu[D];
        Real sum_y = (Real)0;
        for (int a = 1; a < D; ++a)
        {
          const Real y = Y_face[(std::size_t)q * (D - 1) + (a - 1)];
          mu[a] = y;
          sum_y += y;
        }
        mu[0] = (Real)1 - sum_y;

        Real lambda[D + 1];
        for (int a = 0; a <= D; ++a) { lambda[a] = (Real)0; }

        // sigma is local-face-position -> canonical-face-position.
        for (int local = 0; local < D; ++local)
        {
          const int canonical = sigma[local];
          lambda[fv[local]] = mu[canonical];
        }

        for (int a = 0; a < D; ++a)
        {
          Xf[(std::size_t)q * D + a] = lambda[a + 1];
        }
      }
    }
  }

  void build_face_operator_data()
  {
    T_ref.assign((std::size_t)kf * M * nsigma * nface, (Real)0);
    Fgrad_ref.assign((std::size_t)kf * M * D * nsigma * nface, (Real)0);
    Mface_ref.assign((std::size_t)kf * kf * nface, (Real)0);

    std::vector<int> alpha_vol, tail_vol;
    std::vector<Real> invh_vol;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha_vol, tail_vol, invh_vol);

    std::array<std::array<Real, D + 1>, D> k_deriv{};
    std::array<std::vector<int>, D> alpha_deriv;
    std::array<std::vector<int>, D> tail_deriv;
    std::array<std::vector<Real>, D> invh_deriv;
    std::vector<Real> D1((std::size_t)D * M * M, (Real)0); // row-major per axis

    for (int a = 0; a < D; ++a)
    {
      Basis<D,Real>::derivative_kappa_shift(kappa.data(), a, k_deriv[(std::size_t)a].data());
      Basis<D,Real>::build_structures(n, k_deriv[(std::size_t)a].data(),
                                      alpha_deriv[(std::size_t)a],
                                      tail_deriv[(std::size_t)a],
                                      invh_deriv[(std::size_t)a]);
      DMat<D,Real>::build_tprod_natural_pruned_dense(
        n, (unsigned int)q_vol, kappa.data(), a,
        D1.data() + (std::size_t)a * M * M);
    }

    for (int face_id = 0; face_id < nface; ++face_id)
    {
      const Real fscale = face_ref_scale[(std::size_t)face_id];

      // Face mass matrix for this embedded reference face:
      //   Mface_ref = V_face^T * diag(W_face * fscale) * V_face.
      std::vector<Real> WV_face((std::size_t)nq_face * kf, Real(0));
      for (int col = 0; col < kf; ++col)
      {
        for (int q = 0; q < nq_face; ++q)
        {
          WV_face[(std::size_t)q + (std::size_t)nq_face * col] =
            (W_face[(std::size_t)q] * fscale) *
            V_face[(std::size_t)q + (std::size_t)nq_face * col];
        }
      }
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        kf,
        kf,
        nq_face,
        Real(1),
        V_face.data(),
        nq_face,
        WV_face.data(),
        nq_face,
        Real(0),
        &Mface_ref[(std::size_t)kf * kf * face_id],
        kf);

      for (int sigma_index = 0; sigma_index < nsigma; ++sigma_index)
      {
        int sigma[D];
        lehmer_index_to_perm<D>(sigma_index, sigma);

        std::vector<Real> Xf;
        embed_face_points(face_id, sigma, Xf);

        std::vector<Real> Vv((std::size_t)nq_face * M, (Real)0);
        Basis<D,Real>::eval_all(
          Xf.data(), D, 1, nq_face, kappa.data(), n,
          alpha_vol.data(), tail_vol.data(), invh_vol.data(), Vv.data(), nq_face, nullptr);

        // T_ref block:
        //   T_ref = V_face^T * diag(W_face * fscale) * Vv.
        std::vector<Real> WVv((std::size_t)nq_face * M, Real(0));
        for (int col = 0; col < M; ++col)
        {
          for (int q = 0; q < nq_face; ++q)
          {
            WVv[(std::size_t)q + (std::size_t)nq_face * col] =
              (W_face[(std::size_t)q] * fscale) *
              Vv[(std::size_t)q + (std::size_t)nq_face * col];
          }
        }
        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasTrans,
          CblasNoTrans,
          kf,
          M,
          nq_face,
          Real(1),
          V_face.data(),
          nq_face,
          WVv.data(),
          nq_face,
          Real(0),
          &T_ref[idx_T(0, 0, sigma_index, face_id)],
          kf);

        for (int a = 0; a < D; ++a)
        {
          std::vector<Real> Vrng((std::size_t)nq_face * M, (Real)0);
          Basis<D,Real>::eval_all(
            Xf.data(), D, 1, nq_face, k_deriv[(std::size_t)a].data(), n,
            alpha_deriv[(std::size_t)a].data(),
            tail_deriv[(std::size_t)a].data(),
            invh_deriv[(std::size_t)a].data(),
            Vrng.data(), nq_face, nullptr);

          std::vector<Real> dV((std::size_t)nq_face * M, Real(0));
          // dV = Vrng * D_a.  Vrng is column-major nq_face x M.
          // D_a is stored row-major M x M; interpreted as column-major it is
          // D_a^T, so use transB = CblasTrans.
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasNoTrans,
            CblasTrans,
            nq_face,
            M,
            M,
            Real(1),
            Vrng.data(),
            nq_face,
            D1.data() + (std::size_t)a * M * M,
            M,
            Real(0),
            dV.data(),
            nq_face);

          // Fgrad_ref block:
          //   Fgrad_ref = V_face^T * diag(W_face * fscale) * dV.
          std::vector<Real> WdV((std::size_t)nq_face * M, Real(0));
          for (int col = 0; col < M; ++col)
          {
            for (int q = 0; q < nq_face; ++q)
            {
              WdV[(std::size_t)q + (std::size_t)nq_face * col] =
                (W_face[(std::size_t)q] * fscale) *
                dV[(std::size_t)q + (std::size_t)nq_face * col];
            }
          }
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasTrans,
            CblasNoTrans,
            kf,
            M,
            nq_face,
            Real(1),
            V_face.data(),
            nq_face,
            WdV.data(),
            nq_face,
            Real(0),
            &Fgrad_ref[idx_F(0, 0, a, sigma_index, face_id)],
            kf);
        }
      }
    }
  }

  std::size_t idx_T(int row, int col, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)sigma + (std::size_t)nsigma * face));
  }

  std::size_t idx_F(int row, int col, int axis, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)axis + (std::size_t)D *
      ((std::size_t)sigma + (std::size_t)nsigma * face)));
  }
};

} // namespace jsimplex

#endif // JPRECOMP_HH
