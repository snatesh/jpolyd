#ifndef JCONORMAL_HH
#define JCONORMAL_HH

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <jgeom.hh>
#include <jmult.hh>
#include <jprecomp.hh>

namespace jsimplex {

/*
  Assemble the degree-n co-normal face moment map for one affine simplex:

    F_A[f] c = int_{physical face f} psi (n^T A grad u_c) w_f dS.

  A is the ORIGINAL divergence-form tensor in the base volume kappa family,
  packed as A[(r*D+s)*dimPi(D,p_A)+j]. It must not be the sign-negated,
  kappa+2-promoted principal coefficient used by the expanded PDE adapter.

  face_sigma_index: length D+1, local face -> canonical orientation index.
  normal_scaled: row-major (D+1) x D, physical face scale * outward unit normal.
  face_quad: intrinsic common-face quadrature in pre.kappa_face, supporting
    degree n and exact for multiplication degree p_A with n-by-n face maps.

  F_out: stacked ((D+1)*kf) x M, column-major with leading dimension
    (D+1)*kf. The caller owns the multiplication workspace.

  The stored pre.T_ref and pre.Fgrad_ref include reference-face measure
  scale, so both are divided by pre.face_ref_scale[f] to recover canonical
  face polynomial coefficients. normal_scaled accounts for the physical
  measure exactly once.
*/
template<int D, class Real>
void jdsimplex_assemble_F_conormal_full_common_face(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const int* face_sigma_index,
  const Real* normal_scaled,
  int p_A,
  const Real* A_coefficients,
  const MultByQQuadraturePlan<D-1,Real>& face_quad,
  MultByQQuadratureWorkspace<Real>& mult_workspace,
  Real* F_out)
{
  if (!geom.valid || !face_sigma_index || !normal_scaled ||
      !A_coefficients || !F_out)
    throw std::invalid_argument("co-normal face map: invalid input");
  if (p_A < 0 || p_A > pre.n)
    throw std::invalid_argument("co-normal face map: invalid diffusion degree");
  if (face_quad.max_degree < pre.n ||
      face_quad.q_order <= 0 ||
      2 * face_quad.q_order - 1 < 2 * pre.n + p_A)
    throw std::invalid_argument("co-normal face map: insufficient multiplication quadrature");
  if constexpr (D > 1)
  {
    if (2 * pre.q_face - 1 < pre.n + std::max(pre.n - 1, p_A))
      throw std::invalid_argument("co-normal face map: insufficient reference face quadrature");
  }

  const int M = pre.M;
  const int kf = pre.kf;
  const int nb = (D + 1) * kf;
  const int Mp = Basis<D,Real>::dim_Pi(p_A);
  const std::size_t ref_block = (std::size_t)kf * M;

  std::fill(F_out, F_out + (std::size_t)nb * M, Real(0));

  std::vector<Real> weighted_tensor((std::size_t)Mp, Real(0));
  std::vector<Real> face_coeff((std::size_t)kf, Real(0));
  std::vector<Real> multiplier((std::size_t)kf * kf, Real(0));
  std::vector<Real> physical_gradient((std::size_t)kf * M, Real(0));

  for (int f = 0; f < D + 1; ++f)
  {
    const int sigma = face_sigma_index[f];
    if (sigma < 0 || sigma >= pre.nsigma)
      throw std::invalid_argument("co-normal face map: invalid face sigma");

    const Real refscale = pre.face_ref_scale[(std::size_t)f];
    if (!(refscale > Real(0)))
      throw std::invalid_argument("co-normal face map: invalid reference face scale");
    const Real invscale = Real(1) / refscale;
    const std::size_t oriented =
      (std::size_t)sigma + (std::size_t)pre.nsigma * f;
    const Real* Tref = pre.T_ref.data() + ref_block * oriented;
    Real* Ff = F_out + (std::size_t)f * kf;

    for (int s = 0; s < D; ++s)
    {
      // Form normal_scaled^T A column s in the original volume basis.
      std::fill(weighted_tensor.begin(), weighted_tensor.end(), Real(0));
      for (int r = 0; r < D; ++r)
      {
        const Real nr = normal_scaled[(std::size_t)f * D + r];
        const Real* Ars = A_coefficients + ((std::size_t)r * D + s) * Mp;
        for (int j = 0; j < Mp; ++j)
          weighted_tensor[(std::size_t)j] += nr * Ars[j];
      }

      // Intrinsic canonical face coefficients of normal_scaled^T A[:,s].
      std::fill(face_coeff.begin(), face_coeff.end(), Real(0));
      for (int j = 0; j < Mp; ++j)
        for (int i = 0; i < kf; ++i)
          face_coeff[(std::size_t)i] +=
            invscale * Tref[(std::size_t)i + (std::size_t)kf * j] *
            weighted_tensor[(std::size_t)j];

      assemble_restricted_mult_quadrature<D-1,Real>(
        face_quad, face_coeff.data(), p_A, pre.n, pre.n,
        mult_workspace, multiplier.data());

      // Intrinsic face coefficients of d_{x_s} phi_j.
      std::fill(physical_gradient.begin(), physical_gradient.end(), Real(0));
      for (int a = 0; a < D; ++a)
      {
        const Real factor = geom.BinvT[(std::size_t)s + (std::size_t)D * a]
          * invscale;
        const Real* Fgrad = pre.Fgrad_ref.data()
          + ref_block * ((std::size_t)a + (std::size_t)D * oriented);
        for (std::size_t j = 0; j < physical_gradient.size(); ++j)
          physical_gradient[j] += factor * Fgrad[j];
      }

      // Accumulate M_{normal_scaled^T A[:,s]} * G_s into face rows.
      detail::BlasGemm<Real>::run(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        kf, M, kf,
        Real(1), multiplier.data(), kf,
        physical_gradient.data(), kf,
        Real(1), Ff, nb);
    }
  }
}

/*
  Assemble the HIGH-degree physical co-normal flux observation Q_H on a
  selected set of local faces, normally the element's global exterior faces.

  For u in Pi_n and diffusion A in Pi_pA, the physical co-normal flux has
  degree at most p_H = n + p_A - 1.  Only degrees n+1,...,p_H are retained.
  The degree-n trace and numerical flux skeleton are NOT enlarged.

  selected_faces: length n_selected, local face IDs in desired row order.
  k_H = dim(Pi_pH^(D-1)) - pre.kf, or zero for p_A <= 1 or D == 1.
  QH_out: column-major (n_selected*k_H) x pre.M.  Consecutive selected
          faces occupy k_H rows, ordered by increasing canonical face mode.
          If k_H == 0 or n_selected == 0, no output entries are written.

  Inputs A_coefficients, face_sigma_index, and normal_scaled have exactly
  the same conventions as jdsimplex_assemble_F_conormal_full_common_face.
  In particular, normal_scaled includes the PHYSICAL face scale already.

  The face multiplier is formed as M_w^{p_H <- n}; its rows 0:pre.kf are
  discarded ONLY AFTER the full rectangular Galerkin multiplication.
  This is the physical high flux, without the degree-n tau penalty.
*/
template<int D, class Real>
void jdsimplex_assemble_QH_conormal_selected_common_face(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const int* face_sigma_index,
  const Real* normal_scaled,
  int p_A,
  const Real* A_coefficients,
  const MultByQQuadraturePlan<D-1,Real>& face_quad,
  MultByQQuadratureWorkspace<Real>& mult_workspace,
  int n_selected,
  const int* selected_faces,
  Real* QH_out)
{
  if (p_A < 0 || p_A > pre.n || n_selected < 0 || n_selected > D + 1)
    throw std::invalid_argument("high co-normal map: invalid degrees/count");

  const int p_H = pre.n + std::max(0, p_A - 1);
  const int kL = pre.kf;
  const int kFull = Basis<D-1,Real>::dim_Pi(p_H);
  const int kH = kFull - kL;

  // Includes the zero-dimensional face case D=1.
  if (n_selected == 0 || kH == 0)
    return;

  if (!geom.valid || !face_sigma_index || !normal_scaled ||
      !A_coefficients || !selected_faces || !QH_out)
    throw std::invalid_argument("high co-normal map: null/invalid input");
  if (face_quad.max_degree < p_H || face_quad.q_order <= 0 ||
      2 * face_quad.q_order - 1 < p_H + pre.n + p_A)
    throw std::invalid_argument(
      "high co-normal map: insufficient rectangular multiplication quadrature");
  if constexpr (D > 1)
  {
    if (2 * pre.q_face - 1 < pre.n + std::max(pre.n - 1, p_A))
      throw std::invalid_argument(
        "high co-normal map: insufficient reference face quadrature");
  }

  const int M = pre.M;
  const int Mp = Basis<D,Real>::dim_Pi(p_A);
  const int ldQ = n_selected * kH;
  const std::size_t ref_block = (std::size_t)kL * M;
  std::fill(QH_out, QH_out + (std::size_t)ldQ * M, Real(0));

  std::array<bool,D+1> seen{};
  std::vector<Real> weighted_tensor((std::size_t)Mp, Real(0));
  std::vector<Real> face_coeff((std::size_t)kL, Real(0));
  std::vector<Real> multiplier((std::size_t)kFull * kL, Real(0));
  std::vector<Real> physical_gradient((std::size_t)kL * M, Real(0));

  for (int selected = 0; selected < n_selected; ++selected)
  {
    const int f = selected_faces[selected];
    if (f < 0 || f >= D+1 || seen[(std::size_t)f])
      throw std::invalid_argument("high co-normal map: invalid/duplicate face ID");
    seen[(std::size_t)f] = true;

    const int sigma = face_sigma_index[f];
    if (sigma < 0 || sigma >= pre.nsigma)
      throw std::invalid_argument("high co-normal map: invalid face sigma");

    const Real refscale = pre.face_ref_scale[(std::size_t)f];
    if (!(refscale > Real(0)))
      throw std::invalid_argument("high co-normal map: invalid reference face scale");
    const Real invscale = Real(1) / refscale;
    const std::size_t oriented =
      (std::size_t)sigma + (std::size_t)pre.nsigma * f;
    const Real* Tref = pre.T_ref.data() + ref_block * oriented;
    Real* Qf = QH_out + (std::size_t)selected * kH;

    for (int s = 0; s < D; ++s)
    {
      // Coefficients of (physical scale * n)^T A[:,s] in volume kappa.
      std::fill(weighted_tensor.begin(), weighted_tensor.end(), Real(0));
      for (int r = 0; r < D; ++r)
      {
        const Real nr = normal_scaled[(std::size_t)f * D + r];
        const Real* Ars = A_coefficients + ((std::size_t)r * D + s) * Mp;
        for (int j = 0; j < Mp; ++j)
          weighted_tensor[(std::size_t)j] += nr * Ars[j];
      }

      std::fill(face_coeff.begin(), face_coeff.end(), Real(0));
      for (int j = 0; j < Mp; ++j)
        for (int i = 0; i < kL; ++i)
          face_coeff[(std::size_t)i] +=
            invscale * Tref[(std::size_t)i + (std::size_t)kL * j]
            * weighted_tensor[(std::size_t)j];

      // Full rectangular multiplication is essential: truncating to n
      // before multiplying would erase the desired high modes.
      assemble_restricted_mult_quadrature<D-1,Real>(
        face_quad, face_coeff.data(), p_A, pre.n, p_H,
        mult_workspace, multiplier.data());

      std::fill(physical_gradient.begin(), physical_gradient.end(), Real(0));
      for (int a = 0; a < D; ++a)
      {
        const Real factor =
          geom.BinvT[(std::size_t)s + (std::size_t)D * a] * invscale;
        const Real* Fgrad = pre.Fgrad_ref.data()
          + ref_block * ((std::size_t)a + (std::size_t)D * oriented);
        for (std::size_t j = 0; j < physical_gradient.size(); ++j)
          physical_gradient[j] += factor * Fgrad[j];
      }

      // GEMM starts at row kL of the full kFull x kL multiplier.
      // No temporary high-row block is needed; the leading dimension
      // remains kFull even though GEMM sees only the bottom kH rows.
      detail::BlasGemm<Real>::run(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        kH, M, kL,
        Real(1), multiplier.data() + kL, kFull,
        physical_gradient.data(), kL,
        Real(1), Qf, ldQ);
    }
  }
}

} // namespace jsimplex

#endif // JCONORMAL_HH
