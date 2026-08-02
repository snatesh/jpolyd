#ifndef JLEAF_HH
#define JLEAF_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <jdetail.hh>
#include <jgeom.hh>
#include <jlaplace.hh>
#include <jelliptic.hh>
#include <jperms.hh>
#include <jprecomp.hh>
#include <jlsmr.hh>

namespace jsimplex {

enum class LeafOperatorMode
{
  MatrixFree,
  Dense,
  Verify
};

template<class Real>
struct LeafOptions
{
  // Dense preserves every legacy constructor/reset call and every public
  // dense diagnostic field. MatrixFree stores neither L, T, F, nor A_tau.
  // Verify builds both backends, uses the matrix-free solve, and compares it
  // with the dense fallback.
  LeafOperatorMode operator_mode = LeafOperatorMode::Dense;

  Real verify_tolerance =
    std::is_same_v<Real,double> ? Real(1.0e-9) : Real(2.0e-4);

  bool verify_each_solve = true;

  // A positive value bypasses both the legacy dense Frobenius calculation and
  // the matrix-free identity-column norm sweep. Zero preserves the old
  // row-RMS normalization exactly.
  Real interior_scale_override = Real(0);
};


template<int D, class Real = double>
class Leaf
{
public:
  static_assert(D >= 1, "Leaf requires D>=1");
  static_assert(
    std::is_floating_point<Real>::value,
    "Leaf requires floating-point Real");
  static_assert(
    std::is_same_v<Real,float> || std::is_same_v<Real,double>,
    "Leaf currently supports only Real=float or Real=double");

  using LsmrOptions = detail::LsmrOptions<Real>;
  using LsmrInfo = detail::LsmrInfo<Real>;
  using Options = LeafOptions<Real>;

  const RefSimplexPrecomp<D,Real>* pre = nullptr;

  int n = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int nface = D + 1;
  int nb = 0;
  int ntau_rows = 0;

  std::array<int,D + 1> global_vids{};

  // Geometry. V_phys is D x (D+1), column-major.
  std::vector<Real> V_phys;
  DSimplexGeom<D,Real> geom{};
  std::array<Real,D * D> G{};

  // Face metadata and physical geometry.
  std::array<int,D + 1> face_sigma_index{};
  std::vector<Real> face_scale;
  std::vector<Real> face_ref_scale;
  std::vector<Real> face_h;
  std::vector<Real> unit_normal;
  std::vector<Real> normal_scaled;

  // Legacy dense diagnostics/fallback. These are populated only in Dense and
  // Verify mode.
  //
  //   L:     m_int x M
  //   T,F:   nb x M
  //   A_tau: (m_int + nb) x M
  std::vector<Real> L;
  std::vector<Real> T;
  std::vector<Real> F;
  std::vector<Real> A_tau;

  std::vector<Real> tau_face;
  std::vector<Real> tau_rows;
  std::vector<Real> sqrt_tau_rows;
  Real tau_C = Real(10);
  Real sL = Real(1);

  LsmrOptions lsmr_options{};
  Options leaf_options{};
  LeafOperatorMode operator_mode = LeafOperatorMode::Dense;

  Leaf() = default;

  // Legacy Poisson/Laplace constructor. Existing calls remain Dense because
  // LeafOptions defaults to Dense.
  Leaf(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    Real tau_C_in = Real(10),
    const LsmrOptions& opts = LsmrOptions(),
    const Options& leaf_opts = Options())
  {
    reset(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      tau_C_in,
      opts,
      leaf_opts);
  }

  // Legacy elliptic constructor. EllipticWorkspace is the backward-compatible
  // alias for EllipticDenseWorkspace.
  Leaf(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticWorkspace<D,Real>& elliptic_work,
    Real tau_C_in = Real(10),
    const LsmrOptions& opts = LsmrOptions(),
    const Options& leaf_opts = Options())
  {
    reset(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      elliptic_plan,
      coeffs,
      elliptic_work,
      tau_C_in,
      opts,
      leaf_opts);
  }

  // New mode-aware elliptic constructor. MatrixFree requires no dense
  // workspace. Dense/Verify create a temporary dense workspace internally.
  Leaf(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    Real tau_C_in,
    const LsmrOptions& opts,
    const Options& leaf_opts)
  {
    reset(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      elliptic_plan,
      coeffs,
      tau_C_in,
      opts,
      leaf_opts);
  }

  void reset(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    Real tau_C_in = Real(10),
    const LsmrOptions& opts = LsmrOptions(),
    const Options& leaf_opts = Options())
  {
    initialize_common(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      tau_C_in,
      opts,
      leaf_opts);

    EllipticDegreeSpec degree_spec;
    degree_spec.p2 = 0;
    degree_spec.p1 = -1;
    degree_spec.p0 = -1;

    owned_elliptic_plan_ =
      std::make_unique<EllipticPlan<D,Real>>(
        *pre,
        degree_spec);
    elliptic_plan_ = owned_elliptic_plan_.get();

    std::array<Real, D * D> identity_coefficients{};
    const Real one_coeff =
      Real(1) / elliptic_plan_->phi0_res;
    for (int axis = 0; axis < D; ++axis)
    {
      identity_coefficients[
        (std::size_t)axis * D
        + (std::size_t)axis] = one_coeff;
    }

    EllipticElementCoefficientsView<D,Real>
      poisson_coefficients;
    poisson_coefficients.A =
      identity_coefficients.data();
    poisson_coefficients.b = nullptr;
    poisson_coefficients.c = nullptr;

    if (uses_matrix_free_backend())
    {
      initialize_action_backend(
        *elliptic_plan_,
        poisson_coefficients);
    }

    if (uses_dense_backend())
    {
      EllipticDenseWorkspace<D,Real> dense_work(
        *elliptic_plan_);
      assemble_dense_backend(
        *elliptic_plan_,
        poisson_coefficients,
        dense_work);
    }
    else
    {
      clear_dense_storage();
      set_matrix_free_interior_scale();
    }

    finalize_operator_initialization();
  }

  // Legacy reset overload: unchanged call shape, Dense by default.
  void reset(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticWorkspace<D,Real>& elliptic_work,
    Real tau_C_in = Real(10),
    const LsmrOptions& opts = LsmrOptions(),
    const Options& leaf_opts = Options())
  {
    initialize_common(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      tau_C_in,
      opts,
      leaf_opts);

    elliptic_plan_ = &elliptic_plan;

    if (uses_matrix_free_backend())
    {
      initialize_action_backend(
        elliptic_plan,
        coeffs);
    }

    if (uses_dense_backend())
    {
      assemble_dense_backend(
        elliptic_plan,
        coeffs,
        elliptic_work);
    }
    else
    {
      clear_dense_storage();
      set_matrix_free_interior_scale();
    }

    finalize_operator_initialization();
  }

  // New reset overload for MatrixFree/Verify callers that should not allocate a
  // dense workspace unless the selected mode requires it.
  void reset(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    Real tau_C_in,
    const LsmrOptions& opts,
    const Options& leaf_opts)
  {
    initialize_common(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      tau_C_in,
      opts,
      leaf_opts);

    elliptic_plan_ = &elliptic_plan;

    if (uses_matrix_free_backend())
    {
      initialize_action_backend(
        elliptic_plan,
        coeffs);
    }

    if (uses_dense_backend())
    {
      EllipticDenseWorkspace<D,Real> dense_work(
        elliptic_plan);
      assemble_dense_backend(
        elliptic_plan,
        coeffs,
        dense_work);
    }
    else
    {
      clear_dense_storage();
      set_matrix_free_interior_scale();
    }

    finalize_operator_initialization();
  }

  bool uses_matrix_free_backend() const
  {
    return operator_mode != LeafOperatorMode::Dense;
  }

  bool uses_dense_backend() const
  {
    return operator_mode != LeafOperatorMode::MatrixFree;
  }

  bool has_dense_local_operator() const
  {
    return !L.empty() && !A_tau.empty();
  }

  int face_offset(int face_id) const
  {
    check_face_id(face_id);
    return face_id * kf;
  }

  std::array<int,D> face_key(int face_id) const
  {
    check_face_id(face_id);
    int fv[D];
    dsimplex_face_vertices<D>(face_id, fv);

    std::array<int,D> key{};
    for (int i = 0; i < D; ++i)
    {
      key[(std::size_t)i] =
        global_vids[(std::size_t)fv[i]];
    }
    std::sort(key.begin(), key.end());
    return key;
  }

  void set_tau_face(const Real* tau_face_in)
  {
    if (!tau_face_in)
    {
      throw std::invalid_argument(
        "Leaf::set_tau_face: null input");
    }

    tau_face.assign((std::size_t)nface, Real(0));
    tau_rows.assign((std::size_t)nb, Real(0));
    sqrt_tau_rows.assign((std::size_t)nb, Real(0));

    for (int face = 0; face < nface; ++face)
    {
      if (!(tau_face_in[face] > Real(0)))
      {
        throw std::invalid_argument(
          "Leaf::set_tau_face: tau must be positive");
      }

      tau_face[(std::size_t)face] =
        tau_face_in[face];
      const Real square_root =
        std::sqrt(tau_face_in[face]);

      for (int row = 0; row < kf; ++row)
      {
        const int global_row = face * kf + row;
        tau_rows[(std::size_t)global_row] =
          tau_face_in[face];
        sqrt_tau_rows[(std::size_t)global_row] =
          square_root;
      }
    }

    build_A_tau();
    if (operator_mode == LeafOperatorMode::Verify)
    {
      verify_backends();
    }
  }

  void build_tau_from_face_diameters(Real C)
  {
    if (!(C > Real(0)))
    {
      throw std::invalid_argument(
        "Leaf: tau constant must be positive");
    }

    tau_face.assign((std::size_t)nface, Real(0));
    tau_rows.assign((std::size_t)nb, Real(0));
    sqrt_tau_rows.assign((std::size_t)nb, Real(0));

    const Real np1 = Real(n + 1);
    for (int face = 0; face < nface; ++face)
    {
      const Real h = std::max(
        face_h[(std::size_t)face],
        Real(100) * std::numeric_limits<Real>::epsilon());
      const Real tau = C * np1 * np1 / h;
      const Real square_root = std::sqrt(tau);

      tau_face[(std::size_t)face] = tau;
      for (int row = 0; row < kf; ++row)
      {
        const int global_row = face * kf + row;
        tau_rows[(std::size_t)global_row] = tau;
        sqrt_tau_rows[(std::size_t)global_row] =
          square_root;
      }
    }
  }

  // Apply T or F to nrhs column vectors. MatrixFree and Verify traverse the
  // reference CSC blocks directly. Dense uses the legacy explicit maps.
  void apply_trace_columns(
    const Real* X,
    int ldx,
    int nrhs,
    Real* Y,
    int ldy) const
  {
    validate_boundary_dense_args(
      X, ldx, nrhs, Y, ldy,
      "Leaf::apply_trace_columns");

    if (operator_mode == LeafOperatorMode::Dense)
    {
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        nb,
        nrhs,
        M,
        Real(1),
        T.data(),
        nb,
        X,
        ldx,
        Real(0),
        Y,
        ldy);
      return;
    }

    apply_trace_csc_columns(
      X,
      ldx,
      nrhs,
      Y,
      ldy);
  }

  void apply_flux_columns(
    const Real* X,
    int ldx,
    int nrhs,
    Real* Y,
    int ldy) const
  {
    validate_boundary_dense_args(
      X, ldx, nrhs, Y, ldy,
      "Leaf::apply_flux_columns");

    if (operator_mode == LeafOperatorMode::Dense)
    {
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        nb,
        nrhs,
        M,
        Real(1),
        F.data(),
        nb,
        X,
        ldx,
        Real(0),
        Y,
        ldy);
      return;
    }

    apply_flux_csc_columns(
      X,
      ldx,
      nrhs,
      Y,
      ldy);
  }

  LsmrInfo apply(
    const Real* lambda,
    const Real* f_int,
    Real* c_out,
    Real* trace_out,
    Real* raw_flux_out,
    Real* aug_flux_out,
    Real* trace_mismatch_out = nullptr,
    Real* pde_residual_out = nullptr) const
  {
    if (!lambda)
    {
      throw std::invalid_argument(
        "Leaf::apply: null lambda");
    }
    if (!f_int)
    {
      throw std::invalid_argument(
        "Leaf::apply: null f_int");
    }
    if (!c_out)
    {
      throw std::invalid_argument(
        "Leaf::apply: null c_out");
    }
    if (!trace_out || !raw_flux_out || !aug_flux_out)
    {
      throw std::invalid_argument(
        "Leaf::apply: null boundary output");
    }

    std::vector<Real> rhs(
      (std::size_t)ntau_rows,
      Real(0));

    for (int row = 0; row < m_int; ++row)
    {
      rhs[(std::size_t)row] =
        -sL * f_int[row];
    }
    for (int row = 0; row < nb; ++row)
    {
      rhs[(std::size_t)m_int + row] =
        sqrt_tau_rows[(std::size_t)row]
        * lambda[row];
    }

    LsmrInfo info{};
    int return_code = 0;

    if (operator_mode == LeafOperatorMode::Dense)
    {
      return_code = lsmr_dense_solve_colmajor<Real>(
        ntau_rows,
        M,
        A_tau.data(),
        rhs.data(),
        c_out,
        lsmr_options,
        &info);
    }
    else
    {
      return_code = solve_matrix_free(
        rhs.data(),
        c_out,
        &info);
    }

    if (return_code != 0)
    {
      throw std::runtime_error(
        "Leaf::apply: LSMR solve failed");
    }

    if (operator_mode == LeafOperatorMode::Verify
        && leaf_options.verify_each_solve)
    {
      verify_solution_against_dense(
        rhs.data(),
        c_out);
    }

    apply_trace_columns(
      c_out,
      M,
      1,
      trace_out,
      nb);
    apply_flux_columns(
      c_out,
      M,
      1,
      raw_flux_out,
      nb);

    for (int row = 0; row < nb; ++row)
    {
      const Real mismatch =
        trace_out[row] - lambda[row];
      if (trace_mismatch_out)
      {
        trace_mismatch_out[row] = mismatch;
      }
      aug_flux_out[row] =
        raw_flux_out[row]
        + tau_rows[(std::size_t)row] * mismatch;
    }

    if (pde_residual_out)
    {
      apply_interior_operator(
        c_out,
        pde_residual_out);
      for (int row = 0; row < m_int; ++row)
      {
        pde_residual_out[row] += f_int[row];
      }
    }

    return info;
  }

  LsmrInfo apply_zero_source(
    const Real* lambda,
    Real* c_out,
    Real* trace_out,
    Real* raw_flux_out,
    Real* aug_flux_out,
    Real* trace_mismatch_out = nullptr,
    Real* pde_residual_out = nullptr) const
  {
    std::vector<Real> zero_source(
      (std::size_t)m_int,
      Real(0));

    return apply(
      lambda,
      zero_source.data(),
      c_out,
      trace_out,
      raw_flux_out,
      aug_flux_out,
      trace_mismatch_out,
      pde_residual_out);
  }

private:
  void check_face_id(int face_id) const
  {
    if (face_id < 0 || face_id >= nface) { throw std::out_of_range("Leaf: face_id out of range"); }
  }

  void build_geometry()
  {
    dsimplex_affine_from_verts<D,Real>(V_phys.data(), geom);
    if (!geom.valid) { throw std::runtime_error("Leaf: singular affine simplex"); }

    for (int i = 0; i < D; ++i)
    {
      for (int j = 0; j < D; ++j)
      {
        Real acc = Real(0);
        for (int k = 0; k < D; ++k)
        {
          acc += geom.BinvT[(std::size_t)k + (std::size_t)D * i]
               * geom.BinvT[(std::size_t)k + (std::size_t)D * j];
        }
        G[(std::size_t)i + (std::size_t)D * j] = acc;
      }
    }

    for (int f = 0; f < nface; ++f)
    {
      face_sigma_index[(std::size_t)f] =
        dsimplex_compute_face_sigma<D>(global_vids.data(), f);
    }

    build_physical_face_geometry();
  }

  void build_physical_face_geometry()
  {
    face_scale.assign((std::size_t)nface, Real(0));
    face_ref_scale.assign((std::size_t)nface, Real(0));
    face_h.assign((std::size_t)nface, Real(0));
    unit_normal.assign((std::size_t)nface * D, Real(0));
    normal_scaled.assign((std::size_t)nface * D, Real(0));

    for (int f = 0; f < nface; ++f)
    {
      face_ref_scale[(std::size_t)f] = pre->face_ref_scale[(std::size_t)f];
    }

    if constexpr (D == 1)
    {
      const Real x0 = V_phys[0];
      const Real x1 = V_phys[1];
      const Real len = std::abs(x1 - x0);
      const Real sgn = (x1 >= x0) ? Real(1) : Real(-1);

      face_scale[0] = Real(1);
      face_scale[1] = Real(1);
      face_h[0] = len;
      face_h[1] = len;

      // face 0 is opposite vertex 0, i.e. endpoint vertex 1.
      unit_normal[0] = sgn;
      unit_normal[1] = -sgn;
      normal_scaled[0] = unit_normal[0];
      normal_scaled[1] = unit_normal[1];
    }
    else
    {
      for (int f = 0; f < nface; ++f)
      {
        int fv[D];
        dsimplex_face_vertices<D>(f, fv);

        Real Vface[D * D]; // D x D, columns are face vertices.
        for (int j = 0; j < D; ++j)
        {
          const int v = fv[j];
          for (int r = 0; r < D; ++r)
          {
            Vface[(std::size_t)r + (std::size_t)D * j] =
              V_phys[(std::size_t)r + (std::size_t)D * v];
          }
        }

        const Real s = dsimplex_embedded_simplex_measure_scale_colmajor<D,D-1,Real>(Vface);
        if (!(s > Real(0))) { throw std::runtime_error("Leaf: degenerate physical face"); }
        face_scale[(std::size_t)f] = s;

        Real h = Real(0);
        for (int a = 0; a < D; ++a)
        {
          for (int b = a + 1; b < D; ++b)
          {
            Real d2 = Real(0);
            for (int r = 0; r < D; ++r)
            {
              const Real diff = Vface[(std::size_t)r + (std::size_t)D * a]
                              - Vface[(std::size_t)r + (std::size_t)D * b];
              d2 += diff * diff;
            }
            h = std::max(h, std::sqrt(d2));
          }
        }
        face_h[(std::size_t)f] = h;

        std::array<Real,D> nvec{};
        compute_face_unit_normal_from_svd(Vface, nvec.data());

        const int opp = f;
        Real dot_to_opp = Real(0);
        for (int r = 0; r < D; ++r)
        {
          const Real p_opp = V_phys[(std::size_t)r + (std::size_t)D * opp];
          const Real p0 = Vface[(std::size_t)r];
          dot_to_opp += nvec[(std::size_t)r] * (p_opp - p0);
        }
        if (dot_to_opp > Real(0))
        {
          for (int r = 0; r < D; ++r) { nvec[(std::size_t)r] = -nvec[(std::size_t)r]; }
        }

        for (int r = 0; r < D; ++r)
        {
          unit_normal[(std::size_t)f * D + r] = nvec[(std::size_t)r];
          normal_scaled[(std::size_t)f * D + r] = s * nvec[(std::size_t)r];
        }
      }
    }
  }

  static void compute_face_unit_normal_from_svd(const Real* Vface, Real* n_out)
  {
    static_assert(D >= 2, "SVD normal helper only used for D>=2");
    if (!Vface || !n_out) { throw std::invalid_argument("Leaf: null normal helper input"); }

    constexpr int m = D - 1;
    constexpr int ncols = D;

    std::vector<Real> A((std::size_t)m * ncols, Real(0)); // A = E^T, col-major m x D.
    for (int col = 0; col < ncols; ++col)
    {
      for (int row = 0; row < m; ++row)
      {
        A[(std::size_t)row + (std::size_t)m * col] =
          Vface[(std::size_t)col + (std::size_t)D * (row + 1)]
        - Vface[(std::size_t)col + (std::size_t)D * 0];
      }
    }

    std::vector<Real> S((std::size_t)std::min(m, ncols), Real(0));
    std::vector<Real> U((std::size_t)m * m, Real(0));
    std::vector<Real> VT((std::size_t)ncols * ncols, Real(0));

    const lapack_int ret = detail::LapackGesdd<Real>::run(
      'A',
      (lapack_int)m,
      (lapack_int)ncols,
      A.data(),
      (lapack_int)m,
      S.data(),
      U.data(),
      (lapack_int)m,
      VT.data(),
      (lapack_int)ncols);

    if (ret != 0) { throw std::runtime_error("Leaf: SVD failed while computing face normal"); }

    Real nrm2 = Real(0);
    for (int r = 0; r < D; ++r)
    {
      const Real nr = VT[(std::size_t)(D - 1) + (std::size_t)D * r];
      n_out[r] = nr;
      nrm2 += nr * nr;
    }
    const Real nrm = std::sqrt(nrm2);
    if (!(nrm > Real(0))) { throw std::runtime_error("Leaf: zero face normal from SVD"); }
    for (int r = 0; r < D; ++r) { n_out[r] /= nrm; }
  }


  std::unique_ptr<EllipticPlan<D,Real>>
    owned_elliptic_plan_;
  const EllipticPlan<D,Real>* elliptic_plan_ = nullptr;

  std::vector<Real> owned_A_;
  std::vector<Real> owned_b_;
  std::vector<Real> owned_c_;
  EllipticElementCoefficientsView<D,Real>
    owned_coeffs_{};

  mutable std::unique_ptr<
    EllipticActionWorkspace<D,Real>>
    elliptic_action_work_;

  // Physical factors multiplying the reference boundary CSC blocks.
  std::vector<Real> trace_ratio_face_;
  std::vector<Real> flux_eta_;

  // One workspace per Leaf. Parallel leaves are independent; calls on the same
  // Leaf must not overlap.
  mutable std::vector<Real> action_coeff_input_;
  mutable std::vector<Real> action_coeff_output_;
  mutable std::vector<Real> action_row_input_;
  mutable std::vector<Real> action_row_output_;
  mutable std::vector<Real> action_interior_scaled_;
  mutable std::vector<Real> action_boundary_scaled_;

  void initialize_common(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    Real tau_C_in,
    const LsmrOptions& opts,
    const Options& leaf_opts)
  {
    if (!V_phys_colmajor)
    {
      throw std::invalid_argument(
        "Leaf: null V_phys");
    }
    if (!global_vids_in)
    {
      throw std::invalid_argument(
        "Leaf: null global_vids");
    }
    if (!(tau_C_in > Real(0)))
    {
      throw std::invalid_argument(
        "Leaf: tau constant must be positive");
    }
    if (leaf_opts.verify_tolerance < Real(0))
    {
      throw std::invalid_argument(
        "Leaf: verify tolerance must be nonnegative");
    }
    if (leaf_opts.interior_scale_override < Real(0))
    {
      throw std::invalid_argument(
        "Leaf: interior scale override must be nonnegative");
    }

    pre = &pre_in;
    n = pre->n;
    M = pre->M;
    m_int = pre->m_int;
    kf = pre->kf;
    nface = D + 1;
    nb = nface * kf;
    ntau_rows = m_int + nb;
    tau_C = tau_C_in;
    lsmr_options = opts;
    leaf_options = leaf_opts;
    operator_mode = leaf_options.operator_mode;

    V_phys.assign(
      V_phys_colmajor,
      V_phys_colmajor
        + (std::size_t)D * (D + 1));
    for (int vertex = 0;
         vertex <= D;
         ++vertex)
    {
      global_vids[(std::size_t)vertex] =
        global_vids_in[vertex];
    }

    owned_elliptic_plan_.reset();
    elliptic_plan_ = nullptr;
    owned_A_.clear();
    owned_b_.clear();
    owned_c_.clear();
    owned_coeffs_ = {};
    elliptic_action_work_.reset();

    clear_dense_storage();

    build_geometry();
    build_boundary_action_metadata();

    action_coeff_input_.assign(
      (std::size_t)M,
      Real(0));
    action_coeff_output_.assign(
      (std::size_t)M,
      Real(0));
    action_row_input_.assign(
      (std::size_t)ntau_rows,
      Real(0));
    action_row_output_.assign(
      (std::size_t)ntau_rows,
      Real(0));
    action_interior_scaled_.assign(
      (std::size_t)m_int,
      Real(0));
    action_boundary_scaled_.assign(
      (std::size_t)nb,
      Real(0));
  }

  void finalize_operator_initialization()
  {
    build_tau_from_face_diameters(tau_C);
    build_A_tau();

    if (operator_mode == LeafOperatorMode::Verify)
    {
      verify_backends();
    }
  }

  void initialize_action_backend(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs)
  {
    copy_coefficients(plan, coeffs);
    elliptic_action_work_ =
      std::make_unique<
        EllipticActionWorkspace<D,Real>>(plan);
  }

  void copy_coefficients(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs)
  {
    const int Mp2 = plan.coefficient_size(2);
    const int Mp1 = plan.coefficient_size(1);
    const int Mp0 = plan.coefficient_size(0);

    if (plan.degrees.p2 >= 0)
    {
      if (!coeffs.A)
      {
        throw std::invalid_argument(
          "Leaf: null principal coefficients");
      }
      owned_A_.assign(
        coeffs.A,
        coeffs.A
          + (std::size_t)D * D * Mp2);
    }
    else
    {
      owned_A_.clear();
    }

    if (plan.degrees.p1 >= 0)
    {
      if (!coeffs.b)
      {
        throw std::invalid_argument(
          "Leaf: null first-order coefficients");
      }
      owned_b_.assign(
        coeffs.b,
        coeffs.b
          + (std::size_t)D * Mp1);
    }
    else
    {
      owned_b_.clear();
    }

    if (plan.degrees.p0 >= 0)
    {
      if (!coeffs.c)
      {
        throw std::invalid_argument(
          "Leaf: null zero-order coefficients");
      }
      owned_c_.assign(
        coeffs.c,
        coeffs.c + Mp0);
    }
    else
    {
      owned_c_.clear();
    }

    owned_coeffs_.A =
      owned_A_.empty() ? nullptr : owned_A_.data();
    owned_coeffs_.b =
      owned_b_.empty() ? nullptr : owned_b_.data();
    owned_coeffs_.c =
      owned_c_.empty() ? nullptr : owned_c_.data();
  }

  void clear_dense_storage()
  {
    L.clear();
    T.clear();
    F.clear();
    A_tau.clear();
  }

  void allocate_dense_storage()
  {
    L.assign(
      (std::size_t)m_int * M,
      Real(0));
    T.assign(
      (std::size_t)nb * M,
      Real(0));
    F.assign(
      (std::size_t)nb * M,
      Real(0));
  }

  void assemble_dense_backend(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticDenseWorkspace<D,Real>& dense_work)
  {
    allocate_dense_storage();

    jdsimplex_assemble_elliptic_L_int_dag<D,Real>(
      *pre,
      geom,
      plan,
      coeffs,
      dense_work,
      L.data());

    assemble_dense_boundary_maps();

    if (leaf_options.interior_scale_override > Real(0))
    {
      sL = leaf_options.interior_scale_override;
    }
    else
    {
      sL = row_rms_scale(
        L.data(),
        m_int,
        M);
    }
  }

  void set_matrix_free_interior_scale()
  {
    if (leaf_options.interior_scale_override > Real(0))
    {
      sL = leaf_options.interior_scale_override;
      return;
    }

    if (!elliptic_plan_ || !elliptic_action_work_)
    {
      throw std::runtime_error(
        "Leaf: matrix-free scale without action backend");
    }

    std::vector<Real> basis_vector(
      (std::size_t)M,
      Real(0));
    std::vector<Real> image(
      (std::size_t)m_int,
      Real(0));

    long double norm_squared = 0.0L;

    for (int column = 0;
         column < M;
         ++column)
    {
      std::fill(
        basis_vector.begin(),
        basis_vector.end(),
        Real(0));
      basis_vector[(std::size_t)column] =
        Real(1);

      jdsimplex_apply_elliptic<D,Real>(
        *pre,
        geom,
        *elliptic_plan_,
        owned_coeffs_,
        *elliptic_action_work_,
        basis_vector.data(),
        image.data());

      for (Real value : image)
      {
        const long double promoted =
          static_cast<long double>(value);
        norm_squared += promoted * promoted;
      }
    }

    const long double entry_count =
      static_cast<long double>(
        std::max<std::size_t>(
          (std::size_t)m_int * M,
          1));
    const long double rms =
      std::sqrt(norm_squared / entry_count);
    const long double tiny =
      static_cast<long double>(
        std::numeric_limits<Real>::min());

    sL = static_cast<Real>(
      1.0L / std::max(rms, tiny));
  }

  void build_boundary_action_metadata()
  {
    trace_ratio_face_.assign(
      (std::size_t)nface,
      Real(0));
    flux_eta_.assign(
      (std::size_t)nface * D,
      Real(0));

    for (int face = 0;
         face < nface;
         ++face)
    {
      const Real reference_scale =
        face_ref_scale[(std::size_t)face];
      if (!(reference_scale > Real(0)))
      {
        throw std::runtime_error(
          "Leaf: bad reference face scale");
      }

      trace_ratio_face_[(std::size_t)face] =
        face_scale[(std::size_t)face]
        / reference_scale;

      for (int axis = 0;
           axis < D;
           ++axis)
      {
        Real eta = Real(0);
        for (int physical_axis = 0;
             physical_axis < D;
             ++physical_axis)
        {
          const Real normal_component =
            normal_scaled[
              (std::size_t)face * D
              + (std::size_t)physical_axis]
            / reference_scale;

          eta +=
            geom.BinvT[
              (std::size_t)physical_axis
              + (std::size_t)D
                * (std::size_t)axis]
            * normal_component;
        }

        flux_eta_[
          (std::size_t)face * D
          + (std::size_t)axis] = eta;
      }
    }
  }

  void assemble_dense_boundary_maps()
  {
    for (int face = 0;
         face < nface;
         ++face)
    {
      const int sigma =
        face_sigma_index[(std::size_t)face];
      const int row_offset = face * kf;
      const Real trace_scale =
        trace_ratio_face_[(std::size_t)face];

      for (int column = 0;
           column < M;
           ++column)
      {
        for (int row = 0;
             row < kf;
             ++row)
        {
          const std::size_t trace_index =
            (std::size_t)row
            + (std::size_t)kf
              * (
                  (std::size_t)column
                  + (std::size_t)M
                    * (
                        (std::size_t)sigma
                        + (std::size_t)pre->nsigma
                          * (std::size_t)face
                      )
                );

          T[
            (std::size_t)(row_offset + row)
            + (std::size_t)nb
              * (std::size_t)column] =
            trace_scale
            * pre->T_ref[trace_index];

          Real flux_value = Real(0);
          for (int axis = 0;
               axis < D;
               ++axis)
          {
            const std::size_t flux_index =
              (std::size_t)row
              + (std::size_t)kf
                * (
                    (std::size_t)column
                    + (std::size_t)M
                      * (
                          (std::size_t)axis
                          + (std::size_t)D
                            * (
                                (std::size_t)sigma
                                + (std::size_t)pre->nsigma
                                  * (std::size_t)face
                              )
                        )
                  );

            flux_value +=
              flux_eta_[
                (std::size_t)face * D
                + (std::size_t)axis]
              * pre->Fgrad_ref[flux_index];
          }

          F[
            (std::size_t)(row_offset + row)
            + (std::size_t)nb
              * (std::size_t)column] =
            flux_value;
        }
      }
    }
  }

  void build_A_tau()
  {
    if (!uses_dense_backend())
    {
      A_tau.clear();
      return;
    }

    if (tau_rows.size() != (std::size_t)nb
        || sqrt_tau_rows.size() != (std::size_t)nb)
    {
      return;
    }
    if (L.size() != (std::size_t)m_int * M
        || T.size() != (std::size_t)nb * M)
    {
      throw std::runtime_error(
        "Leaf: dense backend storage is incomplete");
    }

    A_tau.assign(
      (std::size_t)ntau_rows * M,
      Real(0));

    for (int column = 0;
         column < M;
         ++column)
    {
      for (int row = 0;
           row < m_int;
           ++row)
      {
        A_tau[
          (std::size_t)row
          + (std::size_t)ntau_rows
            * (std::size_t)column] =
          sL
          * L[
              (std::size_t)row
              + (std::size_t)m_int
                * (std::size_t)column];
      }

      for (int row = 0;
           row < nb;
           ++row)
      {
        A_tau[
          (std::size_t)m_int
          + (std::size_t)row
          + (std::size_t)ntau_rows
            * (std::size_t)column] =
          sqrt_tau_rows[(std::size_t)row]
          * T[
              (std::size_t)row
              + (std::size_t)nb
                * (std::size_t)column];
      }
    }
  }

  void validate_boundary_dense_args(
    const Real* X,
    int ldx,
    int nrhs,
    Real* Y,
    int ldy,
    const char* caller) const
  {
    if (!X || !Y)
    {
      throw std::invalid_argument(
        caller);
    }
    if (nrhs < 0 || ldx < M || ldy < nb)
    {
      throw std::invalid_argument(
        caller);
    }
  }

  void apply_trace_csc_columns(
    const Real* X,
    int ldx,
    int nrhs,
    Real* Y,
    int ldy) const
  {
    for (int rhs = 0;
         rhs < nrhs;
         ++rhs)
    {
      std::fill(
        Y + (std::size_t)ldy * rhs,
        Y + (std::size_t)ldy * rhs + nb,
        Real(0));
    }

    for (int face = 0;
         face < nface;
         ++face)
    {
      const int sigma =
        face_sigma_index[(std::size_t)face];
      const auto& block =
        pre->T_ref_csc_block(
          sigma,
          face);
      const Real scale =
        trace_ratio_face_[(std::size_t)face];
      const int row_offset = face * kf;

      for (int column = 0;
           column < block.cols;
           ++column)
      {
        for (int position =
               block.colptr[(std::size_t)column];
             position <
               block.colptr[
                 (std::size_t)column + 1];
             ++position)
        {
          const int row =
            block.rowind[(std::size_t)position];
          const Real value =
            scale
            * block.values[
                (std::size_t)position];

          for (int rhs = 0;
               rhs < nrhs;
               ++rhs)
          {
            Y[
              (std::size_t)(row_offset + row)
              + (std::size_t)ldy * rhs] +=
              value
              * X[
                  (std::size_t)column
                  + (std::size_t)ldx * rhs];
          }
        }
      }
    }
  }

  void apply_flux_csc_columns(
    const Real* X,
    int ldx,
    int nrhs,
    Real* Y,
    int ldy) const
  {
    for (int rhs = 0;
         rhs < nrhs;
         ++rhs)
    {
      std::fill(
        Y + (std::size_t)ldy * rhs,
        Y + (std::size_t)ldy * rhs + nb,
        Real(0));
    }

    for (int face = 0;
         face < nface;
         ++face)
    {
      const int sigma =
        face_sigma_index[(std::size_t)face];
      const int row_offset = face * kf;

      for (int axis = 0;
           axis < D;
           ++axis)
      {
        const Real scale =
          flux_eta_[
            (std::size_t)face * D
            + (std::size_t)axis];
        if (scale == Real(0))
        {
          continue;
        }

        const auto& block =
          pre->Fgrad_ref_csc_block(
            axis,
            sigma,
            face);

        for (int column = 0;
             column < block.cols;
             ++column)
        {
          for (int position =
                 block.colptr[(std::size_t)column];
               position <
                 block.colptr[
                   (std::size_t)column + 1];
               ++position)
          {
            const int row =
              block.rowind[(std::size_t)position];
            const Real value =
              scale
              * block.values[
                  (std::size_t)position];

            for (int rhs = 0;
                 rhs < nrhs;
                 ++rhs)
            {
              Y[
                (std::size_t)(row_offset + row)
                + (std::size_t)ldy * rhs] +=
                value
                * X[
                    (std::size_t)column
                    + (std::size_t)ldx * rhs];
            }
          }
        }
      }
    }
  }

  void apply_trace_csc_transpose(
    const Real* Y,
    Real* X,
    bool add) const
  {
    if (!add)
    {
      std::fill(
        X,
        X + M,
        Real(0));
    }

    for (int face = 0;
         face < nface;
         ++face)
    {
      const int sigma =
        face_sigma_index[(std::size_t)face];
      const auto& block =
        pre->T_ref_csc_block(
          sigma,
          face);
      const Real scale =
        trace_ratio_face_[(std::size_t)face];
      const int row_offset = face * kf;

      for (int column = 0;
           column < block.cols;
           ++column)
      {
        Real sum = Real(0);

        for (int position =
               block.colptr[(std::size_t)column];
             position <
               block.colptr[
                 (std::size_t)column + 1];
             ++position)
        {
          const int row =
            block.rowind[(std::size_t)position];
          sum +=
            scale
            * block.values[
                (std::size_t)position]
            * Y[row_offset + row];
        }

        X[column] += sum;
      }
    }
  }

  void apply_interior_operator(
    const Real* x,
    Real* y) const
  {
    if (operator_mode == LeafOperatorMode::Dense)
    {
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        m_int,
        1,
        M,
        Real(1),
        L.data(),
        m_int,
        x,
        M,
        Real(0),
        y,
        m_int);
      return;
    }

    if (!elliptic_plan_
        || !elliptic_action_work_)
    {
      throw std::runtime_error(
        "Leaf: missing matrix-free elliptic backend");
    }

    jdsimplex_apply_elliptic<D,Real>(
      *pre,
      geom,
      *elliptic_plan_,
      owned_coeffs_,
      *elliptic_action_work_,
      x,
      y);
  }

  void apply_stacked_matrix_free(
    const Real* x,
    Real* y) const
  {
    apply_interior_operator(
      x,
      y);

    for (int row = 0;
         row < m_int;
         ++row)
    {
      y[row] *= sL;
    }

    apply_trace_csc_columns(
      x,
      M,
      1,
      y + m_int,
      nb);

    for (int row = 0;
         row < nb;
         ++row)
    {
      y[m_int + row] *=
        sqrt_tau_rows[(std::size_t)row];
    }
  }

  void apply_stacked_matrix_free_transpose(
    const Real* y,
    Real* x) const
  {
    for (int row = 0;
         row < m_int;
         ++row)
    {
      action_interior_scaled_[(std::size_t)row] =
        sL * y[row];
    }

    jdsimplex_apply_elliptic_transpose<D,Real>(
      *pre,
      geom,
      *elliptic_plan_,
      owned_coeffs_,
      *elliptic_action_work_,
      action_interior_scaled_.data(),
      x);

    for (int row = 0;
         row < nb;
         ++row)
    {
      action_boundary_scaled_[(std::size_t)row] =
        sqrt_tau_rows[(std::size_t)row]
        * y[m_int + row];
    }

    apply_trace_csc_transpose(
      action_boundary_scaled_.data(),
      x,
      true);
  }

  int solve_matrix_free(
    const Real* rhs,
    Real* x_out,
    LsmrInfo* info) const
  {
    if (!rhs || !x_out || !info)
    {
      return -2;
    }
    if (!uses_matrix_free_backend()
        || !elliptic_plan_
        || !elliptic_action_work_)
    {
      return -5;
    }

    const int row_count = ntau_rows;
    const int column_count = M;

    std::vector<double> rhs_double(
      (std::size_t)row_count,
      0.0);
    std::vector<double> u(
      (std::size_t)row_count,
      0.0);
    std::vector<double> v(
      (std::size_t)column_count,
      0.0);
    std::vector<double> x(
      (std::size_t)column_count,
      0.0);

    for (int row = 0;
         row < row_count;
         ++row)
    {
      rhs_double[(std::size_t)row] =
        static_cast<double>(rhs[row]);
    }

    const auto options_double =
      detail::to_double_options(
        lsmr_options);

    const double atol = options_double.atol;
    const double btol = options_double.btol;
    const double conlim = options_double.conlim;
    const int itnlim = options_double.itnlim;
    const int nout = options_double.nout;
    const int localsize = options_double.localsize;
    const int ctest = options_double.ctest;
    const double damp = options_double.damp;

    detail::lsmr_c_set_options(
      &atol,
      &btol,
      &conlim,
      &itnlim,
      &nout,
      &localsize,
      &ctest);

    int action = 0;
    int istop = 0;
    int iteration = 0;
    int stat = 0;
    double normr = 0.0;
    double normA = 0.0;
    double condA = 0.0;
    double normb = 0.0;
    double normx = 0.0;
    double normAr = 0.0;

    bool initialized = false;

    try
    {
      while (true)
      {
        detail::lsmr_c_step(
          &row_count,
          &column_count,
          &action,
          u.data(),
          v.data(),
          rhs_double.data(),
          &damp,
          x.data(),
          &istop,
          &iteration,
          &stat,
          &normr,
          &normA,
          &condA,
          &normb,
          &normx,
          &normAr);

        initialized = true;

        if (action == 0)
        {
          break;
        }

        if (action == 1)
        {
          for (int row = 0;
               row < row_count;
               ++row)
          {
            action_row_input_[(std::size_t)row] =
              static_cast<Real>(
                u[(std::size_t)row]);
          }

          apply_stacked_matrix_free_transpose(
            action_row_input_.data(),
            action_coeff_output_.data());

          for (int column = 0;
               column < column_count;
               ++column)
          {
            v[(std::size_t)column] +=
              static_cast<double>(
                action_coeff_output_[
                  (std::size_t)column]);
          }
        }
        else if (action == 2)
        {
          for (int column = 0;
               column < column_count;
               ++column)
          {
            action_coeff_input_[
              (std::size_t)column] =
              static_cast<Real>(
                v[(std::size_t)column]);
          }

          apply_stacked_matrix_free(
            action_coeff_input_.data(),
            action_row_output_.data());

          for (int row = 0;
               row < row_count;
               ++row)
          {
            u[(std::size_t)row] +=
              static_cast<double>(
                action_row_output_[
                  (std::size_t)row]);
          }
        }
        else
        {
          detail::cleanup_lsmr_state(
            row_count,
            column_count,
            u.data(),
            v.data(),
            rhs_double.data(),
            damp,
            x.data(),
            istop,
            iteration,
            stat,
            normr,
            normA,
            condA,
            normb,
            normx,
            normAr);
          return -4;
        }
      }
    }
    catch (...)
    {
      if (initialized)
      {
        detail::cleanup_lsmr_state(
          row_count,
          column_count,
          u.data(),
          v.data(),
          rhs_double.data(),
          damp,
          x.data(),
          istop,
          iteration,
          stat,
          normr,
          normA,
          condA,
          normb,
          normx,
          normAr);
      }
      throw;
    }

    detail::cleanup_lsmr_state(
      row_count,
      column_count,
      u.data(),
      v.data(),
      rhs_double.data(),
      damp,
      x.data(),
      istop,
      iteration,
      stat,
      normr,
      normA,
      condA,
      normb,
      normx,
      normAr);

    for (int column = 0;
         column < column_count;
         ++column)
    {
      x_out[column] =
        static_cast<Real>(
          x[(std::size_t)column]);
    }

    detail::LsmrInfo<double> double_info;
    double_info.istop = istop;
    double_info.itn = iteration;
    double_info.stat = stat;
    double_info.normr = normr;
    double_info.normA = normA;
    double_info.condA = condA;
    double_info.normb = normb;
    double_info.normx = normx;
    double_info.normAr = normAr;
    detail::copy_info_from_double(
      double_info,
      info);

    return 0;
  }

  void verify_solution_against_dense(
    const Real* rhs,
    const Real* matrix_free_solution) const
  {
    std::vector<Real> dense_solution(
      (std::size_t)M,
      Real(0));
    LsmrInfo dense_info{};

    const int return_code =
      lsmr_dense_solve_colmajor<Real>(
        ntau_rows,
        M,
        A_tau.data(),
        rhs,
        dense_solution.data(),
        lsmr_options,
        &dense_info);

    if (return_code != 0)
    {
      throw std::runtime_error(
        "Leaf Verify: dense fallback solve failed");
    }

    const Real error =
      relative_vector_error(
        matrix_free_solution,
        dense_solution.data(),
        M);

    if (error > leaf_options.verify_tolerance)
    {
      throw std::runtime_error(
        "Leaf Verify: dense/matrix-free solution mismatch");
    }
  }

  void verify_backends() const
  {
    if (operator_mode != LeafOperatorMode::Verify)
    {
      return;
    }
    if (!has_dense_local_operator()
        || !elliptic_action_work_)
    {
      throw std::runtime_error(
        "Leaf Verify: incomplete backend");
    }

    std::vector<Real> x(
      (std::size_t)M,
      Real(0));
    std::vector<Real> y(
      (std::size_t)ntau_rows,
      Real(0));
    std::vector<Real> dense_forward(
      (std::size_t)ntau_rows,
      Real(0));
    std::vector<Real> action_forward(
      (std::size_t)ntau_rows,
      Real(0));
    std::vector<Real> dense_transpose(
      (std::size_t)M,
      Real(0));
    std::vector<Real> action_transpose(
      (std::size_t)M,
      Real(0));

    for (int column = 0;
         column < M;
         ++column)
    {
      x[(std::size_t)column] =
        std::sin(
          Real(0.37)
          * Real(column + 1));
    }
    for (int row = 0;
         row < ntau_rows;
         ++row)
    {
      y[(std::size_t)row] =
        std::cos(
          Real(0.23)
          * Real(row + 1));
    }

    dense_apply(
      A_tau.data(),
      ntau_rows,
      M,
      x.data(),
      dense_forward.data());
    apply_stacked_matrix_free(
      x.data(),
      action_forward.data());

    dense_apply_transpose(
      A_tau.data(),
      ntau_rows,
      M,
      y.data(),
      dense_transpose.data());
    apply_stacked_matrix_free_transpose(
      y.data(),
      action_transpose.data());

    const Real forward_error =
      relative_vector_error(
        action_forward.data(),
        dense_forward.data(),
        ntau_rows);
    const Real transpose_error =
      relative_vector_error(
        action_transpose.data(),
        dense_transpose.data(),
        M);

    std::vector<Real> dense_flux(
      (std::size_t)nb,
      Real(0));
    std::vector<Real> sparse_flux(
      (std::size_t)nb,
      Real(0));

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      nb,
      1,
      M,
      Real(1),
      F.data(),
      nb,
      x.data(),
      M,
      Real(0),
      dense_flux.data(),
      nb);
    apply_flux_csc_columns(
      x.data(),
      M,
      1,
      sparse_flux.data(),
      nb);

    const Real flux_error =
      relative_vector_error(
        sparse_flux.data(),
        dense_flux.data(),
        nb);

    const Real maximum_error =
      std::max(
        forward_error,
        std::max(
          transpose_error,
          flux_error));

    if (maximum_error >
        leaf_options.verify_tolerance)
    {
      throw std::runtime_error(
        "Leaf Verify: dense/matrix-free action mismatch");
    }
  }

  static void dense_apply(
    const Real* matrix,
    int rows,
    int cols,
    const Real* x,
    Real* y)
  {
    std::fill(
      y,
      y + rows,
      Real(0));

    for (int column = 0;
         column < cols;
         ++column)
    {
      const Real x_column = x[column];
      const Real* matrix_column =
        matrix
        + (std::size_t)rows * column;

      for (int row = 0;
           row < rows;
           ++row)
      {
        y[row] +=
          matrix_column[row] * x_column;
      }
    }
  }

  static void dense_apply_transpose(
    const Real* matrix,
    int rows,
    int cols,
    const Real* y,
    Real* x)
  {
    for (int column = 0;
         column < cols;
         ++column)
    {
      const Real* matrix_column =
        matrix
        + (std::size_t)rows * column;
      Real sum = Real(0);

      for (int row = 0;
           row < rows;
           ++row)
      {
        sum +=
          matrix_column[row] * y[row];
      }

      x[column] = sum;
    }
  }

  static Real relative_vector_error(
    const Real* candidate,
    const Real* reference,
    int count)
  {
    long double numerator = 0.0L;
    long double denominator = 0.0L;

    for (int index = 0;
         index < count;
         ++index)
    {
      const long double difference =
        static_cast<long double>(
          candidate[index])
        - static_cast<long double>(
            reference[index]);
      const long double value =
        static_cast<long double>(
          reference[index]);

      numerator += difference * difference;
      denominator += value * value;
    }

    return static_cast<Real>(
      std::sqrt(
        numerator
        / std::max(
            denominator,
            1.0e-300L)));
  }

  static Real row_rms_scale(
    const Real* matrix,
    int rows,
    int cols)
  {
    long double norm_squared = 0.0L;
    const std::size_t entry_count =
      (std::size_t)rows * cols;

    for (std::size_t index = 0;
         index < entry_count;
         ++index)
    {
      const long double value =
        static_cast<long double>(
          matrix[index]);
      norm_squared += value * value;
    }

    const long double rms =
      std::sqrt(
        norm_squared
        / static_cast<long double>(
            std::max<std::size_t>(
              entry_count,
              1)));

    return static_cast<Real>(
      1.0L
      / std::max(
          rms,
          static_cast<long double>(
            std::numeric_limits<Real>::min())));
  }
};

} // namespace jsimplex

#endif // JLEAF_HH
