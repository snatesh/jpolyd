#ifndef JLEAF_HH
#define JLEAF_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
#ifdef TIMING
#include <timer.hh>
#endif

namespace jsimplex {

enum class LeafOperatorMode
{
  MatrixFree = 0,
  Dense = 1,
  Verify = 2,
  DenseSparse = 3
};

enum class LeafLeastSquaresSolver
{
  Auto = 0,
  LSMR = 1,
  DenseQR = 2
};

template<class Real>
struct LeafOptions
{
  // Dense preserves every legacy constructor/reset call and every public
  // dense diagnostic field. DenseSparse stores the composed dense interior
  // matrix L, applies trace/flux through the reference CSC blocks, and uses
  // reverse-communication LSMR without materializing T, F, or A_tau.
  // MatrixFree stores neither L, T, F, nor A_tau. Verify builds both Dense
  // and MatrixFree backends, solves through MatrixFree, and compares against
  // Dense.
  LeafOperatorMode operator_mode = LeafOperatorMode::Dense;

  // Auto selects DenseQR for the fully dense operator mode and LSMR for all
  // action-based modes. DenseQR is valid only when A_tau is materialized.
  LeafLeastSquaresSolver least_squares_solver =
    LeafLeastSquaresSolver::Auto;

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
  using LeastSquaresSolver = LeafLeastSquaresSolver;

  /*
    Caller-owned storage for repeated local LSMR solves.

    One workspace may be reused for every column of a leaf solution map.
    It is not safe to use one workspace concurrently from multiple calls.
    Independent leaves and independent workspaces remain thread-independent.
  */
  struct SolveWorkspace
  {
    int row_count = 0;
    int column_count = 0;

    // Stacked right-hand side in Real precision.
    std::vector<Real> rhs;

    // Dense QR scratch. LAPACK stores Householder vectors and R in qr_factor.
    // qr_panel is column-major with leading dimension row_count and is reused
    // for the complete [boundary-response, source-response] RHS panel.
    std::vector<Real> qr_factor;
    std::vector<Real> qr_tau;
    std::vector<Real> qr_panel;
    std::vector<Real> qr_work;
    lapack_int geqrf_lwork = 0;
    lapack_int ormqr_lwork = 0;
    int ormqr_nrhs = -1;
    const Leaf* qr_owner = nullptr;
    bool qr_factorized = false;

    // Reverse-communication LSMR arrays. The Fortran implementation is
    // double precision, matching the existing matrix-free solve path.
    std::vector<double> rhs_double;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> x;

    // Reused only by Verify mode.
    std::vector<Real> verify_solution;

    SolveWorkspace() = default;

    explicit SolveWorkspace(const Leaf& leaf)
    {
      reset(leaf);
    }

    SolveWorkspace(int rows, int columns)
    {
      reset(rows, columns);
    }

    void reset(const Leaf& leaf)
    {
      reset(leaf.ntau_rows, leaf.M);
    }

    void reset(int rows, int columns)
    {
      if (rows < 0 || columns < 0)
      {
        throw std::invalid_argument(
          "Leaf::SolveWorkspace: negative dimensions");
      }

      row_count = rows;
      column_count = columns;

      rhs.assign((std::size_t)rows, Real(0));
      qr_factor.clear();
      qr_tau.clear();
      qr_panel.clear();
      qr_work.clear();
      geqrf_lwork = 0;
      ormqr_lwork = 0;
      ormqr_nrhs = -1;
      qr_owner = nullptr;
      qr_factorized = false;
      rhs_double.assign((std::size_t)rows, 0.0);
      u.assign((std::size_t)rows, 0.0);
      v.assign((std::size_t)columns, 0.0);
      x.assign((std::size_t)columns, 0.0);
      verify_solution.assign(
        (std::size_t)columns,
        Real(0));
    }

    bool compatible(const Leaf& leaf) const
    {
      return row_count == leaf.ntau_rows
          && column_count == leaf.M
          && rhs.size() == (std::size_t)row_count
          && rhs_double.size() == (std::size_t)row_count
          && u.size() == (std::size_t)row_count
          && v.size() == (std::size_t)column_count
          && x.size() == (std::size_t)column_count
          && verify_solution.size()
               == (std::size_t)column_count;
    }
  };

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

  // Dense diagnostics/fallback. L is also populated in DenseSparse mode;
  // T, F, and A_tau are populated only in Dense and Verify mode.
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

  // tau_C_base is the user-facing stabilization constant.  Elliptic leaves
  // scale it by mR/m2 so that changing the residual row space does not silently
  // change the aggregate interior-versus-boundary least-squares balance.
  // Poisson leaves use factor one because their residual space is Pi_{n-2}.
  Real tau_C_base = Real(1);
  Real tau_residual_row_factor = Real(1);
  Real tau_C = Real(1); // effective constant used in tau_f=C(n+1)^2/h_f
  Real sL = Real(1);

#ifdef TIMING
  // Time spent computing the interior row-RMS scale. In MatrixFree mode this
  // includes the exact identity-column action sweep. A user override reports
  // zero because no scale computation is performed.
  double timing_interior_scale_seconds = 0.0;
  EllipticActionTimings timing_interior_scale_actions{};
#endif

  LsmrOptions lsmr_options{};
  Options leaf_options{};
  LeafOperatorMode operator_mode = LeafOperatorMode::Dense;
  LeafLeastSquaresSolver least_squares_solver =
    LeafLeastSquaresSolver::LSMR;

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
    Real tau_C_base_in = Real(1),
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
      tau_C_base_in,
      opts,
      leaf_opts);
  }

  // New mode-aware elliptic constructor. MatrixFree requires no dense
  // workspace. Dense, DenseSparse, and Verify create a temporary dense
  // workspace internally.
  Leaf(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    Real tau_C_base_in,
    const LsmrOptions& opts,
    const Options& leaf_opts)
  {
    reset(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      elliptic_plan,
      coeffs,
      tau_C_base_in,
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
      pre_in.m_int,
      tau_C_in,
      Real(1),
      opts,
      leaf_opts);

    EllipticDegreeSpec degree_spec;
    degree_spec.p2 = 0;
    degree_spec.p1 = -1;
    degree_spec.p0 = -1;

    owned_elliptic_plan_ =
      std::make_unique<EllipticPlan<D,Real>>(
        *pre,
        degree_spec,
        true,
        EllipticResidualPolicy::SecondDerivativeDegree,
        EllipticMultiplicationAssembler::ClenshawColumns,
        true);
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

    if (uses_clenshaw_backend())
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
    else if (operator_mode == LeafOperatorMode::DenseSparse)
    {
      EllipticDenseWorkspace<D,Real> dense_work(
        *elliptic_plan_);
      assemble_dense_sparse_backend(
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
    Real tau_C_base_in = Real(1),
    const LsmrOptions& opts = LsmrOptions(),
    const Options& leaf_opts = Options())
  {
    initialize_common(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      elliptic_plan.mR,
      tau_C_base_in,
      elliptic_tau_residual_row_factor(elliptic_plan),
      opts,
      leaf_opts);

    elliptic_plan_ = &elliptic_plan;

    if (uses_clenshaw_backend())
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
    else if (operator_mode == LeafOperatorMode::DenseSparse)
    {
      assemble_dense_sparse_backend(
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

  // New reset overload for mode-aware callers that should not allocate a dense
  // workspace unless the selected mode requires it.
  void reset(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    const EllipticPlan<D,Real>& elliptic_plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    Real tau_C_base_in,
    const LsmrOptions& opts,
    const Options& leaf_opts)
  {
    initialize_common(
      pre_in,
      V_phys_colmajor,
      global_vids_in,
      elliptic_plan.mR,
      tau_C_base_in,
      elliptic_tau_residual_row_factor(elliptic_plan),
      opts,
      leaf_opts);

    elliptic_plan_ = &elliptic_plan;

    if (uses_clenshaw_backend())
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
    else if (operator_mode == LeafOperatorMode::DenseSparse)
    {
      EllipticDenseWorkspace<D,Real> dense_work(
        elliptic_plan);
      assemble_dense_sparse_backend(
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

  bool uses_clenshaw_backend() const
  {
    return operator_mode == LeafOperatorMode::MatrixFree
        || operator_mode == LeafOperatorMode::Verify;
  }

  bool uses_dense_backend() const
  {
    return operator_mode == LeafOperatorMode::Dense
        || operator_mode == LeafOperatorMode::Verify;
  }

  bool uses_action_solver() const
  {
    return operator_mode != LeafOperatorMode::Dense;
  }

  bool has_dense_interior_operator() const
  {
    return L.size() == (std::size_t)m_int * M;
  }

  bool has_dense_local_operator() const
  {
    return !L.empty() && !A_tau.empty();
  }

#ifdef TIMING
  EllipticActionTimings elliptic_action_timings() const
  {
    return elliptic_action_work_ ? elliptic_action_work_->timings() : EllipticActionTimings{};
  }

  void reset_elliptic_action_timings() const
  {
    if (elliptic_action_work_) elliptic_action_work_->reset_timings();
  }
#endif

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

  /*
    Solve only for the volume coefficients. No trace, flux, mismatch, or PDE
    residual is evaluated here. Reusing SolveWorkspace removes the repeated
    vector allocations from homogeneous-map column construction.
  */
  LsmrInfo solve_coefficients(
    const Real* lambda,
    const Real* f_int,
    Real* c_out,
    SolveWorkspace& workspace) const
  {
    validate_coefficient_solve_inputs(
      lambda,
      f_int,
      c_out,
      false,
      "Leaf::solve_coefficients");

    ensure_solve_workspace(workspace);
    build_stacked_rhs(
      lambda,
      f_int,
      false,
      workspace.rhs.data());

    return solve_coefficients_from_rhs(
      workspace.rhs.data(),
      c_out,
      workspace,
      "Leaf::solve_coefficients");
  }

  /*
    Homogeneous-source specialization used for every Ulam column. It avoids
    allocating or clearing a separate m_int-vector of zeros.
  */
  LsmrInfo solve_coefficients_zero_source(
    const Real* lambda,
    Real* c_out,
    SolveWorkspace& workspace) const
  {
    validate_coefficient_solve_inputs(
      lambda,
      nullptr,
      c_out,
      true,
      "Leaf::solve_coefficients_zero_source");

    ensure_solve_workspace(workspace);
    build_stacked_rhs(
      lambda,
      nullptr,
      true,
      workspace.rhs.data());

    return solve_coefficients_from_rhs(
      workspace.rhs.data(),
      c_out,
      workspace,
      "Leaf::solve_coefficients_zero_source");
  }

  /*
    Convenience overloads preserve a simple one-shot interface. Repeated
    callers should provide SolveWorkspace explicitly.
  */
  LsmrInfo solve_coefficients(
    const Real* lambda,
    const Real* f_int,
    Real* c_out) const
  {
    SolveWorkspace workspace(*this);
    return solve_coefficients(
      lambda,
      f_int,
      c_out,
      workspace);
  }

  LsmrInfo solve_coefficients_zero_source(
    const Real* lambda,
    Real* c_out) const
  {
    SolveWorkspace workspace(*this);
    return solve_coefficients_zero_source(
      lambda,
      c_out,
      workspace);
  }

  /*
    Materialize the complete reusable local inverse action

      [ U_lambda  U_f ] = A_tau^dagger [ D_lambda  D_f ],

    where D_lambda has sqrt(tau) on the boundary block and D_f has -sL on
    the interior block. response_out is M x (nb + m_int), column-major.

    DenseQR factors A_tau once and solves the full panel with one Q^T apply
    and one triangular solve. LSMR retains the historical columnwise solve
    path, but produces the same complete reusable map.
  */
  void solve_response_maps(
    Real* response_out,
    int ldresponse,
    SolveWorkspace& workspace,
    long long* lsmr_iterations_out = nullptr) const
  {
    if (!response_out)
    {
      throw std::invalid_argument(
        "Leaf::solve_response_maps: null output");
    }
    if (ldresponse < std::max(M, 1))
    {
      throw std::invalid_argument(
        "Leaf::solve_response_maps: invalid leading dimension");
    }

    ensure_solve_workspace(workspace);
    if (lsmr_iterations_out)
    {
      *lsmr_iterations_out = 0;
    }

    if (least_squares_solver ==
        LeafLeastSquaresSolver::DenseQR)
    {
      solve_response_maps_dense_qr(
        response_out,
        ldresponse,
        workspace);
      return;
    }

    const int nrhs = nb + m_int;
    for (int column = 0;
         column < nrhs;
         ++column)
    {
      std::fill(
        workspace.rhs.begin(),
        workspace.rhs.end(),
        Real(0));

      if (column < nb)
      {
        workspace.rhs[(std::size_t)m_int + column] =
          sqrt_tau_rows[(std::size_t)column];
      }
      else
      {
        const int source_column = column - nb;
        workspace.rhs[(std::size_t)source_column] = -sL;
      }

      const LsmrInfo info = solve_coefficients_from_rhs(
        workspace.rhs.data(),
        response_out
          + (std::size_t)ldresponse
            * (std::size_t)column,
        workspace,
        "Leaf::solve_response_maps");

      if (lsmr_iterations_out)
      {
        *lsmr_iterations_out +=
          static_cast<long long>(
            std::max(info.itn, 0));
      }
    }
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
    if (!trace_out || !raw_flux_out || !aug_flux_out)
    {
      throw std::invalid_argument(
        "Leaf::apply: null boundary output");
    }

    SolveWorkspace workspace(*this);
    const LsmrInfo info = solve_coefficients(
      lambda,
      f_int,
      c_out,
      workspace);

    finish_solution_outputs(
      lambda,
      f_int,
      false,
      c_out,
      trace_out,
      raw_flux_out,
      aug_flux_out,
      trace_mismatch_out,
      pde_residual_out);

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
    if (!trace_out || !raw_flux_out || !aug_flux_out)
    {
      throw std::invalid_argument(
        "Leaf::apply_zero_source: null boundary output");
    }

    SolveWorkspace workspace(*this);
    const LsmrInfo info =
      solve_coefficients_zero_source(
        lambda,
        c_out,
        workspace);

    finish_solution_outputs(
      lambda,
      nullptr,
      true,
      c_out,
      trace_out,
      raw_flux_out,
      aug_flux_out,
      trace_mismatch_out,
      pde_residual_out);

    return info;
  }

private:
  void ensure_solve_workspace(
    SolveWorkspace& workspace) const
  {
    if (!workspace.compatible(*this))
    {
      workspace.reset(*this);
    }
  }

  void validate_coefficient_solve_inputs(
    const Real* lambda,
    const Real* f_int,
    const Real* c_out,
    bool zero_source,
    const char* caller) const
  {
    if (!lambda)
    {
      throw std::invalid_argument(
        std::string(caller)
        + ": null lambda");
    }
    if (!c_out)
    {
      throw std::invalid_argument(
        std::string(caller)
        + ": null c_out");
    }
    if (!zero_source
        && m_int > 0
        && !f_int)
    {
      throw std::invalid_argument(
        std::string(caller)
        + ": null f_int");
    }
  }

  void build_stacked_rhs(
    const Real* lambda,
    const Real* f_int,
    bool zero_source,
    Real* rhs) const
  {
    for (int row = 0;
         row < m_int;
         ++row)
    {
      rhs[(std::size_t)row] =
        zero_source
        ? Real(0)
        : -sL * f_int[row];
    }

    for (int row = 0;
         row < nb;
         ++row)
    {
      rhs[(std::size_t)m_int + row] =
        sqrt_tau_rows[(std::size_t)row]
        * lambda[row];
    }
  }

  void factor_dense_qr(
    SolveWorkspace& workspace) const
  {
    if (operator_mode != LeafOperatorMode::Dense
        || A_tau.size()
             != (std::size_t)ntau_rows * (std::size_t)M)
    {
      throw std::runtime_error(
        "Leaf::factor_dense_qr: dense A_tau is unavailable");
    }

    if (workspace.qr_factorized
        && workspace.qr_owner == this)
    {
      return;
    }

    workspace.qr_factor = A_tau;
    workspace.qr_tau.assign(
      (std::size_t)M,
      Real(0));

    lapack_int info = 0;
    if (workspace.geqrf_lwork <= 0)
    {
      Real query = Real(0);
      info = detail::LapackGeqrf<Real>::run(
        (lapack_int)ntau_rows,
        (lapack_int)M,
        workspace.qr_factor.data(),
        (lapack_int)ntau_rows,
        workspace.qr_tau.data(),
        &query,
        (lapack_int)-1);
      if (info != 0)
      {
        throw std::runtime_error(
          "Leaf::factor_dense_qr: GEQRF workspace query failed");
      }

      workspace.geqrf_lwork = std::max<lapack_int>(
        1,
        static_cast<lapack_int>(std::ceil(query)));
    }
    if (workspace.qr_work.size()
        < (std::size_t)workspace.geqrf_lwork)
    {
      workspace.qr_work.resize(
        (std::size_t)workspace.geqrf_lwork);
    }

    info = detail::LapackGeqrf<Real>::run(
      (lapack_int)ntau_rows,
      (lapack_int)M,
      workspace.qr_factor.data(),
      (lapack_int)ntau_rows,
      workspace.qr_tau.data(),
      workspace.qr_work.data(),
      workspace.geqrf_lwork);
    if (info != 0)
    {
      throw std::runtime_error(
        "Leaf::factor_dense_qr: GEQRF factorization failed");
    }

    Real max_diagonal = Real(0);
    for (int i = 0; i < M; ++i)
    {
      max_diagonal = std::max(
        max_diagonal,
        std::abs(
          workspace.qr_factor[
            (std::size_t)i
            + (std::size_t)ntau_rows
              * (std::size_t)i]));
    }
    const Real rank_tolerance =
      std::numeric_limits<Real>::epsilon()
      * Real(std::max(ntau_rows, M))
      * std::max(max_diagonal, Real(1));
    for (int i = 0; i < M; ++i)
    {
      const Real diagonal = std::abs(
        workspace.qr_factor[
          (std::size_t)i
          + (std::size_t)ntau_rows
            * (std::size_t)i]);
      if (!std::isfinite(diagonal)
          || diagonal <= rank_tolerance)
      {
        throw std::runtime_error(
          "Leaf::factor_dense_qr: A_tau is numerically rank deficient");
      }
    }

    workspace.qr_owner = this;
    workspace.qr_factorized = true;
  }

  void apply_dense_qr_pseudoinverse_panel(
    Real* panel,
    int nrhs,
    SolveWorkspace& workspace) const
  {
    if (!panel || nrhs < 0)
    {
      throw std::invalid_argument(
        "Leaf::apply_dense_qr_pseudoinverse_panel: invalid panel");
    }
    if (nrhs == 0)
    {
      return;
    }

    factor_dense_qr(workspace);

    lapack_int info = 0;
    if (workspace.ormqr_lwork <= 0
        || workspace.ormqr_nrhs != nrhs)
    {
      Real query = Real(0);
      info = detail::LapackOrmqr<Real>::run(
        'L',
        'T',
        (lapack_int)ntau_rows,
        (lapack_int)nrhs,
        (lapack_int)M,
        workspace.qr_factor.data(),
        (lapack_int)ntau_rows,
        workspace.qr_tau.data(),
        panel,
        (lapack_int)ntau_rows,
        &query,
        (lapack_int)-1);
      if (info != 0)
      {
        throw std::runtime_error(
          "Leaf::apply_dense_qr_pseudoinverse_panel: ORMQR workspace query failed");
      }

      workspace.ormqr_lwork = std::max<lapack_int>(
        1,
        static_cast<lapack_int>(std::ceil(query)));
      workspace.ormqr_nrhs = nrhs;
    }
    if (workspace.qr_work.size()
        < (std::size_t)workspace.ormqr_lwork)
    {
      workspace.qr_work.resize(
        (std::size_t)workspace.ormqr_lwork);
    }

    info = detail::LapackOrmqr<Real>::run(
      'L',
      'T',
      (lapack_int)ntau_rows,
      (lapack_int)nrhs,
      (lapack_int)M,
      workspace.qr_factor.data(),
      (lapack_int)ntau_rows,
      workspace.qr_tau.data(),
      panel,
      (lapack_int)ntau_rows,
      workspace.qr_work.data(),
      workspace.ormqr_lwork);
    if (info != 0)
    {
      throw std::runtime_error(
        "Leaf::apply_dense_qr_pseudoinverse_panel: ORMQR failed");
    }

    detail::BlasTrsm<Real>::run(
      CblasColMajor,
      CblasLeft,
      CblasUpper,
      CblasNoTrans,
      CblasNonUnit,
      M,
      nrhs,
      Real(1),
      workspace.qr_factor.data(),
      ntau_rows,
      panel,
      ntau_rows);
  }

  void solve_response_maps_dense_qr(
    Real* response_out,
    int ldresponse,
    SolveWorkspace& workspace) const
  {
    if (least_squares_solver !=
        LeafLeastSquaresSolver::DenseQR)
    {
      throw std::runtime_error(
        "Leaf::solve_response_maps_dense_qr: QR solver is not active");
    }

    const int nrhs = nb + m_int;
    workspace.qr_panel.assign(
      (std::size_t)ntau_rows * (std::size_t)nrhs,
      Real(0));

    for (int column = 0; column < nb; ++column)
    {
      workspace.qr_panel[
        (std::size_t)m_int
        + (std::size_t)column
        + (std::size_t)ntau_rows
          * (std::size_t)column] =
        sqrt_tau_rows[(std::size_t)column];
    }
    for (int column = 0; column < m_int; ++column)
    {
      workspace.qr_panel[
        (std::size_t)column
        + (std::size_t)ntau_rows
          * (std::size_t)(nb + column)] = -sL;
    }

    apply_dense_qr_pseudoinverse_panel(
      workspace.qr_panel.data(),
      nrhs,
      workspace);

    for (int column = 0;
         column < nrhs;
         ++column)
    {
      std::copy_n(
        workspace.qr_panel.data()
          + (std::size_t)ntau_rows
            * (std::size_t)column,
        M,
        response_out
          + (std::size_t)ldresponse
            * (std::size_t)column);
    }
  }

  void solve_dense_qr_rhs(
    const Real* rhs,
    Real* c_out,
    SolveWorkspace& workspace) const
  {
    workspace.qr_panel.assign(
      (std::size_t)ntau_rows,
      Real(0));
    std::copy_n(
      rhs,
      ntau_rows,
      workspace.qr_panel.data());

    apply_dense_qr_pseudoinverse_panel(
      workspace.qr_panel.data(),
      1,
      workspace);
    std::copy_n(
      workspace.qr_panel.data(),
      M,
      c_out);
  }

  LsmrInfo solve_coefficients_from_rhs(
    const Real* rhs,
    Real* c_out,
    SolveWorkspace& workspace,
    const char* caller) const
  {
    LsmrInfo info{};
    int return_code = 0;

    if (least_squares_solver ==
        LeafLeastSquaresSolver::DenseQR)
    {
      solve_dense_qr_rhs(
        rhs,
        c_out,
        workspace);
      info.istop = 0;
      info.itn = 0;
      info.stat = 0;
      return_code = 0;
    }
    else if (operator_mode == LeafOperatorMode::Dense)
    {
      return_code =
        lsmr_dense_solve_colmajor<Real>(
          ntau_rows,
          M,
          A_tau.data(),
          rhs,
          c_out,
          lsmr_options,
          &info);
    }
    else
    {
      return_code = solve_action(
        rhs,
        c_out,
        workspace,
        &info);
    }

    if (return_code != 0)
    {
      throw std::runtime_error(
        std::string(caller)
        + ": local least-squares solve failed");
    }

    if (operator_mode == LeafOperatorMode::Verify
        && leaf_options.verify_each_solve)
    {
      verify_solution_against_dense(
        rhs,
        c_out,
        workspace);
    }

    return info;
  }

  void finish_solution_outputs(
    const Real* lambda,
    const Real* f_int,
    bool zero_source,
    const Real* c_out,
    Real* trace_out,
    Real* raw_flux_out,
    Real* aug_flux_out,
    Real* trace_mismatch_out,
    Real* pde_residual_out) const
  {
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

    for (int row = 0;
         row < nb;
         ++row)
    {
      const Real mismatch =
        trace_out[row] - lambda[row];

      if (trace_mismatch_out)
      {
        trace_mismatch_out[row] =
          mismatch;
      }

      aug_flux_out[row] =
        raw_flux_out[row]
        + tau_rows[(std::size_t)row]
          * mismatch;
    }

    if (pde_residual_out)
    {
      apply_interior_operator(
        c_out,
        pde_residual_out);

      if (!zero_source)
      {
        for (int row = 0;
             row < m_int;
             ++row)
        {
          pde_residual_out[row] +=
            f_int[row];
        }
      }
    }
  }

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

  static Real elliptic_tau_residual_row_factor(
    const EllipticPlan<D,Real>& plan)
  {
    if (plan.m2 <= 0 || plan.mR <= 0)
    {
      throw std::invalid_argument(
        "Leaf: invalid elliptic residual dimensions for tau scaling");
    }

    return static_cast<Real>(plan.mR)
         / static_cast<Real>(plan.m2);
  }

  void initialize_common(
    const RefSimplexPrecomp<D,Real>& pre_in,
    const Real* V_phys_colmajor,
    const int* global_vids_in,
    int interior_dim,
    Real tau_C_base_in,
    Real tau_residual_row_factor_in,
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
    if (!(tau_C_base_in > Real(0))
        || !std::isfinite(tau_C_base_in))
    {
      throw std::invalid_argument(
        "Leaf: tau base constant must be finite and positive");
    }
    if (!(tau_residual_row_factor_in > Real(0))
        || !std::isfinite(tau_residual_row_factor_in))
    {
      throw std::invalid_argument(
        "Leaf: tau residual-row factor must be finite and positive");
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
    if (interior_dim <= 0 || interior_dim > M)
    {
      throw std::invalid_argument(
        "Leaf: invalid interior residual dimension");
    }
    m_int = interior_dim;
    kf = pre->kf;
    nface = D + 1;
    nb = nface * kf;
    ntau_rows = m_int + nb;
    tau_C_base = tau_C_base_in;
    tau_residual_row_factor = tau_residual_row_factor_in;
    tau_C = tau_C_base * tau_residual_row_factor;
    if (!(tau_C > Real(0)) || !std::isfinite(tau_C))
    {
      throw std::invalid_argument(
        "Leaf: effective tau constant must be finite and positive");
    }
    lsmr_options = opts;
    leaf_options = leaf_opts;
    operator_mode = leaf_options.operator_mode;
    least_squares_solver =
      leaf_options.least_squares_solver;
    if (least_squares_solver ==
        LeafLeastSquaresSolver::Auto)
    {
      least_squares_solver =
        operator_mode == LeafOperatorMode::Dense
        ? LeafLeastSquaresSolver::DenseQR
        : LeafLeastSquaresSolver::LSMR;
    }
    if (least_squares_solver ==
          LeafLeastSquaresSolver::DenseQR
        && operator_mode != LeafOperatorMode::Dense)
    {
      throw std::invalid_argument(
        "Leaf: DenseQR requires LeafOperatorMode::Dense");
    }
#ifdef TIMING
    timing_interior_scale_seconds = 0.0;
    timing_interior_scale_actions.reset();
#endif

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

  void allocate_dense_interior_storage()
  {
    L.assign(
      (std::size_t)m_int * M,
      Real(0));
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
    A_tau.clear();
  }

  void assemble_dense_backend(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticDenseWorkspace<D,Real>& dense_work)
  {
    allocate_dense_storage();

    assemble_dense_interior(
      plan,
      coeffs,
      dense_work);

    assemble_dense_boundary_maps();
  }

  void assemble_dense_sparse_backend(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticDenseWorkspace<D,Real>& dense_work)
  {
    allocate_dense_interior_storage();

    assemble_dense_interior(
      plan,
      coeffs,
      dense_work);
  }

  void assemble_dense_interior(
    const EllipticPlan<D,Real>& plan,
    const EllipticElementCoefficientsView<D,Real>& coeffs,
    EllipticDenseWorkspace<D,Real>& dense_work)
  {

    jdsimplex_assemble_elliptic_L_int_dag<D,Real>(
      *pre,
      geom,
      plan,
      coeffs,
      dense_work,
      L.data());

#ifdef TIMING
    timing_interior_scale_actions.reset();
#endif

    if (leaf_options.interior_scale_override > Real(0))
    {
      sL = leaf_options.interior_scale_override;
#ifdef TIMING
      timing_interior_scale_seconds = 0.0;
#endif
    }
    else
    {
#ifdef TIMING
      timer scale_timer;
      scale_timer.tic();
#endif

      sL = row_rms_scale(
        L.data(),
        m_int,
        M);

#ifdef TIMING
      timing_interior_scale_seconds =
        scale_timer.toc();
#endif
    }
  }

  void set_matrix_free_interior_scale()
  {
    if (leaf_options.interior_scale_override > Real(0))
    {
      sL = leaf_options.interior_scale_override;
#ifdef TIMING
      timing_interior_scale_seconds = 0.0;
      timing_interior_scale_actions.reset();
      if (elliptic_action_work_) elliptic_action_work_->reset_timings();
#endif
      return;
    }

#ifdef TIMING
    timer scale_timer;
    scale_timer.tic();
#endif

    if (!elliptic_plan_ || !elliptic_action_work_)
    {
      throw std::runtime_error(
        "Leaf: matrix-free scale without action backend");
    }

#ifdef TIMING
    elliptic_action_work_->reset_timings();
#endif

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

#ifdef TIMING
    timing_interior_scale_seconds =
      scale_timer.toc();
    timing_interior_scale_actions = elliptic_action_work_->timings();
    elliptic_action_work_->reset_timings();
#endif
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
    if (operator_mode == LeafOperatorMode::Dense
        || operator_mode == LeafOperatorMode::DenseSparse)
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

  void apply_interior_operator_transpose(
    const Real* y,
    Real* x) const
  {
    if (operator_mode == LeafOperatorMode::Dense
        || operator_mode == LeafOperatorMode::DenseSparse)
    {
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        M,
        1,
        m_int,
        Real(1),
        L.data(),
        m_int,
        y,
        m_int,
        Real(0),
        x,
        M);
      return;
    }

    if (!elliptic_plan_
        || !elliptic_action_work_)
    {
      throw std::runtime_error(
        "Leaf: missing matrix-free elliptic transpose backend");
    }

    jdsimplex_apply_elliptic_transpose<D,Real>(
      *pre,
      geom,
      *elliptic_plan_,
      owned_coeffs_,
      *elliptic_action_work_,
      y,
      x);
  }

  void apply_stacked_action(
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

  void apply_stacked_action_transpose(
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

    apply_interior_operator_transpose(
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

  int solve_action(
    const Real* rhs,
    Real* x_out,
    SolveWorkspace& workspace,
    LsmrInfo* info) const
  {
    if (!rhs || !x_out || !info)
    {
      return -2;
    }
    if (!uses_action_solver())
    {
      return -5;
    }
    if (uses_clenshaw_backend()
        && (!elliptic_plan_ || !elliptic_action_work_))
    {
      return -5;
    }
    if (operator_mode == LeafOperatorMode::DenseSparse
        && !has_dense_interior_operator())
    {
      return -5;
    }

    const int row_count = ntau_rows;
    const int column_count = M;

    ensure_solve_workspace(workspace);

    std::vector<double>& rhs_double =
      workspace.rhs_double;
    std::vector<double>& u =
      workspace.u;
    std::vector<double>& v =
      workspace.v;
    std::vector<double>& x =
      workspace.x;

    std::fill(
      rhs_double.begin(),
      rhs_double.end(),
      0.0);
    std::fill(
      u.begin(),
      u.end(),
      0.0);
    std::fill(
      v.begin(),
      v.end(),
      0.0);
    std::fill(
      x.begin(),
      x.end(),
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

          apply_stacked_action_transpose(
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

          apply_stacked_action(
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
    const Real* matrix_free_solution,
    SolveWorkspace& workspace) const
  {
    ensure_solve_workspace(workspace);
    std::vector<Real>& dense_solution =
      workspace.verify_solution;
    std::fill(
      dense_solution.begin(),
      dense_solution.end(),
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
    apply_stacked_action(
      x.data(),
      action_forward.data());

    dense_apply_transpose(
      A_tau.data(),
      ntau_rows,
      M,
      y.data(),
      dense_transpose.data());
    apply_stacked_action_transpose(
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
