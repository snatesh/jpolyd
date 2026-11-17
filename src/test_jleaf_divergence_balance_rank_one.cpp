#include <jdivergence.hh>
#include <jleaf.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Equality-constrained dense-QR leaf response maps.
//
// Tests with general polynomial A(x), b(x), c(x), nonzero div(b),
// arbitrary kappa, and full residual degree R=n.
//
// The unconstrained panel [U_lambda U_s] is constructed before the
// balance row. The constrained panel is constructed after installing it.
// The test checks:
//   (1) every response column satisfies the exact affine balance;
//   (2) arbitrary trace/source combinations conserve;
//   (3) the unconstrained panel obeys the least-squares normal equations;
//   (4) the corrected panel obeys the KKT stationarity equations;
//   (5) the objective increase is defect^2/(a^T G^{-1} a);
//   (6) the correction has rank <= 1 and is nontrivial;
//   (7) invalidating the balance row restores ordinary QR behavior.
//
// No modifications to the HPS merge or C wrappers are required.

namespace {

using Real = double;

template<int D>
using Point = std::array<Real, D>;

template<int D>
using Precomp = jsimplex::RefSimplexPrecomp<D, Real>;

template<int D>
using Leaf = jsimplex::Leaf<D, Real>;

Real factorial(int n)
{
  Real value = Real(1);
  for (int i = 2; i <= n; ++i)
    value *= Real(i);
  return value;
}

void check_close(const std::string& label, Real actual,
                 Real expected, Real tolerance, Real& max_error)
{
  const Real error = std::abs(actual - expected);
  max_error = std::max(max_error, error);
  if (!std::isfinite(actual) ||
      error > tolerance * std::max(Real(1), std::abs(expected)))
  {
    std::ostringstream message;
    message << std::setprecision(17) << label
            << ": actual=" << actual
            << ", expected=" << expected
            << ", error=" << error
            << ", tolerance=" << tolerance;
    throw std::runtime_error(message.str());
  }
}

Real dot(const std::vector<Real>& a, const std::vector<Real>& b)
{
  if (a.size() != b.size())
    throw std::runtime_error("dot: length mismatch");
  Real result = Real(0);
  for (std::size_t i = 0; i < a.size(); ++i)
    result += a[i] * b[i];
  return result;
}

std::vector<Real> matvec(const std::vector<Real>& A, int rows,
                         int columns, const std::vector<Real>& x)
{
  if (A.size() != static_cast<std::size_t>(rows) * columns ||
      x.size() != static_cast<std::size_t>(columns))
    throw std::runtime_error("matvec: dimension mismatch");
  std::vector<Real> result(static_cast<std::size_t>(rows), Real(0));
  for (int j = 0; j < columns; ++j)
    for (int i = 0; i < rows; ++i)
      result[static_cast<std::size_t>(i)] +=
        A[static_cast<std::size_t>(i) +
          static_cast<std::size_t>(rows) * j] * x[j];
  return result;
}

template<int D>
std::array<Real, D * (D + 1)> make_vertices(bool reversed)
{
  std::array<Real, D * (D + 1)> V{};
  for (int r = 0; r < D; ++r)
    V[static_cast<std::size_t>(r)] = Real(0.09 + 0.07 * r);
  for (int j = 0; j < D; ++j)
    for (int r = 0; r < D; ++r)
      V[static_cast<std::size_t>(r) +
        static_cast<std::size_t>(D) * (j + 1)] =
        V[static_cast<std::size_t>(r)] +
        (r == j ? Real(0.83 + 0.09 * j) :
                  Real(0.035 * (r + 1) / (j + 1)));
  if (reversed)
  {
    if constexpr (D == 1)
      V[1] = V[0] - (V[1] - V[0]);
    else
      for (int r = 0; r < D; ++r)
        std::swap(V[static_cast<std::size_t>(r) + D],
                  V[static_cast<std::size_t>(r) + 2 * D]);
  }
  return V;
}

template<int D>
Point<D> physical_point(
  const std::array<Real, D * (D + 1)>& vertices,
  const jsimplex::DSimplexGeom<D, Real>& geom,
  const Real* xi)
{
  Point<D> x{};
  for (int r = 0; r < D; ++r)
  {
    x[r] = vertices[static_cast<std::size_t>(r)];
    for (int a = 0; a < D; ++a)
      x[r] += geom.B[static_cast<std::size_t>(r) + D * a] * xi[a];
  }
  return x;
}

template<int D>
struct Manufactured
{
  Point<D> v{};
  Point<D> w{};
  Point<D> alpha{};
  Point<D> theta{};
  Real epsilon = Real(0.045);

  Manufactured()
  {
    for (int r = 0; r < D; ++r)
    {
      v[r] = Real(0.17 + 0.031 * r);
      w[r] = Real(0.36 - 0.041 * r);
      alpha[r] = Real(0.095 + 0.019 * r);
      theta[r] = Real(0.13 + 0.024 * r);
    }
  }

  Real t(const Point<D>& x) const
  {
    Real value = Real(1.14);
    for (int r = 0; r < D; ++r)
      value += v[r] * x[r];
    return value;
  }

  Real a0(const Point<D>& x) const
  {
    Real value = Real(1.8);
    for (int r = 0; r < D; ++r)
      value += alpha[r] * x[r];
    return value;
  }

  Real a1(const Point<D>& x) const
  {
    Real value = Real(1);
    for (int r = 0; r < D; ++r)
      value += theta[r] * x[r];
    return value;
  }

  // Symmetric, positive definite, genuinely variable diffusion tensor.
  Real A(int r, int s, const Point<D>& x) const
  {
    return Real(r == s) * a0(x) +
      epsilon * a1(x) * w[r] * w[s];
  }

  // (div A)_s = sum_r partial_{x_r} A_{rs}.
  Real div_A(int s) const
  {
    Real value = alpha[s];
    for (int r = 0; r < D; ++r)
      value += epsilon * theta[r] * w[r] * w[s];
    return value;
  }

  Real b(int r, const Point<D>& x) const
  {
    return Real(0.24 + 0.037 * r) +
      Real(0.09 + 0.025 * r) * x[r] +
      Real(0.042 + 0.009 * r) * x[r] * x[r] +
      Real(0.052) * x[(r + 1) % D];
  }

  // This divergence is genuinely nonconstant.
  Real div_b(const Point<D>& x) const
  {
    Real value = Real(0);
    for (int r = 0; r < D; ++r)
      value += Real(0.09 + 0.025 * r) +
        Real(2) * Real(0.042 + 0.009 * r) * x[r];
    if constexpr (D == 1)
      value += Real(0.052);
    return value;
  }

  // Nonconstant zeroth-order coefficient.
  Real c(const Point<D>& x) const
  {
    Real value = Real(0.37);
    for (int r = 0; r < D; ++r)
      value += Real(0.052 + 0.008 * r) * x[r];
    return value;
  }

  // Degree-3 polynomial with exact gradient and Hessian.
  Real u(const Point<D>& x) const
  {
    const Real value = t(x);
    return value * value * value;
  }

  Real grad_u(int r, const Point<D>& x) const
  {
    const Real value = t(x);
    return Real(3) * value * value * v[r];
  }

  Real diffusion_source(const Point<D>& x) const
  {
    const Real value = t(x);
    Real divA_dot_grad = Real(0);
    Real principal = Real(0);
    for (int s = 0; s < D; ++s)
    {
      divA_dot_grad += div_A(s) * grad_u(s, x);
      for (int r = 0; r < D; ++r)
        principal += A(r, s, x) * Real(6) * value * v[r] * v[s];
    }
    return -(divA_dot_grad + principal);
  }

  Real f(const Point<D>& x, bool with_b, bool with_c) const
  {
    Real result = diffusion_source(x);
    if (with_b)
      for (int r = 0; r < D; ++r)
        result += b(r, x) * grad_u(r, x);
    if (with_c)
      result += c(x) * u(x);
    return result;
  }

  Real sigma(const Point<D>& x, bool with_b, bool with_c) const
  {
    return (with_c ? c(x) : Real(0)) -
      (with_b ? div_b(x) : Real(0));
  }
};

template<int D, class Func>
std::vector<Real> project(
  const Precomp<D>& pre,
  const std::array<Real, D * (D + 1)>& vertices,
  const jsimplex::DSimplexGeom<D, Real>& geom,
  int p, bool residual_family, Func&& fun)
{
  const int Mp = jsimplex::Basis<D, Real>::dim_Pi(p);
  const int nq = pre.nq_vol; // X_res and X_vol use the same count.
  const auto& X = residual_family ? pre.X_res : pre.X_vol;
  const auto& W = residual_family ? pre.W_res : pre.W_vol;
  const auto& V = residual_family ? pre.V_res : pre.V_vol;
  std::vector<Real> result(static_cast<std::size_t>(Mp), Real(0));
  for (int iq = 0; iq < nq; ++iq)
  {
    const auto x = physical_point<D>(
      vertices, geom, X.data() + static_cast<std::size_t>(iq) * D);
    const Real value = fun(x);
    for (int j = 0; j < Mp; ++j)
      result[static_cast<std::size_t>(j)] +=
        W[static_cast<std::size_t>(iq)] * value *
        V[static_cast<std::size_t>(iq) + static_cast<std::size_t>(nq) * j];
  }
  return result;
}

Real max_abs(const std::vector<Real>& v)
{
  Real result = Real(0);
  for (Real x : v)
    result = std::max(result, std::abs(x));
  return result;
}

Real norm_sq(const std::vector<Real>& v)
{
  Real result = Real(0);
  for (Real x : v)
    result += x * x;
  return result;
}

std::vector<Real> column(const std::vector<Real>& U,
                         int rows, int j)
{
  return std::vector<Real>(
    U.begin() + static_cast<std::size_t>(rows) * j,
    U.begin() + static_cast<std::size_t>(rows) * (j + 1));
}

std::vector<Real> transpose_matvec(
  const std::vector<Real>& A, int rows, int cols,
  const std::vector<Real>& x)
{
  if (x.size() != static_cast<std::size_t>(rows) ||
      A.size() != static_cast<std::size_t>(rows) * cols)
    throw std::runtime_error("transpose_matvec: size mismatch");
  std::vector<Real> y(static_cast<std::size_t>(cols), Real(0));
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
      y[static_cast<std::size_t>(j)] +=
        A[static_cast<std::size_t>(i) +
          static_cast<std::size_t>(rows) * j] *
        x[static_cast<std::size_t>(i)];
  return y;
}

std::vector<Real> panel_action(const std::vector<Real>& U,
                               int rows, int cols,
                               const std::vector<Real>& x)
{
  return matvec(U, rows, cols, x);
}

template<int D>
void run_rank_one_case(int degree, bool half, bool reversed,
                       Real tolerance,
                       const std::string& stencil_folder)
{
  constexpr int p_A = 1;
  constexpr int p_b = 2;
  constexpr int p_c = 1;
  const auto vertices = make_vertices<D>(reversed);
  jsimplex::DSimplexGeom<D, Real> geom{};
  jsimplex::dsimplex_affine_from_verts<D, Real>(
    vertices.data(), geom);

  std::array<int, D + 1> global_ids{};
  for (int i = 0; i <= D; ++i)
    global_ids[static_cast<std::size_t>(i)] =
      23 + 5 * ((i + 2) % (D + 1));

  std::array<Real, D + 1> kappa{};
  for (int i = 0; i <= D; ++i)
    kappa[static_cast<std::size_t>(i)] = half
      ? Real(0.5) : Real(0.11 + 0.19 * i + 0.037 * (i % 2));

  const int q = degree + 4;
  Precomp<D> pre(degree, 4, q, q, kappa.data(), stencil_folder);
  const int M = pre.M;
  const int MpA = jsimplex::Basis<D, Real>::dim_Pi(p_A);
  const int MpB = jsimplex::Basis<D, Real>::dim_Pi(p_b);
  Manufactured<D> problem;

  std::vector<Real> A(static_cast<std::size_t>(D) * D * MpA);
  std::vector<Real> b(static_cast<std::size_t>(D) * MpB);
  for (int r = 0; r < D; ++r)
  {
    for (int s = 0; s < D; ++s)
    {
      const auto coeff = project<D>(
        pre, vertices, geom, p_A, false,
        [&](const Point<D>& x) { return problem.A(r, s, x); });
      std::copy(coeff.begin(), coeff.end(),
        A.begin() + (static_cast<std::size_t>(r) * D + s) * MpA);
    }
    const auto coeff = project<D>(
      pre, vertices, geom, p_b, false,
      [&](const Point<D>& x) { return problem.b(r, x); });
    std::copy(coeff.begin(), coeff.end(),
      b.begin() + static_cast<std::size_t>(r) * MpB);
  }
  const auto c = project<D>(
    pre, vertices, geom, p_c, false,
    [&](const Point<D>& x) { return problem.c(x); });

  const auto expanded = jsimplex::expand_divergence_form_element<D, Real>(
    pre, geom, p_A, A.data(), p_b, b.data(), p_c, c.data());
  jsimplex::EllipticPlan<D, Real> plan(
    pre, expanded.degrees, true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature, false);
  if (plan.mR != M)
    throw std::runtime_error("Stage 2b requires full R=n residual");

  jsimplex::EllipticDenseWorkspace<D, Real> elliptic_work(plan);
  jsimplex::MultByQQuadraturePlan<D - 1, Real> face_quad(
    pre.kappa_face.data(), degree, degree + p_A + 2);
  jsimplex::MultByQQuadratureWorkspace<Real> face_work;

  typename Leaf<D>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;
  typename Leaf<D>::LsmrOptions solver_options{};
  Leaf<D> leaf(pre, vertices.data(), global_ids.data(),
    plan, expanded.view(), elliptic_work, Real(0.85),
    solver_options, options);
  if (leaf.M != M || leaf.m_int != M)
    throw std::runtime_error("Leaf does not use full residual degree");

  leaf.set_dense_conormal_flux(p_A, A.data(), face_quad, face_work);
  if (leaf.balance_row_ready)
    throw std::runtime_error("New leaf must not enforce balance yet");

  typename Leaf<D>::SolveWorkspace workspace(leaf);
  const int nb = leaf.nb;
  const int nrhs = nb + M;
  const int rows = leaf.ntau_rows;

  // Construct the ORIGINAL, unconstrained response panel first.
  std::vector<Real> U0(static_cast<std::size_t>(M) * nrhs);
  leaf.solve_response_maps(U0.data(), M, workspace);

  // Assemble the validated Stage 2a physical balance row, using the
  // ORIGINAL (not expanded) b and c modal coefficients.
  leaf.assemble_dense_divergence_balance_row(
    p_b, b.data(), p_c, c.data());
  if (!leaf.balance_row_ready ||
      leaf.balance_a.size() != static_cast<std::size_t>(M) ||
      leaf.balance_g.size() != static_cast<std::size_t>(nb) ||
      leaf.balance_h.size() != static_cast<std::size_t>(M))
    throw std::runtime_error("Balance row is not available");

  std::vector<Real> Uc(static_cast<std::size_t>(M) * nrhs);
  leaf.solve_response_maps(Uc.data(), M, workspace);

  // A second constrained solve with the same workspace must be identical.
  std::vector<Real> Uc_repeat(Uc.size());
  leaf.solve_response_maps(Uc_repeat.data(), M, workspace);

  Real max_map_identity = Real(0);
  Real max_repeat_error = Real(0);
  Real max_correction = Real(0);
  Real max_kkt_error = Real(0);
  Real max_objective_error = Real(0);
  Real max_combined_balance = Real(0);

  std::vector<Real> defects(static_cast<std::size_t>(nrhs));
  int active_correction_col = -1;

  // Every response column, not merely a particular test solution,
  // must satisfy a^T U_lambda + g^T = 0 or a^T U_s + h^T = 0.
  for (int j = 0; j < nrhs; ++j)
  {
    const auto u0 = column(U0, M, j);
    const auto uc = column(Uc, M, j);
    Real affine = j < nb ? leaf.balance_g[static_cast<std::size_t>(j)]
                         : leaf.balance_h[static_cast<std::size_t>(j - nb)];
    defects[static_cast<std::size_t>(j)] =
      dot(leaf.balance_a, u0) + affine;
    const Real residual = dot(leaf.balance_a, uc) + affine;
    max_map_identity = std::max(max_map_identity, std::abs(residual));
    check_close("balance identity at response column " + std::to_string(j),
      residual, Real(0), tolerance, max_map_identity);

    Real correction_norm = Real(0);
    for (int i = 0; i < M; ++i)
    {
      const std::size_t index = static_cast<std::size_t>(i) +
        static_cast<std::size_t>(M) * j;
      const Real diff = Uc[index] - U0[index];
      correction_norm += diff * diff;
      max_correction = std::max(max_correction, std::abs(diff));
      max_repeat_error = std::max(
        max_repeat_error, std::abs(Uc[index] - Uc_repeat[index]));
    }
    if (active_correction_col < 0 &&
        correction_norm > Real(1e-14) &&
        std::abs(defects[static_cast<std::size_t>(j)]) > Real(1e-8))
      active_correction_col = j;
  }

  if (active_correction_col < 0 || max_correction < Real(1e-9))
    throw std::runtime_error("Conservation correction was unexpectedly zero");
  check_close("repeat constrained solve", max_repeat_error,
    Real(0), tolerance, max_repeat_error);

  // A rank-one update means every nonzero correction column is collinear
  // with a single common direction. Check this independently of the
  // internal implementation of the projection.
  const auto anchor_before = column(U0, M, active_correction_col);
  const auto anchor_after = column(Uc, M, active_correction_col);
  std::vector<Real> anchor(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i)
    anchor[static_cast<std::size_t>(i)] =
      anchor_after[static_cast<std::size_t>(i)] -
      anchor_before[static_cast<std::size_t>(i)];
  const Real anchor_norm = norm_sq(anchor);
  Real max_rank_one_error = Real(0);
  for (int j = 0; j < nrhs; ++j)
  {
    const auto u0 = column(U0, M, j);
    const auto uc = column(Uc, M, j);
    std::vector<Real> diff(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i)
      diff[static_cast<std::size_t>(i)] =
        uc[static_cast<std::size_t>(i)] - u0[static_cast<std::size_t>(i)];
    const Real coefficient = dot(anchor, diff) / anchor_norm;
    for (int i = 0; i < M; ++i)
    {
      const Real error = diff[static_cast<std::size_t>(i)] -
        coefficient * anchor[static_cast<std::size_t>(i)];
      max_rank_one_error = std::max(max_rank_one_error, std::abs(error));
    }
  }
  check_close("rank-one update", max_rank_one_error,
    Real(0), tolerance, max_rank_one_error);

  // Independently read the already computed QR factor and calculate
  // a^T (H^T H)^(-1) a = ||R^(-T) a||^2 by forward substitution.
  // This is separate from the implementation's BLAS TRSM calls.
  if (!workspace.qr_factorized ||
      workspace.qr_factor.size() !=
        static_cast<std::size_t>(rows) * M)
    throw std::runtime_error("Missing dense QR factorization");
  const auto& R = workspace.qr_factor;
  std::vector<Real> w(static_cast<std::size_t>(M), Real(0));
  for (int i = 0; i < M; ++i)
  {
    Real value = leaf.balance_a[static_cast<std::size_t>(i)];
    for (int j = 0; j < i; ++j)
      value -= R[static_cast<std::size_t>(j) +
        static_cast<std::size_t>(rows) * i] *
        w[static_cast<std::size_t>(j)];
    const Real pivot = R[static_cast<std::size_t>(i) +
      static_cast<std::size_t>(rows) * i];
    w[static_cast<std::size_t>(i)] = value / pivot;
  }
  const Real denominator = norm_sq(w);
  if (!(denominator > Real(0)) || !std::isfinite(denominator))
    throw std::runtime_error("Degenerate projected constraint");

  // Use three RHS combinations: trace-only, source-only, and mixed.
  // Each is evaluated without calling Leaf's direct single-RHS solver,
  // which intentionally does NOT enforce conservation in Stage 2b.
  for (int rhs_case = 0; rhs_case < 3; ++rhs_case)
  {
    std::vector<Real> lambda(static_cast<std::size_t>(nb), Real(0));
    std::vector<Real> s(static_cast<std::size_t>(M), Real(0));
    if (rhs_case != 1)
    {
      for (int j = 0; j < nb; ++j)
      {
        const int f = j / leaf.kf;
        lambda[static_cast<std::size_t>(j)] =
          leaf.face_scale[static_cast<std::size_t>(f)] *
          (Real(0.21) * std::cos(Real(0.27) * (j + 1)) +
           Real(0.04) * std::sin(Real(0.93) * (j + 2)));
      }
    }
    if (rhs_case != 0)
    {
      for (int j = 0; j < M; ++j)
        s[static_cast<std::size_t>(j)] =
          Real(0.15) * std::sin(Real(0.17) * (j + 1)) +
          Real(0.06) * std::cos(Real(0.71) * (j + 2));
    }

    std::vector<Real> parameters(static_cast<std::size_t>(nrhs));
    std::copy(lambda.begin(), lambda.end(), parameters.begin());
    std::copy(s.begin(), s.end(), parameters.begin() + nb);
    const auto z0 = panel_action(U0, M, nrhs, parameters);
    const auto zc = panel_action(Uc, M, nrhs, parameters);

    const Real boundary_source = dot(leaf.balance_g, lambda) +
      dot(leaf.balance_h, s);
    const Real defect = dot(leaf.balance_a, z0) + boundary_source;
    const Real balance = dot(leaf.balance_a, zc) + boundary_source;
    max_combined_balance = std::max(
      max_combined_balance, std::abs(balance));
    check_close("combined physical balance", balance,
      Real(0), tolerance, max_combined_balance);
    check_close("public balance diagnostic",
      leaf.dense_divergence_balance_residual(
        zc.data(), lambda.data(), s.data()),
      Real(0), tolerance, max_combined_balance);

    // Build the SAME original stacked RHS in the source sign convention:
    // d=[-sL*s; sqrt(tau)*lambda].
    std::vector<Real> d(static_cast<std::size_t>(rows), Real(0));
    for (int i = 0; i < M; ++i)
      d[static_cast<std::size_t>(i)] = -leaf.sL * s[static_cast<std::size_t>(i)];
    for (int i = 0; i < nb; ++i)
      d[static_cast<std::size_t>(M + i)] =
        leaf.sqrt_tau_rows[static_cast<std::size_t>(i)] *
        lambda[static_cast<std::size_t>(i)];

    auto r0 = matvec(leaf.A_tau, rows, M, z0);
    auto rc = matvec(leaf.A_tau, rows, M, zc);
    std::vector<Real> Hdelta(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i)
    {
      r0[static_cast<std::size_t>(i)] -= d[static_cast<std::size_t>(i)];
      rc[static_cast<std::size_t>(i)] -= d[static_cast<std::size_t>(i)];
      Hdelta[static_cast<std::size_t>(i)] =
        rc[static_cast<std::size_t>(i)] - r0[static_cast<std::size_t>(i)];
    }
    const Real phi0 = norm_sq(r0);
    const Real phic = norm_sq(rc);
    const Real expected_increase = (defect * defect) / denominator;
    const Real observed_increase = phic - phi0;
    check_close("least-squares cost increase",
      observed_increase, expected_increase,
      tolerance * Real(30), max_objective_error);
    if (observed_increase < -tolerance * Real(30))
      throw std::runtime_error("Projection decreased the minimum LS cost");
    check_close("objective increment equals ||H delta z||^2",
      observed_increase, norm_sq(Hdelta),
      tolerance * Real(30), max_objective_error);

    const auto grad0 = transpose_matvec(leaf.A_tau, rows, M, r0);
    const auto gradc = transpose_matvec(leaf.A_tau, rows, M, rc);
    check_close("unconstrained least-squares normal equations",
      max_abs(grad0), Real(0),
      tolerance * Real(30), max_kkt_error);

    // For the constrained minimizer, there exists a scalar mu such
    // that H^T(H zc-d) + mu*a = 0.
    const Real a_norm_sq = norm_sq(leaf.balance_a);
    if (!(a_norm_sq > Real(0)))
      throw std::runtime_error("Zero balance normal");
    const Real mu = -dot(leaf.balance_a, gradc) / a_norm_sq;
    Real kkt_inf = Real(0);
    for (int i = 0; i < M; ++i)
    {
      const Real stationarity = gradc[static_cast<std::size_t>(i)] +
        mu * leaf.balance_a[static_cast<std::size_t>(i)];
      kkt_inf = std::max(kkt_inf, std::abs(stationarity));
    }
    check_close("constrained KKT stationarity", kkt_inf,
      Real(0), tolerance * Real(30), max_kkt_error);
  }

  // Replacing the conormal map must invalidate the balance row.
  // Without the row, the response panel must revert to the original
  // unconstrained dense-QR map.
  leaf.set_dense_conormal_flux(p_A, A.data(), face_quad, face_work);
  if (leaf.balance_row_ready)
    throw std::runtime_error("Conormal replacement did not clear constraint");
  std::vector<Real> Urestored(U0.size());
  leaf.solve_response_maps(Urestored.data(), M, workspace);
  Real max_restore_error = Real(0);
  for (std::size_t i = 0; i < U0.size(); ++i)
    max_restore_error = std::max(max_restore_error,
      std::abs(Urestored[i] - U0[i]));
  check_close("disabled mode matches original dense QR",
    max_restore_error, Real(0), tolerance, max_restore_error);

  std::cout << "D=" << D
            << ", n=" << degree
            << ", kappa=" << (half ? "half" : "mixed")
            << ", orientation=" << (reversed ? "reversed" : "positive")
            << ", column=" << std::scientific << std::setprecision(6)
            << max_map_identity
            << ", balance=" << max_combined_balance
            << ", kkt=" << max_kkt_error
            << ", objective=" << max_objective_error
            << ", rank1=" << max_rank_one_error
            << ", correction=" << max_correction
            << "  PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  // Usage: test_jleaf_divergence_balance_rank_one
  //   [n=4] [tol=2e-9] [stencils=stencils] [dimension=0]
  // dimension=0 tests D=1,2,3,4.
  const int n = argc > 1 ? std::stoi(argv[1]) : 4;
  const Real tolerance = argc > 2 ? std::stod(argv[2]) : Real(2e-9);
  const std::string stencils = argc > 3 ? argv[3] : "stencils";
  const int dimension = argc > 4 ? std::stoi(argv[4]) : 0;
  if (n < 4 || dimension < 0 || dimension > 4 || tolerance <= Real(0))
  {
    std::cerr << "Require n>=4, tolerance>0 and dimension=0,1,2,3,4\n";
    return 2;
  }
  try
  {
    if (dimension == 0 || dimension == 1)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_rank_one_case<1>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 2)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_rank_one_case<2>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 3)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_rank_one_case<3>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 4)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_rank_one_case<4>(n, half, reversed, tolerance, stencils);
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "All dense divergence-form rank-one tests passed.\n";
  return 0;
}
