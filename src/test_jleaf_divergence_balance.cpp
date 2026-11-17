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

// Verify the UNENFORCED, physical-measure conservation row
//    a^T z + g^T lambda + h^T s = 0,     s = -f_physical,
// for a general divergence-form PDE
//    -div(A grad u) + b.grad(u) + c*u = f_physical.
//
// This test uses only the dense-QR divergence-form leaf. It checks both
// consistent lambda=T*z and perturbed lambda, including the tau penalty.
// Independent references use a manufactured polynomial and unweighted
// volume/face quadrature; the source, reaction, advection and diffusion are
// evaluated analytically rather than through the assembled balance vectors.

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

template<int D>
struct IndependentIntegrals
{
  Real volume_sigma = Real(0);
  Real volume_source = Real(0);
  Real boundary_adv = Real(0);
  Real boundary_conormal = Real(0);
  Real perturbation_flux = Real(0);
};

template<int D>
IndependentIntegrals<D> integrate_independently(
  const Leaf<D>& leaf,
  const std::array<Real, D * (D + 1)>& vertices,
  const std::array<int, D + 1>& global_ids,
  const Manufactured<D>& problem,
  bool with_b, bool with_c,
  const std::vector<Real>& delta_moments,
  int n)
{
  IndependentIntegrals<D> integrals;
  const int q = n + 5; // Overresolved, independent unweighted quadrature.
  const int nq = static_cast<int>(
    jsimplex::QuadMapped<D, Real>::npoints(static_cast<unsigned int>(q)));
  std::array<Real, D + 1> half{};
  half.fill(Real(0.5));
  std::vector<Real> X(static_cast<std::size_t>(nq) * D);
  std::vector<Real> W(static_cast<std::size_t>(nq));
  if (jsimplex::QuadMapped<D, Real>::build_kappa(
        static_cast<unsigned int>(q), half.data(), X.data(), W.data()) != nq)
    throw std::runtime_error("independent volume quadrature failed");

  const Real vol_factor = leaf.geom.detBabs / factorial(D);
  for (int iq = 0; iq < nq; ++iq)
  {
    const auto x = physical_point<D>(
      vertices, leaf.geom, X.data() + static_cast<std::size_t>(iq) * D);
    const Real dvol = vol_factor * W[static_cast<std::size_t>(iq)];
    integrals.volume_sigma += dvol *
      problem.sigma(x, with_b, with_c) * problem.u(x);
    integrals.volume_source += dvol * problem.f(x, with_b, with_c);
  }

  // Canonical face parameterization derived solely from the vertex IDs,
  // not from T_ref, F, or the balance row.
  const int nqf = D == 1 ? 1 : static_cast<int>(
    jsimplex::QuadMapped<D - 1, Real>::npoints(
      static_cast<unsigned int>(q)));
  std::vector<Real> Y;
  std::vector<Real> Wf;
  std::vector<Real> Vf;
  if constexpr (D > 1)
  {
    std::array<Real, D> face_half{};
    face_half.fill(Real(0.5));
    Y.resize(static_cast<std::size_t>(nqf) * (D - 1));
    Wf.resize(static_cast<std::size_t>(nqf));
    if (jsimplex::QuadMapped<D - 1, Real>::build_kappa(
          static_cast<unsigned int>(q), face_half.data(),
          Y.data(), Wf.data()) != nqf)
      throw std::runtime_error("independent face quadrature failed");

    std::vector<int> alpha, tail;
    std::vector<Real> invh;
    jsimplex::Basis<D - 1, Real>::build_structures(
      n, leaf.pre->kappa_face.data(), alpha, tail, invh);
    Vf.resize(static_cast<std::size_t>(nqf) * leaf.kf);
    jsimplex::Basis<D - 1, Real>::eval_all(
      Y.data(), D - 1, 1, nqf,
      leaf.pre->kappa_face.data(), n,
      alpha.data(), tail.data(), invh.data(),
      Vf.data(), nqf, nullptr);
  }

  for (int f = 0; f <= D; ++f)
  {
    int local_vertices[D];
    int local_to_canonical[D];
    jsimplex::dsimplex_face_vertices<D>(f, local_vertices);
    jsimplex::dsimplex_compute_face_sigma_array<D>(
      global_ids.data(), f, local_to_canonical);
    std::array<Point<D>, D> canonical_vertices{};
    for (int l = 0; l < D; ++l)
      for (int r = 0; r < D; ++r)
        canonical_vertices[static_cast<std::size_t>(
          local_to_canonical[l])][r] =
          vertices[static_cast<std::size_t>(r) +
            static_cast<std::size_t>(D) * local_vertices[l]];

    const Real face_measure = leaf.face_scale[static_cast<std::size_t>(f)] /
      factorial(D - 1);
    for (int iq = 0; iq < nqf; ++iq)
    {
      Point<D> x = canonical_vertices[0];
      if constexpr (D > 1)
      {
        const Real* y = Y.data() + static_cast<std::size_t>(iq) * (D - 1);
        for (int a = 0; a < D - 1; ++a)
          for (int r = 0; r < D; ++r)
            x[r] += y[a] * (canonical_vertices[a + 1][r] -
                            canonical_vertices[0][r]);
      }
      const Real weight = face_measure * (D == 1 ? Real(1) :
        Wf[static_cast<std::size_t>(iq)]);
      Real bn = Real(0);
      Real conormal = Real(0);
      for (int r = 0; r < D; ++r)
      {
        const Real normal = leaf.unit_normal[
          static_cast<std::size_t>(f) * D + r];
        if (with_b)
          bn += normal * problem.b(r, x);
        for (int s = 0; s < D; ++s)
          conormal += normal * problem.A(r, s, x) *
            problem.grad_u(s, x);
      }
      integrals.boundary_adv += weight * bn * problem.u(x);
      integrals.boundary_conormal += weight * conormal;

      Real delta = Real(0);
      for (int j = 0; j < leaf.kf; ++j)
      {
        const Real psi = D == 1 ? Real(1) :
          Vf[static_cast<std::size_t>(iq) +
             static_cast<std::size_t>(nqf) * j];
        delta += delta_moments[static_cast<std::size_t>(f) * leaf.kf + j] *
          psi / leaf.face_scale[static_cast<std::size_t>(f)];
      }
      integrals.perturbation_flux += weight *
        (bn - leaf.tau_face[static_cast<std::size_t>(f)]) * delta;
    }
  }
  return integrals;
}

template<int D>
void run_case(int degree, bool half, bool reversed,
              Real tolerance, const std::string& stencil_folder)
{
  constexpr int p_A = 1;
  constexpr int p_b = 2;
  constexpr int p_c = 1;
  if (degree < 4)
    throw std::invalid_argument("manufactured source requires n >= 4");

  const auto vertices = make_vertices<D>(reversed);
  jsimplex::DSimplexGeom<D, Real> geom;
  jsimplex::dsimplex_affine_from_verts<D, Real>(vertices.data(), geom);
  if (!geom.valid || (geom.detB < 0) != reversed)
    throw std::runtime_error("unexpected simplex geometry/orientation");

  std::array<int, D + 1> global_ids{};
  for (int i = 0; i <= D; ++i)
    global_ids[i] = 23 + 5 * ((i + 2) % (D + 1));
  std::array<Real, D + 1> kappa{};
  for (int i = 0; i <= D; ++i)
    kappa[i] = half ? Real(0.5) :
      Real(0.11 + 0.19 * i + 0.037 * (i % 2));

  const int q = degree + 4;
  Precomp<D> pre(degree, 4, q, q, kappa.data(), stencil_folder);
  const int M = pre.M;
  const int MpA = jsimplex::Basis<D, Real>::dim_Pi(p_A);
  const int MpB = jsimplex::Basis<D, Real>::dim_Pi(p_b);
  const int MpC = jsimplex::Basis<D, Real>::dim_Pi(p_c);
  Manufactured<D> problem;

  std::vector<Real> A(static_cast<std::size_t>(D) * D * MpA, Real(0));
  std::vector<Real> b(static_cast<std::size_t>(D) * MpB, Real(0));
  for (int r = 0; r < D; ++r)
  {
    for (int s = 0; s < D; ++s)
    {
      auto coeff = project<D>(pre, vertices, geom, p_A, false,
        [&](const Point<D>& x) { return problem.A(r, s, x); });
      std::copy(coeff.begin(), coeff.end(),
        A.begin() + (static_cast<std::size_t>(r) * D + s) * MpA);
    }
    auto coeff = project<D>(pre, vertices, geom, p_b, false,
      [&](const Point<D>& x) { return problem.b(r, x); });
    std::copy(coeff.begin(), coeff.end(),
      b.begin() + static_cast<std::size_t>(r) * MpB);
  }
  const auto c = project<D>(pre, vertices, geom, p_c, false,
    [&](const Point<D>& x) { return problem.c(x); });
  const auto u = project<D>(pre, vertices, geom, degree, false,
    [&](const Point<D>& x) { return problem.u(x); });

  const auto expanded = jsimplex::expand_divergence_form_element<D, Real>(
    pre, geom, p_A, A.data(), p_b, b.data(), p_c, c.data());
  jsimplex::EllipticPlan<D, Real> plan(
    pre, expanded.degrees, true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature, false);
  if (plan.mR != M)
    throw std::runtime_error("full residual convention was not preserved");
  jsimplex::EllipticDenseWorkspace<D, Real> elliptic_work(plan);
  jsimplex::MultByQQuadraturePlan<D - 1, Real> face_quad(
    pre.kappa_face.data(), degree, degree + p_A + 2);
  jsimplex::MultByQQuadratureWorkspace<Real> face_work;

  typename Leaf<D>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;
  typename Leaf<D>::LsmrOptions solver_options{};
  Leaf<D> leaf(pre, vertices.data(), global_ids.data(), plan,
    expanded.view(), elliptic_work, Real(0.85), solver_options, options);
  if (leaf.m_int != M || leaf.M != M)
    throw std::runtime_error("leaf residual dimension must be full M");

  // The row must not be formed with the default normal derivative map.
  bool rejected = false;
  try
  {
    leaf.assemble_dense_divergence_balance_row(
      p_b, b.data(), p_c, c.data());
  }
  catch (const std::runtime_error&)
  {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("balance row accepted missing conormal map");

  leaf.set_dense_conormal_flux(p_A, A.data(), face_quad, face_work);
  if (!leaf.dense_conormal_flux_installed || leaf.balance_row_ready)
    throw std::runtime_error("incorrect row/conormal initial state");

  // Nontrivial canonical-face perturbation (moment coordinates, not
  // expansion coefficients). Including high modes checks weighted basis use.
  std::vector<Real> delta(static_cast<std::size_t>(leaf.nb), Real(0));
  for (int f = 0; f <= D; ++f)
    for (int j = 0; j < leaf.kf; ++j)
    {
      const Real amp = Real(0.006) *
        std::sin(Real(0.61) * (1 + 2 * f + 3 * j)) /
        (Real(1) + Real(0.4) * j);
      delta[static_cast<std::size_t>(f) * leaf.kf + j] =
        leaf.face_scale[static_cast<std::size_t>(f)] * amp;
    }

  const auto lambda = matvec(leaf.T, leaf.nb, leaf.M, u);
  std::vector<Real> lambda_perturbed = lambda;
  for (std::size_t i = 0; i < lambda.size(); ++i)
    lambda_perturbed[i] += delta[i];

  Real max_balance_error = Real(0);
  Real max_perturb_error = Real(0);
  Real max_component_error = Real(0);

  for (int case_index = 0; case_index < 4; ++case_index)
  {
    const bool with_b = (case_index == 0 || case_index == 1);
    const bool with_c = (case_index == 0 || case_index == 2);
    const std::string case_label =
      with_b ? (with_c ? "b+c" : "b only") :
               (with_c ? "c only" : "neither");
    const auto source_coeff = project<D>(
      pre, vertices, geom, degree, true,
      [&](const Point<D>& x) {
        return problem.f(x, with_b, with_c);
      });
    std::vector<Real> internal_source = source_coeff;
    for (Real& v : internal_source)
      v = -v;

    leaf.assemble_dense_divergence_balance_row(
      with_b ? p_b : -1, with_b ? b.data() : nullptr,
      with_c ? p_c : -1, with_c ? c.data() : nullptr);
    if (!leaf.balance_row_ready ||
        leaf.balance_a.size() != static_cast<std::size_t>(M) ||
        leaf.balance_g.size() != static_cast<std::size_t>(leaf.nb) ||
        leaf.balance_h.size() != static_cast<std::size_t>(M))
      throw std::runtime_error("bad balance row size/readiness");

    const auto reference = integrate_independently<D>(
      leaf, vertices, global_ids, problem,
      with_b, with_c, delta, degree);
    const Real reference_base = reference.volume_sigma +
      reference.boundary_adv - reference.boundary_conormal -
      reference.volume_source;
    check_close("independent physical conservation " + case_label,
      reference_base, Real(0), tolerance, max_balance_error);

    check_close("source integral " + case_label,
      dot(leaf.balance_h, internal_source),
      -reference.volume_source, tolerance, max_component_error);

    const auto F_u = matvec(leaf.F, leaf.nb, M, u);
    Real conormal_moment_integral = Real(0);
    for (int f = 0; f <= D; ++f)
      for (int j = 0; j < leaf.kf; ++j)
        conormal_moment_integral +=
          pre.face_moment_to_lebesgue[static_cast<std::size_t>(j)] *
          F_u[static_cast<std::size_t>(f) * leaf.kf + j];
    check_close("projected conormal physical integral " + case_label,
      conormal_moment_integral, reference.boundary_conormal,
      tolerance, max_component_error);

    const Real residual_base = leaf.dense_divergence_balance_residual(
      u.data(), lambda.data(), internal_source.data());
    check_close("assembled consistent-trace row " + case_label,
      residual_base, reference_base, tolerance, max_balance_error);

    const Real residual_perturbed = leaf.dense_divergence_balance_residual(
      u.data(), lambda_perturbed.data(), internal_source.data());
    check_close("assembled perturbed-trace row " + case_label,
      residual_perturbed,
      reference_base + reference.perturbation_flux,
      tolerance, max_perturb_error);
    check_close("isolated g dot delta " + case_label,
      residual_perturbed - residual_base,
      reference.perturbation_flux,
      tolerance, max_perturb_error);
    check_close("g dot delta directly " + case_label,
      dot(leaf.balance_g, delta),
      reference.perturbation_flux,
      tolerance, max_perturb_error);
  }

  // Installing a new conormal map invalidates an existing row.
  leaf.set_dense_conormal_flux(p_A, A.data(), face_quad, face_work);
  if (leaf.balance_row_ready || !leaf.balance_a.empty() ||
      !leaf.balance_g.empty() || !leaf.balance_h.empty())
    throw std::runtime_error("conormal replacement did not invalidate row");

  std::cout << "D=" << D
            << ", n=" << degree
            << ", kappa=" << (half ? "half" : "mixed")
            << ", orientation=" << (reversed ? "reversed" : "positive")
            << ", balance=" << std::scientific << std::setprecision(6)
            << max_balance_error
            << ", perturb=" << max_perturb_error
            << ", components=" << max_component_error
            << "  PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  // Usage: test_jleaf_divergence_balance [n=4] [tol=2e-9]
  //        [stencils=stencils] [dimension=0]
  // dimension=0 runs D=1..4. Full residual degree R=n is always used.
  const int n = argc > 1 ? std::stoi(argv[1]) : 4;
  const Real tolerance = argc > 2 ? std::stod(argv[2]) : Real(2e-9);
  const std::string stencils = argc > 3 ? argv[3] : "stencils";
  const int dimension = argc > 4 ? std::stoi(argv[4]) : 0;
  if (n < 4 || dimension < 0 || dimension > 4)
  {
    std::cerr << "Require n>=4 and dimension=0,1,2,3,4\n";
    return 2;
  }

  try
  {
    if (dimension == 0 || dimension == 1)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_case<1>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 2)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_case<2>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 3)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_case<3>(n, half, reversed, tolerance, stencils);
    if (dimension == 0 || dimension == 4)
      for (bool half : {false, true})
        for (bool reversed : {false, true})
          run_case<4>(n, half, reversed, tolerance, stencils);
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "All dense divergence-form balance-row tests passed.\n";
  return 0;
}
