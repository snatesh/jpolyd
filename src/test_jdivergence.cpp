#include <jdivergence.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Real = double;
constexpr int solution_degree = 5;
constexpr int diffusion_degree = 3;
constexpr int advection_degree = 2;
constexpr int reaction_degree = 1;

template<int D>
using Point = std::array<Real,D>;

template<int D>
struct Manufactured
{
  Point<D> v{};
  Point<D> eta{};
  Point<D> w{};
  Real t0 = Real(1.8);
  Real epsilon = Real(0.07);
  Real sigma = Real(0.4);
  Real vv = Real(0);
  Real ew = Real(0);

  Manufactured()
  {
    for (int i = 0; i < D; ++i)
    {
      v[i] = Real(0.16 + 0.035 * i);
      eta[i] = Real(0.12 + 0.025 * i);
      w[i] = Real(0.35 - 0.055 * i);
      vv += v[i] * v[i];
      ew += eta[i] * w[i];
    }
  }

  Real t(const Point<D>& x) const
  {
    Real result = t0;
    for (int i = 0; i < D; ++i)
      result += v[i] * x[i];
    return result;
  }

  Real s(const Point<D>& x) const
  {
    Real result = Real(0);
    for (int i = 0; i < D; ++i)
      result += eta[i] * x[i];
    return result;
  }

  Real G(int r, int j, const Point<D>& x) const
  {
    const Real sx = s(x);
    return Real(r == j) + epsilon * (Real(1) + sx * sx) * w[r] * w[j];
  }

  Real Gv(int j, const Point<D>& x) const
  {
    Real result = Real(0);
    for (int r = 0; r < D; ++r)
      result += G(j, r, x) * v[r];
    return result;
  }

  Real A(int r, int j, const Point<D>& x) const
  {
    return t(x) * G(r, j, x);
  }

  // Column divergence: (div A)_j = sum_r d_{x_r} A_{rj}.
  Real divA(int j, const Point<D>& x) const
  {
    return Gv(j, x) +
      Real(2) * epsilon * t(x) * s(x) * ew * w[j];
  }

  Real b(int j, const Point<D>& x) const
  {
    const Real tx = t(x);
    return divA(j, x) + Real(4) * Gv(j, x) -
      sigma * tx * tx * v[j] / (Real(5) * vv);
  }

  Real c(const Point<D>& x) const
  {
    return sigma * t(x);
  }

  Real u(const Point<D>& x) const
  {
    const Real tx = t(x);
    return tx * tx * tx * tx * tx;
  }

  Real expanded_A(int r, int j, const Point<D>& x) const
  {
    return -A(r, j, x);
  }

  Real expanded_b(int j, const Point<D>& x) const
  {
    const Real tx = t(x);
    return Real(4) * Gv(j, x) -
      sigma * tx * tx * v[j] / (Real(5) * vv);
  }

  Real expanded_c(const Point<D>& x) const
  {
    return c(x);
  }

  // Continuous expanded terms; these cancel pointwise to give f == 0.
  void expanded_terms(const Point<D>& x,
                      Real& principal, Real& first, Real& zeroth) const
  {
    const Real tx = t(x);
    principal = Real(0);
    first = Real(0);
    for (int r = 0; r < D; ++r)
    {
      for (int j = 0; j < D; ++j)
        principal += expanded_A(r, j, x) *
          Real(20) * tx * tx * tx * v[r] * v[j];
      first += expanded_b(r, x) * Real(5) *
        tx * tx * tx * tx * v[r];
    }
    zeroth = c(x) * u(x);
  }
};

template<int D>
std::array<Real,D*(D+1)> make_vertices(bool reverse_orientation)
{
  // D x (D+1), column-major; nontrivial skew affine map.
  std::array<Real,D*(D+1)> vertices{};
  for (int i = 0; i < D; ++i)
    vertices[i] = Real(0.07 + 0.06 * i);
  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
    {
      const Real Bij = i == j
        ? Real(0.85 + 0.10 * j)
        : Real(0.045 * (i + 1) / (j + 1));
      vertices[i + D * (j+1)] = vertices[i] + Bij;
    }
  }
  if (reverse_orientation)
  {
    if constexpr (D >= 2)
    {
      for (int i = 0; i < D; ++i)
        std::swap(vertices[i + D], vertices[i + 2*D]);
    }
    else
    {
      vertices[1] = vertices[0] - (vertices[1] - vertices[0]);
    }
  }
  return vertices;
}

template<int D>
Point<D> physical_point(
  const std::array<Real,D*(D+1)>& vertices,
  const jsimplex::DSimplexGeom<D,Real>& geom,
  const Real* xhat)
{
  Point<D> x{};
  for (int r = 0; r < D; ++r)
  {
    x[r] = vertices[r];
    for (int j = 0; j < D; ++j)
      x[r] += geom.B[r + D*j] * xhat[j];
  }
  return x;
}

template<int D, class Fun>
std::vector<Real> project_polynomial(
  const std::array<Real,D*(D+1)>& vertices,
  const jsimplex::DSimplexGeom<D,Real>& geom,
  const std::vector<Real>& X,
  const std::vector<Real>& weights,
  const std::vector<Real>& V,
  int nq, int M, int p,
  Fun&& fun)
{
  const int Mp = jsimplex::Basis<D,Real>::dim_Pi(p);
  std::vector<Real> moments(Mp, Real(0));
  for (int k = 0; k < nq; ++k)
  {
    const auto x = physical_point<D>(vertices, geom, X.data() + (std::size_t)k*D);
    const Real val = fun(x);
    for (int j = 0; j < Mp; ++j)
      moments[j] += V[(std::size_t)k + (std::size_t)nq*j] * weights[k] * val;
  }
  return moments;
}

// Compare a packed coefficient block with independent kappa+2 projections,
// and report the weighted L2 error by evaluating at the residual quadrature.
template<int D, class Fun>
void check_field(
  const std::string& name,
  const Real* coeffs, int p,
  const std::array<Real,D*(D+1)>& vertices,
  const jsimplex::DSimplexGeom<D,Real>& geom,
  const jsimplex::RefSimplexPrecomp<D,Real>& pre,
  Fun&& exact,
  Real& maximum_weighted_error,
  Real& maximum_modal_error)
{
  const int Mp = jsimplex::Basis<D,Real>::dim_Pi(p);
  const auto expected = project_polynomial<D>(
    vertices, geom, pre.X_res, pre.W_res, pre.V_res,
    pre.nq_vol, pre.M, p, exact);

  Real max_delta = Real(0);
  Real modal_scale = Real(0);
  for (int j = 0; j < Mp; ++j)
  {
    max_delta = std::max(max_delta, std::abs(coeffs[j] - expected[j]));
    modal_scale = std::max(modal_scale, std::abs(expected[j]));
  }
  const Real modal_err = max_delta / std::max(Real(1), modal_scale);
  maximum_modal_error = std::max(maximum_modal_error, modal_err);

  Real err2 = Real(0);
  Real exact2 = Real(0);
  Real total_weight = Real(0);
  for (int k = 0; k < pre.nq_vol; ++k)
  {
    Real numerical = Real(0);
    for (int j = 0; j < Mp; ++j)
      numerical += coeffs[j] * pre.V_res[(std::size_t)k + (std::size_t)pre.nq_vol*j];
    const auto x = physical_point<D>(vertices, geom, pre.X_res.data() + (std::size_t)k*D);
    const Real truth = exact(x);
    err2 += pre.W_res[k] * (numerical - truth) * (numerical - truth);
    exact2 += pre.W_res[k] * truth * truth;
    total_weight += pre.W_res[k];
  }
  const Real normalized = std::sqrt(std::max(Real(0),err2)) /
    std::max(std::sqrt(std::max(Real(0),exact2)),
             std::sqrt(total_weight));
  maximum_weighted_error = std::max(maximum_weighted_error, normalized);
  std::cout << "      " << name << ": weighted L2 = " << normalized
            << ", modal = " << modal_err << '\n';
}

template<int D>
void test_case(const std::string& stencil_dir, bool flip)
{
  constexpr int n = solution_degree;
  const auto vertices = make_vertices<D>(flip);
  jsimplex::DSimplexGeom<D,Real> geom;
  jsimplex::dsimplex_affine_from_verts<D,Real>(vertices.data(), geom);
  if (!geom.valid)
    throw std::runtime_error("invalid test geometry");
  if (flip != (geom.detB < Real(0)))
    throw std::runtime_error("unexpected affine orientation");

  std::array<Real,D+1> kappa{};
  for (int j = 0; j <= D; ++j)
    kappa[j] = Real(0.71 + 0.17*j);

  // Exact for every projection used below, including degree-5 u * trial basis.
  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);

  const auto deg = jsimplex::divergence_form_expanded_degrees(
    n, diffusion_degree, advection_degree, reaction_degree);
  if (deg.p2 != 3 || deg.p1 != 2 || deg.p0 != 1)
    throw std::runtime_error("incorrect expanded degree rule");

  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(diffusion_degree);
  const int MpB = jsimplex::Basis<D,Real>::dim_Pi(advection_degree);
  const int MpC = jsimplex::Basis<D,Real>::dim_Pi(reaction_degree);
  const int M = pre.M;
  Manufactured<D> problem;

  std::vector<Real> A((std::size_t)D*D*MpA);
  std::vector<Real> b((std::size_t)D*MpB);
  for (int r = 0; r < D; ++r)
  {
    for (int j = 0; j < D; ++j)
    {
      const auto coeff = project_polynomial<D>(
        vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
        pre.nq_vol, pre.M, diffusion_degree,
        [&](const Point<D>& x) { return problem.A(r,j,x); });
      std::copy(coeff.begin(), coeff.end(),
                A.begin() + ((std::size_t)r*D+j)*MpA);
    }
    const auto coeff = project_polynomial<D>(
      vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
      pre.nq_vol, pre.M, advection_degree,
      [&](const Point<D>& x) { return problem.b(r,x); });
    std::copy(coeff.begin(), coeff.end(), b.begin() + (std::size_t)r*MpB);
  }
  const auto c = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, pre.M, reaction_degree,
    [&](const Point<D>& x) { return problem.c(x); });

  const auto expanded = jsimplex::expand_divergence_form_element<D,Real>(
    pre, geom, diffusion_degree, A.data(), advection_degree, b.data(),
    reaction_degree, c.data());
  if (expanded.degrees.p2 != 3 || expanded.degrees.p1 != 2 ||
      expanded.degrees.p0 != 1)
    throw std::runtime_error("adapter returned wrong degrees");

  std::cout << "D=" << D << ", n=" << n
            << ", orientation=" << (flip ? "reversed" : "positive")
            << ", detB=" << geom.detB << '\n';

  Real max_l2 = Real(0), max_modal = Real(0);
  for (int r = 0; r < D; ++r)
  {
    for (int j = 0; j < D; ++j)
      check_field<D>("a[" + std::to_string(r) + "," + std::to_string(j) + "]",
        expanded.A.data() + ((std::size_t)r*D+j)*MpA, diffusion_degree,
        vertices, geom, pre,
        [&](const Point<D>& x) { return problem.expanded_A(r,j,x); },
        max_l2, max_modal);
    check_field<D>("b[" + std::to_string(r) + "]",
      expanded.b.data() + (std::size_t)r*MpB, advection_degree,
      vertices, geom, pre,
      [&](const Point<D>& x) { return problem.expanded_b(r,x); },
      max_l2, max_modal);
  }
  check_field<D>("c", expanded.c.data(), reaction_degree,
    vertices, geom, pre,
    [&](const Point<D>& x) { return problem.expanded_c(x); },
    max_l2, max_modal);

  // Independently verify the homogeneous strong-form identity at quadrature.
  Real identity_err2 = Real(0), terms2 = Real(0);
  for (int k = 0; k < pre.nq_vol; ++k)
  {
    const auto x = physical_point<D>(vertices, geom, pre.X_res.data() + (std::size_t)k*D);
    Real p, a, z;
    problem.expanded_terms(x, p, a, z);
    identity_err2 += pre.W_res[k] * (p+a+z)*(p+a+z);
    terms2 += pre.W_res[k] * (p*p+a*a+z*z);
  }
  const Real identity_error = std::sqrt(identity_err2 /
    std::max(Real(1), terms2));

  // Project u=t^5 into base kappa and apply production elliptic assembly.
  const auto u_coeff = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, pre.M, n,
    [&](const Point<D>& x) { return problem.u(x); });

  jsimplex::EllipticPlan<D,Real> plan(
    pre, expanded.degrees, true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature,
    false);
  jsimplex::EllipticDenseWorkspace<D,Real> work(plan);
  std::vector<Real> L((std::size_t)M*M, Real(0));
  jsimplex::jdsimplex_assemble_elliptic_L_int_dag<D,Real>(
    pre, geom, plan, expanded.view(), work, L.data());

  Real residual_inf = Real(0), residual_norm2 = Real(0);
  for (int i = 0; i < M; ++i)
  {
    Real ri = Real(0);
    for (int j = 0; j < M; ++j)
      ri += L[(std::size_t)i + (std::size_t)M*j] * u_coeff[j];
    residual_inf = std::max(residual_inf, std::abs(ri));
    residual_norm2 += ri*ri;
  }
  const Real projected_residual = std::sqrt(residual_norm2) /
    (geom.detBabs * std::max(Real(1), std::sqrt(terms2)));

  std::cout << "    largest expanded coefficient weighted L2 error: "
            << max_l2 << '\n';
  std::cout << "    largest expanded modal error: " << max_modal << '\n';
  std::cout << "    continuous f=0 identity weighted L2 error: "
            << identity_error << '\n';
  std::cout << "    projected L_int*u residual: " << projected_residual
            << " (raw infinity norm " << residual_inf << ")\n";

  // Loose enough for high-D quadrature and reference operator conditioning,
  // strict enough to detect index/sign/degree mistakes in the adapter.
  const Real tol = D >= 4 ? Real(2e-8) : Real(2e-9);
  if (!(max_l2 < tol && max_modal < tol &&
        identity_error < tol && projected_residual < tol))
    throw std::runtime_error("degree-5 divergence adapter regression failed");

  // Additional check: constant diffusion must generate no first-order term.
  const int MpConstant = jsimplex::Basis<D,Real>::dim_Pi(0);
  std::vector<Real> A0((std::size_t)D*D*MpConstant, Real(0));
  const Real phi0 = pre.V_vol[0];
  for (int r = 0; r < D; ++r)
    for (int j = 0; j < D; ++j)
      A0[(std::size_t)r*D+j] = (r == j ? Real(1.3) : Real(0.08)) / phi0;

  const auto const_exp = jsimplex::expand_divergence_form_element<D,Real>(
    pre, geom, 0, A0.data(), -1, nullptr, -1, nullptr);
  if (const_exp.degrees.p1 != -1 || !const_exp.b.empty() ||
      !const_exp.c.empty())
    throw std::runtime_error("constant diffusion yielded spurious b/c");
  std::cout << "    constant diffusion sanity check: PASS\n";
}

template<int D>
void run_dimension(const std::string& stencil_dir)
{
  test_case<D>(stencil_dir, false);
  test_case<D>(stencil_dir, true);
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const std::string stencil_dir = argc > 1 ? argv[1] : "stencils";
    const int max_D = argc > 2 ? std::stoi(argv[2]) : 4;
    if (max_D < 1 || max_D > 4)
      throw std::invalid_argument("max_D must be in 1..4");

    std::cout << std::scientific << std::setprecision(6);
    run_dimension<1>(stencil_dir);
    if (max_D >= 2) run_dimension<2>(stencil_dir);
    if (max_D >= 3) run_dimension<3>(stencil_dir);
    if (max_D >= 4) run_dimension<4>(stencil_dir);
    std::cout << "PASS: divergence-form adapter, D=1.." << max_D
              << ", n=5, both orientations\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
