#include <jdivergence.hh>
#include <jnode.hh>
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


// Relative Euclidean error, normalized by max(1, ||reference||_2).
Real relative_error(const std::vector<Real>& actual,
                    const std::vector<Real>& reference)
{
  if (actual.size() != reference.size())
    throw std::runtime_error("vector size mismatch");
  Real error2 = 0, scale2 = 0;
  for (std::size_t i = 0; i < actual.size(); ++i)
  {
    const Real delta = actual[i] - reference[i];
    error2 += delta * delta;
    scale2 += reference[i] * reference[i];
  }
  return std::sqrt(error2) / std::max(Real(1), std::sqrt(scale2));
}

// Column-major matrix-vector product y = A*x.
std::vector<Real> matvec(const std::vector<Real>& A,
                         int nrow, int ncol,
                         const std::vector<Real>& x)
{
  if (A.size() != (std::size_t)nrow * ncol ||
      x.size() != (std::size_t)ncol)
    throw std::runtime_error("matvec dimensions do not agree");

  std::vector<Real> y((std::size_t)nrow, Real(0));
  for (int j = 0; j < ncol; ++j)
    for (int i = 0; i < nrow; ++i)
      y[(std::size_t)i] += A[(std::size_t)i + (std::size_t)nrow * j] * x[(std::size_t)j];
  return y;
}

// Check S=(F+tau*T)Ulam-tau*I and Gf=(F+tau*T)Uf
// without using Leaf::apply_flux_columns or the Node materializer.
template<int D>
std::pair<Real,Real> response_errors(
  const jsimplex::Leaf<D,Real>& leaf,
  const jsimplex::Node<D,Real>& node)
{
  const int nb = leaf.nb;
  const int M = leaf.M;
  const int mR = leaf.m_int;

  std::vector<Real> expected_S((std::size_t)nb * nb, Real(0));
  std::vector<Real> expected_Gf((std::size_t)nb * mR, Real(0));

  for (int j = 0; j < nb; ++j)
    for (int k = 0; k < M; ++k)
    {
      const Real response = node.Ulam[(std::size_t)k + (std::size_t)M * j];
      for (int i = 0; i < nb; ++i)
        expected_S[(std::size_t)i + (std::size_t)nb * j] +=
          (leaf.F[(std::size_t)i + (std::size_t)nb * k] +
           leaf.tau_rows[(std::size_t)i] *
           leaf.T[(std::size_t)i + (std::size_t)nb * k]) * response;
    }

  for (int i = 0; i < nb; ++i)
    expected_S[(std::size_t)i + (std::size_t)nb * i] -= leaf.tau_rows[(std::size_t)i];

  for (int j = 0; j < mR; ++j)
    for (int k = 0; k < M; ++k)
    {
      const Real response = node.Uf[(std::size_t)k + (std::size_t)M * j];
      for (int i = 0; i < nb; ++i)
        expected_Gf[(std::size_t)i + (std::size_t)nb * j] +=
          (leaf.F[(std::size_t)i + (std::size_t)nb * k] +
           leaf.tau_rows[(std::size_t)i] *
           leaf.T[(std::size_t)i + (std::size_t)nb * k]) * response;
    }

  return {relative_error(node.S, expected_S),
          relative_error(node.Gf, expected_Gf)};
}

template<int D>
void test_case(const std::string& stencil_dir, bool reversed)
{
  constexpr int n = solution_degree;
  constexpr Real tau_C = Real(1);
  const auto vertices = make_vertices<D>(reversed);

  jsimplex::Mesh<D,Real> mesh;
  std::array<int,D+1> global_ids{};
  for (int i = 0; i <= D; ++i)
  {
    global_ids[(std::size_t)i] = 17 + 3 * ((i + 2) % (D + 1));
    Point<D> x{};
    for (int r = 0; r < D; ++r)
      x[(std::size_t)r] = vertices[(std::size_t)r + (std::size_t)D * i];
    mesh.add_vertex(global_ids[(std::size_t)i], x);
  }
  mesh.add_simplex(global_ids);
  mesh.build();
  const auto& elem = mesh.element(0);
  const auto& geom = elem.geom;
  if (reversed != (geom.detB < Real(0)))
    throw std::runtime_error("unexpected element orientation");

  std::array<Real,D+1> kappa{};
  for (int j = 0; j <= D; ++j)
    kappa[(std::size_t)j] = Real(0.71 + 0.17 * j);

  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);
  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(diffusion_degree);
  const int MpB = jsimplex::Basis<D,Real>::dim_Pi(advection_degree);
  const int M = pre.M;
  Manufactured<D> problem;

  std::vector<Real> A((std::size_t)D * D * MpA);
  std::vector<Real> b((std::size_t)D * MpB);
  for (int r = 0; r < D; ++r)
  {
    for (int s = 0; s < D; ++s)
    {
      const auto coeff = project_polynomial<D>(
        vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
        pre.nq_vol, M, diffusion_degree,
        [&](const Point<D>& x) { return problem.A(r,s,x); });
      std::copy(coeff.begin(), coeff.end(),
        A.begin() + ((std::size_t)r * D + s) * MpA);
    }
    const auto coeff = project_polynomial<D>(
      vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
      pre.nq_vol, M, advection_degree,
      [&](const Point<D>& x) { return problem.b(r,x); });
    std::copy(coeff.begin(), coeff.end(),
      b.begin() + (std::size_t)r * MpB);
  }
  const auto c = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, M, reaction_degree,
    [&](const Point<D>& x) { return problem.c(x); });

  const auto expanded = jsimplex::expand_divergence_form_element<D,Real>(
    pre, geom, diffusion_degree, A.data(),
    advection_degree, b.data(), reaction_degree, c.data());

  jsimplex::EllipticPlan<D,Real> plan(
    pre, expanded.degrees, true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature,
    false);

  jsimplex::MultByQQuadraturePlan<D-1,Real> face_quad(
    pre.kappa_face.data(), n, n + diffusion_degree + 2);
  jsimplex::MultByQQuadratureWorkspace<Real> face_work;

  typename jsimplex::Leaf<D,Real>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;

  // Factory under test: assemble elliptic L and replace F before
  // generating the Ulam, Uf, S, Gf response panels.
  jsimplex::EllipticDenseWorkspace<D,Real> work_A(plan);
  jsimplex::Leaf<D,Real> conormal_leaf;
  auto conormal_node =
    jsimplex::make_divergence_form_homogeneous_leaf_node<D,Real>(
      mesh, 0, pre, plan, expanded.view(), diffusion_degree, A.data(),
      face_quad, face_work, work_A, conormal_leaf, options, tau_C);

  // Control factory, identical elliptic volume L and tau stabilization,
  // but with the existing ordinary normal-derivative F.
  jsimplex::EllipticDenseWorkspace<D,Real> work_N(plan);
  jsimplex::Leaf<D,Real> normal_leaf;
  auto normal_node = jsimplex::make_elliptic_homogeneous_leaf_node<D,Real>(
    mesh, 0, pre, plan, expanded.view(), work_N, normal_leaf, options, tau_C);

  conormal_node.validate();
  normal_node.validate();
  const int nb = conormal_node.nb();
  if (nb != (D+1)*pre.kf || conormal_node.source_dim != M)
    throw std::runtime_error("invalid leaf node dimensions");

  // Interior solve and penalty are independent of the flux observation.
  const Real eL = relative_error(conormal_leaf.L, normal_leaf.L);
  const Real eT = relative_error(conormal_leaf.T, normal_leaf.T);
  const Real eTau = relative_error(conormal_leaf.tau_rows, normal_leaf.tau_rows);
  const Real eATau = relative_error(conormal_leaf.A_tau, normal_leaf.A_tau);
  const Real eUlam = relative_error(conormal_node.Ulam, normal_node.Ulam);
  const Real eUf = relative_error(conormal_node.Uf, normal_node.Uf);

  // The installed flux map must agree with the independently tested builder.
  std::vector<Real> independently_assembled_F((std::size_t)nb * M);
  std::array<Real,(D+1)*D> normal_scaled{};
  for (int f = 0; f < D+1; ++f)
    for (int r = 0; r < D; ++r)
      normal_scaled[(std::size_t)f * D + r] =
        elem.face_geom[(std::size_t)f].normal_scaled[(std::size_t)r];
  jsimplex::jdsimplex_assemble_F_conormal_full_common_face<D,Real>(
    pre, geom, elem.face_sigma_index.data(), normal_scaled.data(),
    diffusion_degree, A.data(), face_quad, face_work,
    independently_assembled_F.data());
  const Real eF = relative_error(conormal_leaf.F, independently_assembled_F);

  const auto [eS, eGf] = response_errors(conormal_leaf, conormal_node);

  // The entire change in S and Gf must come from (F_A - F_normal).
  std::vector<Real> deltaF((std::size_t)nb * M, Real(0));
  for (std::size_t k = 0; k < deltaF.size(); ++k)
    deltaF[k] = conormal_leaf.F[k] - normal_leaf.F[k];
  std::vector<Real> predicted_deltaS((std::size_t)nb * nb, Real(0));
  std::vector<Real> predicted_deltaGf((std::size_t)nb * M, Real(0));
  for (int j = 0; j < nb; ++j)
    for (int k = 0; k < M; ++k)
      for (int i = 0; i < nb; ++i)
        predicted_deltaS[(std::size_t)i + (std::size_t)nb * j] +=
          deltaF[(std::size_t)i + (std::size_t)nb * k] *
          conormal_node.Ulam[(std::size_t)k + (std::size_t)M * j];
  for (int j = 0; j < M; ++j)
    for (int k = 0; k < M; ++k)
      for (int i = 0; i < nb; ++i)
        predicted_deltaGf[(std::size_t)i + (std::size_t)nb * j] +=
          deltaF[(std::size_t)i + (std::size_t)nb * k] *
          conormal_node.Uf[(std::size_t)k + (std::size_t)M * j];
  std::vector<Real> observed_deltaS = conormal_node.S;
  std::vector<Real> observed_deltaGf = conormal_node.Gf;
  for (std::size_t k = 0; k < observed_deltaS.size(); ++k)
    observed_deltaS[k] -= normal_node.S[k];
  for (std::size_t k = 0; k < observed_deltaGf.size(); ++k)
    observed_deltaGf[k] -= normal_node.Gf[k];
  const Real eDeltaS = relative_error(observed_deltaS, predicted_deltaS);
  const Real eDeltaGf = relative_error(observed_deltaGf, predicted_deltaGf);

  // Polynomial manufactured solution: f=0, so the exact volume
  // coefficients must be obtained from the exact trace lambda=T*c_star.
  const auto exact_u = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, M, n,
    [&](const Point<D>& x) { return problem.u(x); });
  const auto lambda = matvec(conormal_leaf.T, nb, M, exact_u);
  const auto recovered_u = matvec(conormal_node.Ulam, M, nb, lambda);
  const auto exact_flux = matvec(conormal_leaf.F, nb, M, exact_u);
  const auto root_flux = matvec(conormal_node.S, nb, nb, lambda);
  const Real eU = relative_error(recovered_u, exact_u);
  const Real eFlux = relative_error(root_flux, exact_flux);

  // Unsupported modes must fail immediately, before any leaf reset.
  if constexpr (D == 1)
  {
    auto rejected_options = options;
    rejected_options.operator_mode = jsimplex::LeafOperatorMode::MatrixFree;
    jsimplex::EllipticDenseWorkspace<D,Real> rejected_work(plan);
    jsimplex::Leaf<D,Real> rejected_leaf;
    bool rejected = false;
    try
    {
      (void)jsimplex::make_divergence_form_homogeneous_leaf_node<D,Real>(
        mesh, 0, pre, plan, expanded.view(), diffusion_degree, A.data(),
        face_quad, face_work, rejected_work, rejected_leaf,
        rejected_options, tau_C);
    }
    catch (const std::invalid_argument&)
    {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("non-Dense co-normal factory was not rejected");
  }

  const Real max_operator = std::max({
    eL, eT, eTau, eATau, eUlam, eUf, eF,
    eS, eGf, eDeltaS, eDeltaGf});
  const Real tol = D >= 4 ? Real(2e-8) : Real(2e-9);
  std::cout << "D=" << D
            << " n=" << n
            << " orientation=" << (reversed ? "reversed" : "positive")
            << "  maps=" << max_operator
            << "  u=" << eU
            << "  conormal_flux=" << eFlux
            << "  deltaS=" << eDeltaS
            << "  deltaGf=" << eDeltaGf << '\n';
  if (!(max_operator < tol && eU < tol && eFlux < tol))
    throw std::runtime_error("divergence-form leaf-node regression failed");
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
    const int max_D = argc > 2 ? std::stoi(argv[2]) : 3;
    if (max_D < 1 || max_D > 4)
      throw std::invalid_argument("max_D must be in 1..4");
    std::cout << std::scientific << std::setprecision(5);
    run_dimension<1>(stencil_dir);
    if (max_D >= 2) run_dimension<2>(stencil_dir);
    if (max_D >= 3) run_dimension<3>(stencil_dir);
    if (max_D >= 4) run_dimension<4>(stencil_dir);
    std::cout << "PASS: divergence-form Dense leaf-node response, n=5, D=1.."
              << max_D << ", both orientations\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
