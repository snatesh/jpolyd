#include <jconormal.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Real = double;
constexpr int solution_degree = 5;
constexpr int diffusion_degree = 3;

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

  Manufactured()
  {
    for (int i = 0; i < D; ++i)
    {
      v[i] = Real(0.16 + 0.035 * i);
      eta[i] = Real(0.12 + 0.025 * i);
      w[i] = Real(0.35 - 0.055 * i);
    }
  }

  Real t(const Point<D>& x) const
  {
    Real result = t0;
    for (int i = 0; i < D; ++i)
      result += v[i] * x[i];
    return result;
  }

  Real A(int r, int s, const Point<D>& x) const
  {
    Real dot = Real(0);
    for (int i = 0; i < D; ++i)
      dot += eta[i] * x[i];
    const Real G = Real(r == s) +
      epsilon * (Real(1) + dot * dot) * w[r] * w[s];
    return t(x) * G;
  }

  Real u(const Point<D>& x) const
  {
    const Real tx = t(x);
    return tx * tx * tx * tx * tx;
  }
};

template<int D>
std::array<Real,D*(D+1)> make_vertices(bool reverse_orientation)
{
  std::array<Real,D*(D+1)> vertices{};
  for (int i = 0; i < D; ++i)
    vertices[i] = Real(0.07 + 0.06 * i);
  for (int j = 0; j < D; ++j)
    for (int i = 0; i < D; ++i)
    {
      const Real Bij = i == j
        ? Real(0.85 + 0.10 * j)
        : Real(0.045 * (i + 1) / (j + 1));
      vertices[i + D * (j+1)] = vertices[i] + Bij;
    }

  if (reverse_orientation)
  {
    if constexpr (D >= 2)
    {
      for (int i = 0; i < D; ++i)
        std::swap(vertices[i + D], vertices[i + 2 * D]);
    }
    else
      vertices[1] = vertices[0] - (vertices[1] - vertices[0]);
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
      x[r] += geom.B[r + D * j] * xhat[j];
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
  std::vector<Real> moments((std::size_t)Mp, Real(0));
  for (int k = 0; k < nq; ++k)
  {
    const auto x = physical_point<D>(
      vertices, geom, X.data() + (std::size_t)k * D);
    const Real value = fun(x);
    for (int j = 0; j < Mp; ++j)
      moments[(std::size_t)j] +=
        V[(std::size_t)k + (std::size_t)nq * j] * weights[k] * value;
  }
  return moments;
}

template<int D>
void test_case(const std::string& stencil_dir, bool flip)
{
  constexpr int n = solution_degree;
  const auto vertices = make_vertices<D>(flip);
  jsimplex::DSimplexGeom<D,Real> geom;
  jsimplex::dsimplex_affine_from_verts<D,Real>(vertices.data(), geom);
  if (!geom.valid)
    throw std::runtime_error("invalid affine geometry");

  std::array<Real,D+1> kappa{};
  for (int r = 0; r <= D; ++r)
    kappa[r] = Real(0.71 + 0.17 * r);

  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);
  jsimplex::MultByQQuadraturePlan<D-1,Real> face_quad(
    pre.kappa_face.data(), n, n + diffusion_degree + 2);
  jsimplex::MultByQQuadratureWorkspace<Real> workspace;

  const int M = pre.M;
  const int kf = pre.kf;
  const int nb = (D + 1) * kf;
  const int pH = n + diffusion_degree - 1;
  const int kFull = jsimplex::Basis<D-1,Real>::dim_Pi(pH);
  const int kHigh = kFull - kf;
  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(diffusion_degree);
  Manufactured<D> problem;

  std::vector<Real> A((std::size_t)D * D * MpA, Real(0));
  for (int r = 0; r < D; ++r)
    for (int s = 0; s < D; ++s)
    {
      const auto projection = project_polynomial<D>(
        vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
        pre.nq_vol, M, diffusion_degree,
        [&](const Point<D>& x) { return problem.A(r, s, x); });
      std::copy(projection.begin(), projection.end(),
        A.begin() + ((std::size_t)r * D + s) * MpA);
    }

  std::array<int,D + 1> global_ids{};
  for (int i = 0; i <= D; ++i)
    global_ids[i] = 7 + 3 * ((i + 2) % (D + 1));
  std::array<int,D + 1> sigmas{};
  for (int f = 0; f < D + 1; ++f)
    sigmas[f] = jsimplex::dsimplex_compute_face_sigma<D>(global_ids.data(), f);

  std::array<jsimplex::DSimplexFaceGeom<D,Real>,D + 1> face_geom{};
  jsimplex::dsimplex_all_physical_face_geometry_colmajor<D,Real>(
    vertices.data(), face_geom.data());
  std::vector<Real> normal_scaled((std::size_t)(D + 1) * D, Real(0));
  for (int f = 0; f < D + 1; ++f)
  {
    if (!face_geom[f].valid)
      throw std::runtime_error("invalid face geometry");
    for (int r = 0; r < D; ++r)
      normal_scaled[(std::size_t)f * D + r] =
        face_geom[f].normal_scaled[(std::size_t)r];
  }

  std::vector<Real> F((std::size_t)nb * M, Real(0));
  jsimplex::jdsimplex_assemble_F_conormal_full_common_face<D,Real>(
    pre, geom, sigmas.data(), normal_scaled.data(),
    diffusion_degree, A.data(), face_quad, workspace, F.data());

  // Only observe high physical flux on these two selected faces;
  // all other local faces intentionally have no high observation rows.
  const int selected_faces[2] = {0, D};
  const int n_selected = 2;
  std::vector<Real> QH((std::size_t)n_selected * kHigh * M, Real(0));
  jsimplex::MultByQQuadraturePlan<D-1,Real> high_quad(
    pre.kappa_face.data(), pH, n + diffusion_degree + 2);
  jsimplex::jdsimplex_assemble_QH_conormal_selected_common_face<D,Real>(
    pre, geom, sigmas.data(), normal_scaled.data(),
    diffusion_degree, A.data(), high_quad, workspace,
    n_selected, selected_faces, QH.empty() ? nullptr : QH.data());

  // Constant A=I must recover the existing normal derivative moment map.
  const Real phi0 = pre.V_vol[0];
  std::vector<Real> identity_A((std::size_t)D * D, Real(0));
  for (int r = 0; r < D; ++r)
    identity_A[(std::size_t)r * D + r] = Real(1) / phi0;
  std::vector<Real> F_identity((std::size_t)nb * M, Real(0));
  jsimplex::jdsimplex_assemble_F_conormal_full_common_face<D,Real>(
    pre, geom, sigmas.data(), normal_scaled.data(),
    0, identity_A.data(), face_quad, workspace, F_identity.data());

  Real id_error2 = 0, id_reference2 = 0;
  for (int f = 0; f < D + 1; ++f)
  {
    const int sigma = sigmas[f];
    const Real refscale = pre.face_ref_scale[(std::size_t)f];
    for (int j = 0; j < M; ++j)
      for (int i = 0; i < kf; ++i)
      {
        Real expected = 0;
        for (int s = 0; s < D; ++s)
          for (int a = 0; a < D; ++a)
          {
            const std::size_t block = (std::size_t)a +
              (std::size_t)D * ((std::size_t)sigma + (std::size_t)pre.nsigma * f);
            const std::size_t idx = (std::size_t)i + (std::size_t)kf *
              ((std::size_t)j + (std::size_t)M * block);
            expected += normal_scaled[(std::size_t)f * D + s]
              * geom.BinvT[(std::size_t)s + (std::size_t)D * a]
              * pre.Fgrad_ref[idx] / refscale;
          }
        const Real actual = F_identity[(std::size_t)f * kf + i + (std::size_t)nb * j];
        id_error2 += (actual - expected) * (actual - expected);
        id_reference2 += expected * expected;
      }
  }
  const Real relative_identity = std::sqrt(id_error2) /
    std::max(Real(1), std::sqrt(id_reference2));

  const auto ucoeff = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, M, n,
    [&](const Point<D>& x) { return problem.u(x); });

  // Independent high-order face quadrature for the exact co-normal flux.
  jsimplex::MultByQQuadraturePlan<D-1,Real> check_quad(
    pre.kappa_face.data(), pH, n + diffusion_degree + 5);
  const int nqf = check_quad.nq;
  Real error2 = 0, reference2 = 0;
  Real high_error2 = 0, high_reference2 = 0;
  for (int f = 0; f < D + 1; ++f)
  {
    const Real scale = face_geom[f].scale;
    std::vector<Real> numerical((std::size_t)kf, Real(0));
    std::vector<Real> expected((std::size_t)kf, Real(0));
    std::vector<Real> expected_high((std::size_t)kHigh, Real(0));
    std::vector<Real> numerical_high((std::size_t)kHigh, Real(0));
    const int selected_slot = f == 0 ? 0 : (f == D ? 1 : -1);
    if (selected_slot >= 0 && kHigh > 0)
      for (int i = 0; i < kHigh; ++i)
        for (int j = 0; j < M; ++j)
          numerical_high[(std::size_t)i] += QH[
            (std::size_t)selected_slot * kHigh + i
            + (std::size_t)(n_selected * kHigh) * j] * ucoeff[j];
    for (int i = 0; i < kf; ++i)
      for (int j = 0; j < M; ++j)
        numerical[(std::size_t)i] +=
          F[(std::size_t)f * kf + i + (std::size_t)nb * j] * ucoeff[j];

    int face_vertices[D];
    jsimplex::dsimplex_face_vertices<D>(f, face_vertices);
    int permutation[D];
    jsimplex::lehmer_index_to_perm<D>(sigmas[f], permutation);
    for (int qf = 0; qf < nqf; ++qf)
    {
      std::array<Real,D> mu{};
      mu[0] = 1;
      for (int a = 1; a < D; ++a)
      {
        mu[a] = check_quad.X[(std::size_t)qf * (D - 1) + (a - 1)];
        mu[0] -= mu[a];
      }
      std::array<Real,D+1> lambda{};
      for (int a = 0; a < D; ++a)
        lambda[face_vertices[a]] = mu[permutation[a]];
      std::array<Real,D> xhat{};
      for (int a = 0; a < D; ++a)
        xhat[a] = lambda[a+1];
      const auto x = physical_point<D>(vertices, geom, xhat.data());
      const Real t = problem.t(x);
      const Real dt = Real(5) * t * t * t * t;
      Real exact_flux = 0;
      for (int r = 0; r < D; ++r)
        for (int s = 0; s < D; ++s)
          exact_flux += face_geom[f].unit_normal[(std::size_t)r]
            * problem.A(r,s,x) * dt * problem.v[s];
      for (int i = 0; i < kf; ++i)
        expected[(std::size_t)i] += scale * check_quad.W[(std::size_t)qf]
          * check_quad.V[(std::size_t)qf + (std::size_t)nqf * i]
          * exact_flux;
      if (selected_slot >= 0)
        for (int i = 0; i < kHigh; ++i)
          expected_high[(std::size_t)i] += scale * check_quad.W[(std::size_t)qf]
            * check_quad.V[(std::size_t)qf + (std::size_t)nqf * (kf + i)]
            * exact_flux;
    }

    // Compare two degree-n projections in the weighted face L2 norm.
    // This does not confuse the projected numerical flux with the
    // higher-degree exact pointwise flux.
    for (int qf = 0; qf < nqf; ++qf)
    {
      Real predicted_value = 0, projected_value = 0;
      for (int i = 0; i < kf; ++i)
      {
        const Real psi = check_quad.V[(std::size_t)qf + (std::size_t)nqf * i];
        predicted_value += (numerical[(std::size_t)i] / scale) * psi;
        projected_value += (expected[(std::size_t)i] / scale) * psi;
      }
      const Real weight = scale * check_quad.W[(std::size_t)qf];
      error2 += weight * (predicted_value - projected_value)
        * (predicted_value - projected_value);
      reference2 += weight * projected_value * projected_value;

      if (selected_slot >= 0)
      {
        Real high_predicted = 0;
        Real high_projected = 0;
        for (int i = 0; i < kHigh; ++i)
        {
          const Real psi = check_quad.V[
            (std::size_t)qf + (std::size_t)nqf * (kf + i)];
          high_predicted += (numerical_high[(std::size_t)i] / scale) * psi;
          high_projected += (expected_high[(std::size_t)i] / scale) * psi;
        }
        high_error2 += weight * (high_predicted - high_projected)
          * (high_predicted - high_projected);
        high_reference2 += weight * high_projected * high_projected;
      }
    }
  }
  const Real relative_high = std::sqrt(high_error2) /
    std::max(Real(1), std::sqrt(high_reference2));
  const Real relative_flux = std::sqrt(error2) /
    std::max(Real(1), std::sqrt(reference2));

  std::cout << "D=" << D << " n=" << n
            << " orientation=" << (flip ? "reversed" : "positive")
            << " A=I map=" << relative_identity
            << " variable conormal L2=" << relative_flux
            << " high co-normal L2=" << relative_high << '\n';
  const Real tol = D >= 4 ? Real(2e-8) : Real(2e-9);
  if (!(relative_identity < tol && relative_flux < tol &&
        relative_high < tol))
    throw std::runtime_error("co-normal face map regression failed");
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
    std::cout << std::scientific << std::setprecision(5);
    run_dimension<1>(stencil_dir);
    if (max_D >= 2) run_dimension<2>(stencil_dir);
    if (max_D >= 3) run_dimension<3>(stencil_dir);
    if (max_D >= 4) run_dimension<4>(stencil_dir);
    std::cout << "PASS: low/high co-normal face maps, n=5, D=1.." << max_D
              << ", both orientations\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
