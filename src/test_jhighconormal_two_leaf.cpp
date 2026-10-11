#include <jconormal.hh>
#include <jmerge.hh>

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


// Multiplication of a column-major matrix by a second column-major matrix.
std::vector<Real> multiply(const std::vector<Real>& A, int m, int k,
                           const std::vector<Real>& B, int n)
{
  if (A.size() != (std::size_t)m*k || B.size() != (std::size_t)k*n)
    throw std::runtime_error("multiply: invalid dimensions");
  std::vector<Real> C((std::size_t)m*n, Real(0));
  for (int j = 0; j < n; ++j)
    for (int l = 0; l < k; ++l)
      for (int i = 0; i < m; ++i)
        C[(std::size_t)i + (std::size_t)m*j] +=
          A[(std::size_t)i + (std::size_t)m*l] *
          B[(std::size_t)l + (std::size_t)k*j];
  return C;
}


template<int D>
void test_case(const std::string& stencil_dir, bool reversed, bool only_A)
{
  constexpr int n = solution_degree;
  constexpr int pA = diffusion_degree;
  const auto vertices = make_vertices<D>(false);

  jsimplex::Mesh<D,Real> mesh;
  for (int j = 0; j <= D; ++j)
  {
    Point<D> x{};
    for (int r = 0; r < D; ++r)
      x[r] = vertices[(std::size_t)r + (std::size_t)D*j];
    mesh.add_vertex(j, x);
  }
  Point<D> xnew{};
  for (int r = 0; r < D; ++r)
  {
    xnew[r] = vertices[(std::size_t)r];
    for (int j = 0; j < D; ++j)
      xnew[r] += vertices[(std::size_t)r + (std::size_t)D*(j+1)]
        - vertices[(std::size_t)r];
  }
  mesh.add_vertex(D+1, xnew);
  std::array<int,D+1> idsA{}, idsB{};
  for (int j = 0; j <= D; ++j)
    idsA[(std::size_t)j] = j;
  for (int j = 0; j < D; ++j)
    idsB[(std::size_t)j] = j+1;
  idsB[(std::size_t)D] = D+1;
  if (reversed)
  {
    std::swap(idsA[1], idsA[2]);
    std::swap(idsB[1], idsB[2]);
  }
  mesh.add_simplex(idsA);
  mesh.add_simplex(idsB);
  mesh.build();
  if (mesh.interior_face_keys().size() != 1)
    throw std::runtime_error("expected one shared face");

  std::array<Real,D+1> kappa{};
  for (int j = 0; j <= D; ++j)
    kappa[(std::size_t)j] = Real(0.71 + 0.17*j);
  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);
  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(pA);
  const int kH = jsimplex::Basis<D-1,Real>::dim_Pi(n+pA-1) - pre.kf;
  Manufactured<D> problem;
  jsimplex::MultByQQuadraturePlan<D-1,Real> high_quad(
    pre.kappa_face.data(), n+pA-1, n+pA+2);
  jsimplex::MultByQQuadratureWorkspace<Real> high_work;

  using Obs = jsimplex::HighConormalObservation<D,Real>;
  typename jsimplex::Leaf<D,Real>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;

  std::array<jsimplex::Leaf<D,Real>,2> leaves;
  std::array<jsimplex::Node<D,Real>,2> nodes;
  std::array<Obs,2> observations;
  std::array<std::vector<Real>,2> sources;
  std::array<std::vector<Real>,2> QH;

  for (int e = 0; e < 2; ++e)
  {
    const auto& elem = mesh.element(e);
    nodes[(std::size_t)e] =
      jsimplex::make_poisson_homogeneous_leaf_node<D,Real>(
        mesh, e, pre, leaves[(std::size_t)e], options, Real(10));

    std::vector<Real> A((std::size_t)D*D*MpA);
    for (int r = 0; r < D; ++r)
      for (int s = 0; s < D; ++s)
      {
        const auto coeff = project_polynomial<D>(
          elem.V_phys, elem.geom, pre.X_vol, pre.W_vol, pre.V_vol,
          pre.nq_vol, pre.M, pA,
          [&](const Point<D>& x) { return problem.A(r,s,x); });
        std::copy(coeff.begin(), coeff.end(),
          A.begin() + ((std::size_t)r*D + s)*MpA);
      }

    std::vector<int> selected;
    if (e == 0 || !only_A)
    {
      for (int f = D; f >= 0; --f)
        if (mesh.is_boundary_face(elem.face_keys[(std::size_t)f]))
          selected.push_back(f);
    }

    std::vector<Real> H, Wf;
    jsimplex::build_dense_leaf_high_conormal_response<D,Real>(
      leaves[(std::size_t)e], nodes[(std::size_t)e], pA, A.data(),
      high_quad, high_work, selected, H, Wf);
    observations[(std::size_t)e] =
      jsimplex::make_high_conormal_leaf_observation<D,Real>(
        nodes[(std::size_t)e], kH, selected, H, Wf);

    const int nh = observations[(std::size_t)e].nrows();
    QH[(std::size_t)e].assign((std::size_t)nh * pre.M, Real(0));
    if (nh > 0)
      jsimplex::jdsimplex_assemble_QH_conormal_selected_common_face<D,Real>(
        pre, elem.geom, leaves[(std::size_t)e].face_sigma_index.data(),
        leaves[(std::size_t)e].normal_scaled.data(), pA, A.data(),
        high_quad, high_work, static_cast<int>(selected.size()),
        selected.data(), QH[(std::size_t)e].data());

    sources[(std::size_t)e].resize((std::size_t)nodes[(std::size_t)e].source_dim);
    for (int j = 0; j < nodes[(std::size_t)e].source_dim; ++j)
      sources[(std::size_t)e][(std::size_t)j] =
        Real(0.32)*std::cos(Real(0.27)*(j+1)*(e+1));
    jsimplex::set_leaf_source(nodes[(std::size_t)e],
      sources[(std::size_t)e].data());
    jsimplex::update_high_conormal_leaf_source(
      observations[(std::size_t)e], nodes[(std::size_t)e],
      sources[(std::size_t)e].data());
  }

  auto parent = jsimplex::merge_nodes<D,Real>(nodes[0], nodes[1], 0, 1);
  auto root_obs = jsimplex::merge_high_conormal_observations<D,Real>(
    parent, nodes[0], observations[0], nodes[1], observations[1]);

  // Selected high-flux rows must follow the parent boundary-face order,
  // even though the leaf selections were supplied in reverse local order.
  std::vector<typename Obs::FaceKey> expected_faces;
  for (const auto& key : parent.boundary_faces)
  {
    const bool selected_A =
      std::find(observations[0].faces.begin(), observations[0].faces.end(), key)
        != observations[0].faces.end();
    const bool selected_B =
      std::find(observations[1].faces.begin(), observations[1].faces.end(), key)
        != observations[1].faces.end();
    if (selected_A || selected_B)
      expected_faces.push_back(key);
  }
  if (expected_faces != root_obs.faces)
    throw std::runtime_error("root observation face ordering mismatch");

  std::vector<Real> lambda_root((std::size_t)parent.nb());
  for (int i = 0; i < parent.nb(); ++i)
    lambda_root[(std::size_t)i] = Real(0.17)*std::sin(Real(0.41)*(i+1));

  const auto check = [&](const char* phase) -> Real {
    std::vector<Real> lambda_A, lambda_B;
    jsimplex::merge_reconstruct_child_traces<D,Real>(
      parent, nodes[0], nodes[1], lambda_root.data(),
      lambda_A, lambda_B);
    std::array<std::vector<Real>,2> lambda = {lambda_A, lambda_B};
    std::array<std::vector<Real>,2> direct;
    Real interface_flux_error = Real(0);
    std::array<std::vector<Real>,2> mu;
    for (int e = 0; e < 2; ++e)
    {
      std::vector<Real> c((std::size_t)pre.M);
      jsimplex::reconstruct_leaf_volume<D,Real>(
        nodes[(std::size_t)e], lambda[(std::size_t)e].data(), c.data());
      direct[(std::size_t)e] = matvec(QH[(std::size_t)e],
        observations[(std::size_t)e].nrows(), pre.M, c);
      mu[(std::size_t)e].resize((std::size_t)nodes[(std::size_t)e].nb());
      jsimplex::apply_node_dtn<D,Real>(
        nodes[(std::size_t)e], lambda[(std::size_t)e].data(),
        mu[(std::size_t)e].data());
    }
    for (int i = 0; i < parent.merge.nI; ++i)
      interface_flux_error = std::max(interface_flux_error,
        std::abs(mu[0][(std::size_t)parent.merge.A_I_idx[(std::size_t)i]]
          + mu[1][(std::size_t)parent.merge.B_I_idx[(std::size_t)i]]));

    std::vector<Real> direct_root((std::size_t)root_obs.nrows());
    for (int f = 0; f < static_cast<int>(root_obs.faces.size()); ++f)
    {
      int owner = -1;
      int slot = -1;
      for (int e = 0; e < 2; ++e)
      {
        const auto& faces = observations[(std::size_t)e].faces;
        const auto it = std::find(faces.begin(), faces.end(), root_obs.faces[(std::size_t)f]);
        if (it != faces.end())
        {
          owner = e;
          slot = static_cast<int>(it - faces.begin());
          break;
        }
      }
      if (owner < 0)
        throw std::runtime_error("selected face missing from leaf");
      for (int j = 0; j < kH; ++j)
        direct_root[(std::size_t)(f*kH+j)] =
          direct[(std::size_t)owner][(std::size_t)(slot*kH+j)];
    }

    auto propagated = matvec(root_obs.H, root_obs.nrows(), parent.nb(), lambda_root);
    for (int i = 0; i < root_obs.nrows(); ++i)
      propagated[(std::size_t)i] += root_obs.z[(std::size_t)i];
    const Real high_error = relative_error(propagated, direct_root);
    if (!(high_error < Real(2e-9) && interface_flux_error < Real(2e-9)))
      throw std::runtime_error(std::string("merge high response mismatch at ") + phase);
    std::cout << "D=" << D << " orientation=" << (reversed ? "reversed" : "positive")
      << " selected=" << (only_A ? "A-only" : "both")
      << " phase=" << phase
      << " high=" << high_error << " interface=" << interface_flux_error << '\n';
    return high_error;
  };

  check("initial");
  const auto previous_H = root_obs.H;
  const auto previous_z = root_obs.z;

  // Change ONLY the B-source. In A-only mode, B has no high-observation rows,
  // but its changed low-order interface flux must change the root high offset.
  for (int j = 0; j < nodes[1].source_dim; ++j)
    sources[1][(std::size_t)j] += Real(0.5)*std::sin(Real(0.33)*(j+1));
  jsimplex::set_leaf_source(nodes[1], sources[1].data());
  jsimplex::update_high_conormal_leaf_source(
    observations[1], nodes[1], sources[1].data());
  jsimplex::update_merge_source<D,Real>(parent, nodes[0], nodes[1]);
  jsimplex::update_high_conormal_merge_source<D,Real>(
    parent, nodes[0], observations[0], nodes[1], observations[1], root_obs);
  if (root_obs.H != previous_H)
    throw std::runtime_error("H changed when only sources changed");
  check("source-refresh");

  if (only_A)
  {
    Real dz = Real(0);
    for (int i = 0; i < root_obs.nrows(); ++i)
      dz = std::max(dz, std::abs(root_obs.z[(std::size_t)i] - previous_z[(std::size_t)i]));
    std::cout << "  inactive-child source influence (max abs delta z)=" << dz << '\n';
    if (!(dz > Real(1e-16)))
      throw std::runtime_error("inactive child's source did not influence active high flux");
  }
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const std::string stencil_dir = argc > 1 ? argv[1] : "stencils";
    const int max_D = argc > 2 ? std::stoi(argv[2]) : 3;
    if (max_D < 2 || max_D > 4)
      throw std::invalid_argument("max_D must be in 2..4");
    std::cout << std::scientific << std::setprecision(5);
    for (bool reversed : {false,true})
      for (bool only_A : {false,true})
      {
        test_case<2>(stencil_dir, reversed, only_A);
        if (max_D >= 3) test_case<3>(stencil_dir, reversed, only_A);
        if (max_D >= 4) test_case<4>(stencil_dir, reversed, only_A);
      }
    std::cout << "PASS: two-leaf selective high co-normal propagation (D=2.."
      << max_D << ")\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
