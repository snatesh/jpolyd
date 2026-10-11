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



// Four simplices forming a fan around a codimension-two simplex.
// This is a conforming mesh in every D >= 2:
//   D=2: four triangles around one vertex;
//   D=3: four tetrahedra around one edge;
//   D=4: four 4-simplices around one triangular 2-face.
// The balanced tree is ((0,1),(2,3)), with two interfaces at the root.
template<int D>
jsimplex::Mesh<D,Real> make_four_leaf_mesh(bool reversed)
{
  static_assert(D >= 2 && D <= 4, "four-leaf regression uses D=2..4");
  jsimplex::Mesh<D,Real> mesh;
  Point<D> center{};
  center[0] = Real(0.13);
  center[1] = Real(-0.08);
  for (int r = 2; r < D; ++r)
    center[(std::size_t)r] = Real(0.045 * r);
  mesh.add_vertex(0, center);

  // A convex four-sided fan in the first two physical coordinates.
  constexpr int xy[4][2] = {{1,0}, {0,1}, {-1,0}, {0,-1}};
  for (int e = 0; e < 4; ++e)
  {
    Point<D> x = center;
    const Real a = Real(xy[e][0]);
    const Real b = Real(xy[e][1]);
    x[0] += Real(0.93)*a + Real(0.14)*b;
    x[1] += Real(0.09)*a + Real(0.82)*b;
    mesh.add_vertex(e+1, x);
  }

  // Common apex vertices form a (D-3)-simplex in the other directions.
  for (int j = 0; j < D-2; ++j)
  {
    Point<D> x = center;
    x[0] += Real(0.055 * (j+1));
    x[1] -= Real(0.035 * (j+1));
    x[(std::size_t)(j+2)] += Real(1.05 + 0.09*j);
    mesh.add_vertex(5+j, x);
  }

  for (int e = 0; e < 4; ++e)
  {
    std::array<int,D+1> ids{};
    ids[0] = 0;
    ids[1] = e+1;
    ids[2] = (e+1)%4 + 1;
    for (int j = 0; j < D-2; ++j)
      ids[(std::size_t)(3+j)] = 5+j;
    if (reversed)
      std::swap(ids[1], ids[2]);
    mesh.add_simplex(ids);
  }
  mesh.build();
  if (mesh.interior_face_keys().size() != 4)
    throw std::runtime_error("expected four interior fan interfaces");
  return mesh;
}

enum class Selection { AllExterior, Leaf0Only, SplitLeaves };

const char* selection_name(Selection selection)
{
  switch (selection)
  {
    case Selection::AllExterior: return "all-exterior";
    case Selection::Leaf0Only: return "leaf0-only";
    case Selection::SplitLeaves: return "leaves0+2";
  }
  return "unknown";
}

template<int D>
void test_case(const std::string& stencil_dir,
               bool reversed, Selection selection)
{
  constexpr int n = solution_degree;
  constexpr int pA = diffusion_degree;
  const auto mesh = make_four_leaf_mesh<D>(reversed);

  std::array<Real,D+1> kappa{};
  for (int j = 0; j <= D; ++j)
    kappa[(std::size_t)j] = Real(0.71 + 0.17*j);
  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);
  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(pA);
  const int kH = jsimplex::Basis<D-1,Real>::dim_Pi(n+pA-1) - pre.kf;
  if (kH <= 0)
    throw std::runtime_error("expected nonzero high-order face space");

  Manufactured<D> problem;
  jsimplex::MultByQQuadraturePlan<D-1,Real> high_quad(
    pre.kappa_face.data(), n+pA-1, n+pA+2);
  jsimplex::MultByQQuadratureWorkspace<Real> high_work;

  using Node = jsimplex::Node<D,Real>;
  using Obs = jsimplex::HighConormalObservation<D,Real>;
  typename jsimplex::Leaf<D,Real>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;

  // Nodes 0..3: leaves, 4: merge(0,1), 5: merge(2,3), 6: root(4,5).
  std::array<jsimplex::Leaf<D,Real>,4> leaves;
  std::array<Node,7> nodes;
  std::array<Obs,7> obs;
  std::array<std::vector<Real>,4> sources;
  std::array<std::vector<Real>,4> QH;
  std::array<bool,4> active{};

  for (int e = 0; e < 4; ++e)
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
          A.begin() + ((std::size_t)r*D+s)*MpA);
      }

    // Selecting a strict subset of true exterior faces is intentional.
    // High observations on the remaining exterior faces are inactive.
    const bool select_leaf =
      selection == Selection::AllExterior || e == 0 ||
      (selection == Selection::SplitLeaves && e == 2);
    std::vector<int> selected;
    if (select_leaf)
      for (int f = D; f >= 0; --f)
        if (mesh.is_boundary_face(elem.face_keys[(std::size_t)f]))
          selected.push_back(f); // deliberately reverse local face order
    active[(std::size_t)e] = !selected.empty();

    std::vector<Real> H, Wf;
    jsimplex::build_dense_leaf_high_conormal_response<D,Real>(
      leaves[(std::size_t)e], nodes[(std::size_t)e], pA, A.data(),
      high_quad, high_work, selected, H, Wf);
    obs[(std::size_t)e] =
      jsimplex::make_high_conormal_leaf_observation<D,Real>(
        nodes[(std::size_t)e], kH, selected, H, Wf);

    const int nh = obs[(std::size_t)e].nrows();
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
      obs[(std::size_t)e], nodes[(std::size_t)e],
      sources[(std::size_t)e].data());
  }

  nodes[4] = jsimplex::merge_nodes<D,Real>(nodes[0], nodes[1], 0, 1);
  nodes[5] = jsimplex::merge_nodes<D,Real>(nodes[2], nodes[3], 2, 3);
  obs[4] = jsimplex::merge_high_conormal_observations<D,Real>(
    nodes[4], nodes[0], obs[0], nodes[1], obs[1]);
  obs[5] = jsimplex::merge_high_conormal_observations<D,Real>(
    nodes[5], nodes[2], obs[2], nodes[3], obs[3]);
  nodes[6] = jsimplex::merge_nodes<D,Real>(nodes[4], nodes[5], 4, 5);
  obs[6] = jsimplex::merge_high_conormal_observations<D,Real>(
    nodes[6], nodes[4], obs[4], nodes[5], obs[5]);

  if (selection == Selection::Leaf0Only &&
      (!obs[4].nrows() || obs[5].nrows() != 0 || !obs[6].nrows()))
    throw std::runtime_error("leaf0-only active ancestry is incorrect");
  if (selection == Selection::SplitLeaves &&
      (!obs[4].nrows() || !obs[5].nrows()))
    throw std::runtime_error("split-leaf active ancestry is incorrect");

  // All root observations must be true exterior faces and in ROOT ordering.
  std::vector<typename Obs::FaceKey> expected;
  for (const auto& face : nodes[6].boundary_faces)
  {
    bool selected = false;
    for (int e = 0; e < 4; ++e)
      selected |= std::find(obs[(std::size_t)e].faces.begin(),
        obs[(std::size_t)e].faces.end(), face)
        != obs[(std::size_t)e].faces.end();
    if (selected)
      expected.push_back(face);
  }
  if (expected != obs[6].faces || expected.empty())
    throw std::runtime_error("root high-observation face ordering mismatch");
  if (selection == Selection::AllExterior &&
      expected.size() != mesh.boundary_face_keys().size())
    throw std::runtime_error("full exterior observation lost a face");

  std::vector<Real> lambda_root((std::size_t)nodes[6].nb());
  for (int j = 0; j < nodes[6].nb(); ++j)
    lambda_root[(std::size_t)j] =
      Real(0.17)*std::sin(Real(0.41)*(j+1));

  // Independent verification uses the ordinary downward trace pass, then
  // reconstruct_leaf_volume and direct QH application on EACH selected leaf.
  // It never uses the propagated high-flux matrices to reconstruct solutions.
  const auto check = [&](const char* phase)
  {
    std::array<std::vector<Real>,7> lambda;
    lambda[6] = lambda_root;
    jsimplex::merge_reconstruct_child_traces<D,Real>(
      nodes[6], nodes[4], nodes[5], lambda[6].data(),
      lambda[4], lambda[5]);
    jsimplex::merge_reconstruct_child_traces<D,Real>(
      nodes[4], nodes[0], nodes[1], lambda[4].data(),
      lambda[0], lambda[1]);
    jsimplex::merge_reconstruct_child_traces<D,Real>(
      nodes[5], nodes[2], nodes[3], lambda[5].data(),
      lambda[2], lambda[3]);

    Real interface_error = Real(0);
    constexpr int child_pairs[3][3] = {
      {4,0,1}, {5,2,3}, {6,4,5}
    };
    for (const auto& pair : child_pairs)
    {
      const int p = pair[0], a = pair[1], b = pair[2];
      std::vector<Real> muA((std::size_t)nodes[(std::size_t)a].nb());
      std::vector<Real> muB((std::size_t)nodes[(std::size_t)b].nb());
      jsimplex::apply_node_dtn<D,Real>(nodes[(std::size_t)a],
        lambda[(std::size_t)a].data(), muA.data());
      jsimplex::apply_node_dtn<D,Real>(nodes[(std::size_t)b],
        lambda[(std::size_t)b].data(), muB.data());
      const auto& md = nodes[(std::size_t)p].merge;
      for (int i = 0; i < md.nI; ++i)
        interface_error = std::max(interface_error,
          std::abs(muA[(std::size_t)md.A_I_idx[(std::size_t)i]] +
                   muB[(std::size_t)md.B_I_idx[(std::size_t)i]]));
    }

    std::array<std::vector<Real>,4> direct;
    for (int e = 0; e < 4; ++e)
    {
      std::vector<Real> c((std::size_t)pre.M);
      jsimplex::reconstruct_leaf_volume<D,Real>(
        nodes[(std::size_t)e], lambda[(std::size_t)e].data(), c.data());
      direct[(std::size_t)e] = matvec(QH[(std::size_t)e],
        obs[(std::size_t)e].nrows(), pre.M, c);
    }

    std::vector<Real> direct_root((std::size_t)obs[6].nrows(), Real(0));
    for (int f = 0; f < static_cast<int>(obs[6].faces.size()); ++f)
    {
      int owner = -1;
      int slot = -1;
      for (int e = 0; e < 4; ++e)
      {
        const auto& faces = obs[(std::size_t)e].faces;
        const auto it = std::find(faces.begin(), faces.end(),
          obs[6].faces[(std::size_t)f]);
        if (it != faces.end())
        {
          if (owner >= 0)
            throw std::runtime_error("duplicate leaf high face ownership");
          owner = e;
          slot = static_cast<int>(it - faces.begin());
        }
      }
      if (owner < 0)
        throw std::runtime_error("root high face has no leaf owner");
      for (int k = 0; k < kH; ++k)
        direct_root[(std::size_t)(f*kH+k)] =
          direct[(std::size_t)owner][(std::size_t)(slot*kH+k)];
    }

    auto propagated = matvec(
      obs[6].H, obs[6].nrows(), nodes[6].nb(), lambda_root);
    for (int i = 0; i < obs[6].nrows(); ++i)
      propagated[(std::size_t)i] += obs[6].z[(std::size_t)i];
    const Real high_error = relative_error(propagated, direct_root);
    if (!(high_error < Real(2e-9) && interface_error < Real(2e-9)))
      throw std::runtime_error(std::string("multilevel high response mismatch: ") + phase);
    std::cout << "D=" << D
      << " orientation=" << (reversed ? "reversed" : "positive")
      << " selected=" << selection_name(selection)
      << " phase=" << phase
      << " high=" << high_error
      << " interface=" << interface_error
      << " high_faces=" << obs[6].faces.size() << '\n';
  };

  check("initial");
  const auto original_H = obs[6].H;
  const auto original_z = obs[6].z;

  // Change source ONLY in leaf 3. Update the ordinary source state along
  // 3 -> 5 -> 6. The observation path is updated only where it is active.
  // In leaf0-only mode, the entire (2,3) subtree has NO high rows, yet
  // changing leaf 3 must still influence the leaf 0 high flux via root.r.
  for (int j = 0; j < nodes[3].source_dim; ++j)
    sources[3][(std::size_t)j] +=
      Real(0.7)*std::sin(Real(0.33)*(j+1));
  jsimplex::set_leaf_source(nodes[3], sources[3].data());
  if (active[3])
    jsimplex::update_high_conormal_leaf_source(
      obs[3], nodes[3], sources[3].data());
  jsimplex::update_merge_source<D,Real>(nodes[5], nodes[2], nodes[3]);
  if (obs[5].nrows())
    jsimplex::update_high_conormal_merge_source<D,Real>(
      nodes[5], nodes[2], obs[2], nodes[3], obs[3], obs[5]);
  jsimplex::update_merge_source<D,Real>(nodes[6], nodes[4], nodes[5]);
  jsimplex::update_high_conormal_merge_source<D,Real>(
    nodes[6], nodes[4], obs[4], nodes[5], obs[5], obs[6]);
  if (obs[6].H != original_H)
    throw std::runtime_error("H_root changed after source update");
  check("remote-source-refresh");

  if (selection == Selection::Leaf0Only)
  {
    Real max_dz = Real(0);
    for (int i = 0; i < obs[6].nrows(); ++i)
      max_dz = std::max(max_dz,
        std::abs(obs[6].z[(std::size_t)i] - original_z[(std::size_t)i]));
    std::cout << "  remote inactive-subtree influence: max abs delta z="
      << max_dz << '\n';
    if (!(max_dz > Real(1e-16)))
      throw std::runtime_error("remote inactive subtree did not influence active observation");
  }

  // Now refresh a source ON the active observation path 0 -> 4 -> 6.
  for (int j = 0; j < nodes[0].source_dim; ++j)
    sources[0][(std::size_t)j] +=
      Real(0.31)*std::cos(Real(0.21)*(j+1));
  jsimplex::set_leaf_source(nodes[0], sources[0].data());
  jsimplex::update_high_conormal_leaf_source(
    obs[0], nodes[0], sources[0].data());
  jsimplex::update_merge_source<D,Real>(nodes[4], nodes[0], nodes[1]);
  jsimplex::update_high_conormal_merge_source<D,Real>(
    nodes[4], nodes[0], obs[0], nodes[1], obs[1], obs[4]);
  jsimplex::update_merge_source<D,Real>(nodes[6], nodes[4], nodes[5]);
  jsimplex::update_high_conormal_merge_source<D,Real>(
    nodes[6], nodes[4], obs[4], nodes[5], obs[5], obs[6]);
  if (obs[6].H != original_H)
    throw std::runtime_error("H_root changed after active-path source refresh");
  check("active-source-refresh");
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
      for (Selection selection : {
        Selection::AllExterior,
        Selection::Leaf0Only,
        Selection::SplitLeaves})
      {
        test_case<2>(stencil_dir, reversed, selection);
        if (max_D >= 3) test_case<3>(stencil_dir, reversed, selection);
        if (max_D >= 4) test_case<4>(stencil_dir, reversed, selection);
      }
    std::cout << "PASS: four-leaf selective high co-normal propagation "
      << "(D=2.." << max_D << ", both orientations)\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
