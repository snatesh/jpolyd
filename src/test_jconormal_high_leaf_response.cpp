#include <jconormal.hh>
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
void test_case(const std::string& stencil_dir, bool reversed)
{
  constexpr int n = solution_degree;
  constexpr int pA = diffusion_degree;
  const auto vertices = make_vertices<D>(reversed);

  jsimplex::Mesh<D,Real> mesh;
  std::array<int,D+1> ids{};
  for (int j = 0; j <= D; ++j)
  {
    ids[(std::size_t)j] = 17 + 3 * ((j + 2) % (D + 1));
    Point<D> point{};
    for (int r = 0; r < D; ++r)
      point[(std::size_t)r] = vertices[(std::size_t)r + (std::size_t)D*j];
    mesh.add_vertex(ids[(std::size_t)j], point);
  }
  mesh.add_simplex(ids);
  mesh.build();
  const auto& elem = mesh.element(0);
  const auto& geom = elem.geom;
  if (reversed != (geom.detB < Real(0)))
    throw std::runtime_error("unexpected affine orientation");

  std::array<Real,D+1> kappa{};
  for (int j = 0; j <= D; ++j)
    kappa[(std::size_t)j] = Real(0.71 + 0.17*j);
  const int q = n + 4;
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 4, q, q, kappa.data(), stencil_dir);

  const int M = pre.M;
  const int MpA = jsimplex::Basis<D,Real>::dim_Pi(pA);
  const int pH = n + pA - 1;
  const int kH = jsimplex::Basis<D-1,Real>::dim_Pi(pH) - pre.kf;
  const int nb = (D+1)*pre.kf;
  Manufactured<D> problem;

  std::vector<Real> A((std::size_t)D*D*MpA);
  for (int r = 0; r < D; ++r)
    for (int s = 0; s < D; ++s)
    {
      const auto coeff = project_polynomial<D>(
        vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
        pre.nq_vol, M, pA,
        [&](const Point<D>& x) { return problem.A(r,s,x); });
      std::copy(coeff.begin(), coeff.end(),
        A.begin() + ((std::size_t)r*D + s)*MpA);
    }

  // Use the existing validated DenseQR source and trace response matrices.
  // The Poisson leaf here deliberately isolates high-flux observation from
  // divergence-form interior assembly, which was tested separately.
  typename jsimplex::Leaf<D,Real>::Options options{};
  options.operator_mode = jsimplex::LeafOperatorMode::Dense;
  options.least_squares_solver = jsimplex::LeafLeastSquaresSolver::DenseQR;
  jsimplex::Leaf<D,Real> leaf;
  const auto node = jsimplex::make_poisson_homogeneous_leaf_node<D,Real>(
    mesh, 0, pre, leaf, options, Real(10));
  node.validate();
  if (node.nb() != nb || node.source_dim != leaf.m_int)
    throw std::runtime_error("invalid leaf response dimensions");

  jsimplex::MultByQQuadraturePlan<D-1,Real> high_quad(
    pre.kappa_face.data(), pH, n + pA + 2);
  jsimplex::MultByQQuadratureWorkspace<Real> high_work;

  // Noncanonical order on purpose: checks that the helper writes rows in
  // selected_faces order, not ascending face-id order.
  const std::vector<int> selected_faces = {D, 0};
  const int nHigh = static_cast<int>(selected_faces.size()) * kH;
  std::vector<Real> H, Wf;
  jsimplex::build_dense_leaf_high_conormal_response<D,Real>(
    leaf, node, pA, A.data(), high_quad, high_work,
    selected_faces, H, Wf);
  if (H.size() != (std::size_t)nHigh*nb ||
      Wf.size() != (std::size_t)nHigh*leaf.m_int)
    throw std::runtime_error("high response output shape mismatch");

  // Independently assemble Q_H and multiply it against the already-validated
  // Ulam and Uf. This verifies H and Wf matrix dimensions/column-major layout.
  std::vector<Real> QH((std::size_t)nHigh*M, Real(0));
  jsimplex::jdsimplex_assemble_QH_conormal_selected_common_face<D,Real>(
    pre, geom, leaf.face_sigma_index.data(), leaf.normal_scaled.data(),
    pA, A.data(), high_quad, high_work,
    static_cast<int>(selected_faces.size()), selected_faces.data(),
    QH.empty() ? nullptr : QH.data());
  const auto H_reference = multiply(QH, nHigh, M, node.Ulam, nb);
  const auto W_reference = multiply(QH, nHigh, M, node.Uf, leaf.m_int);
  const Real eH = relative_error(H, H_reference);
  const Real eW = relative_error(Wf, W_reference);

  // Nonzero arbitrary lambda and f test the complete affine identity and
  // independently exercise the source-response contribution.
  std::vector<Real> lambda((std::size_t)nb);
  std::vector<Real> source((std::size_t)leaf.m_int);
  for (int j = 0; j < nb; ++j)
    lambda[(std::size_t)j] = Real(0.12)*std::sin(Real(0.37)*(j+1));
  for (int j = 0; j < leaf.m_int; ++j)
    source[(std::size_t)j] = Real(0.18)*std::cos(Real(0.29)*(j+1));
  auto coeffs = matvec(node.Ulam, M, nb, lambda);
  const auto cf = matvec(node.Uf, M, leaf.m_int, source);
  for (int j = 0; j < M; ++j)
    coeffs[(std::size_t)j] += cf[(std::size_t)j];
  const auto q_direct = matvec(QH, nHigh, M, coeffs);
  auto q_response = matvec(H, nHigh, nb, lambda);
  const auto q_source = matvec(Wf, nHigh, leaf.m_int, source);
  for (int j = 0; j < nHigh; ++j)
    q_response[(std::size_t)j] += q_source[(std::size_t)j];
  const Real eAffine = relative_error(q_response, q_direct);

  // Source-aware polynomial reconstruction: select a degree-five u, then
  // construct the exact discrete source f_int=-L*c_star and trace T*c_star.
  // A_tau*c_star is now exactly the stacked local right-hand side.
  const auto exact_u = project_polynomial<D>(
    vertices, geom, pre.X_vol, pre.W_vol, pre.V_vol,
    pre.nq_vol, M, n,
    [&](const Point<D>& x) { return problem.u(x); });
  const auto exact_lambda = matvec(leaf.T, nb, M, exact_u);
  auto exact_source = matvec(leaf.L, leaf.m_int, M, exact_u);
  for (Real& v : exact_source)
    v = -v;
  const auto homogeneous = matvec(node.Ulam, M, nb, exact_lambda);
  const auto particular = matvec(node.Uf, M, leaf.m_int, exact_source);
  auto reconstructed = homogeneous;
  for (int j = 0; j < M; ++j)
    reconstructed[(std::size_t)j] += particular[(std::size_t)j];
  const Real eVolume = relative_error(reconstructed, exact_u);

  auto q_manufactured = matvec(H, nHigh, nb, exact_lambda);
  const auto q_manufactured_f = matvec(Wf, nHigh, leaf.m_int, exact_source);
  for (int j = 0; j < nHigh; ++j)
    q_manufactured[(std::size_t)j] += q_manufactured_f[(std::size_t)j];
  const auto q_exact_modal = matvec(QH, nHigh, M, exact_u);
  const Real eManufactured = relative_error(q_manufactured, q_exact_modal);

  // Independent physical weighted face projection of n^T A grad(t^5).
  // Unlike the checks above, this does not use the Q_H assembler as reference.
  jsimplex::MultByQQuadraturePlan<D-1,Real> check_quad(
    pre.kappa_face.data(), pH, n + pA + 5);
  const int nqf = check_quad.nq;
  Real error_sq = 0, reference_sq = 0;
  for (std::size_t slot = 0; slot < selected_faces.size(); ++slot)
  {
    const int f = selected_faces[slot];
    const auto& fg = elem.face_geom[(std::size_t)f];
    const Real scale = fg.scale;
    int face_vertices[D];
    jsimplex::dsimplex_face_vertices<D>(f, face_vertices);
    int perm[D];
    jsimplex::lehmer_index_to_perm<D>(leaf.face_sigma_index[(std::size_t)f], perm);
    std::vector<Real> expected((std::size_t)kH, Real(0));
    for (int qf = 0; qf < nqf; ++qf)
    {
      std::array<Real,D> mu{};
      mu[0] = Real(1);
      for (int a = 1; a < D; ++a)
      {
        mu[(std::size_t)a] = check_quad.X[(std::size_t)qf*(D-1) + (a-1)];
        mu[0] -= mu[(std::size_t)a];
      }
      std::array<Real,D+1> bary{};
      for (int a = 0; a < D; ++a)
        bary[(std::size_t)face_vertices[a]] = mu[(std::size_t)perm[a]];
      std::array<Real,D> xhat{};
      for (int a = 0; a < D; ++a)
        xhat[(std::size_t)a] = bary[(std::size_t)a+1];
      const auto x = physical_point<D>(vertices, geom, xhat.data());
      const Real t = problem.t(x);
      const Real grad_prefactor = Real(5)*t*t*t*t;
      Real qval = Real(0);
      for (int r = 0; r < D; ++r)
        for (int s = 0; s < D; ++s)
          qval += fg.unit_normal[(std::size_t)r]
            * problem.A(r,s,x) * grad_prefactor * problem.v[s];
      for (int j = 0; j < kH; ++j)
        expected[(std::size_t)j] += scale * check_quad.W[(std::size_t)qf]
          * check_quad.V[(std::size_t)qf + (std::size_t)nqf*(pre.kf+j)]
          * qval;
    }
    // Both arrays store physical face moments. The scale in the denominator
    // converts the coefficient error to the weighted physical face L2 norm.
    for (int j = 0; j < kH; ++j)
    {
      const Real diff = q_manufactured[slot*kH + (std::size_t)j] - expected[(std::size_t)j];
      error_sq += diff*diff/scale;
      reference_sq += expected[(std::size_t)j]*expected[(std::size_t)j]/scale;
    }
  }
  const Real ePhysical = std::sqrt(error_sq) /
    std::max(Real(1), std::sqrt(reference_sq));

  // Invalid repeated selections must be rejected, even though 0 selected
  // faces is permitted. No high rows exist for D=1.
  if constexpr (D == 2)
  {
    bool rejected = false;
    try
    {
      std::vector<Real> hbad, wbad;
      jsimplex::build_dense_leaf_high_conormal_response<D,Real>(
        leaf, node, pA, A.data(), high_quad, high_work,
        std::vector<int>{0,0}, hbad, wbad);
    }
    catch (const std::invalid_argument&)
    {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("duplicate selected face was not rejected");
  }

  const Real tol = D >= 4 ? Real(2e-8) : Real(2e-9);
  std::cout << "D=" << D << " n=" << n
            << " orientation=" << (reversed ? "reversed" : "positive")
            << "  H=" << eH << "  Wf=" << eW
            << "  affine=" << eAffine
            << "  u=" << eVolume
            << "  source-flux=" << eManufactured
            << "  physical-high-L2=" << ePhysical << '\n';
  if (!(eH < tol && eW < tol && eAffine < tol &&
        eVolume < tol && eManufactured < tol && ePhysical < tol))
    throw std::runtime_error("high-conormal leaf response regression failed");
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
    std::cout << "PASS: high-order Dense leaf response, n=5, D=1.."
              << max_D << ", both orientations\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
