#include <jelliptic.hh>
#include <jgeom.hh>
#include <jprecomp.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

template<int D>
std::array<double, D + 1> make_kappa()
{
  std::array<double, D + 1> kappa{};
  for (int parameter = 0; parameter <= D; ++parameter)
  {
    kappa[(std::size_t)parameter] =
      0.09
      + 0.08 * parameter
      + 0.015 * ((parameter + 1) % 2);
  }
  return kappa;
}

template<int D>
std::array<double, D * (D + 1)> make_vertices()
{
  std::array<double, D * (D + 1)> vertices{};

  // Nonzero origin.
  for (int row = 0; row < D; ++row)
  {
    vertices[(std::size_t)row] =
      0.07 * (row + 1);
  }

  // Moderately skew, well-conditioned affine map.
  for (int column = 0; column < D; ++column)
  {
    for (int row = 0; row < D; ++row)
    {
      double value = 0.0;
      if (row == column)
      {
        value = 1.0 + 0.13 * column;
      }
      else if (row > column)
      {
        value = 0.04 * (row + column + 1);
      }
      else
      {
        value = -0.025 * (row + 1);
      }

      vertices[
        (std::size_t)row
        + (std::size_t)D * (std::size_t)(column + 1)] =
        vertices[(std::size_t)row] + value;
    }
  }

  return vertices;
}

double relative_frobenius(
  const std::vector<double>& candidate,
  const std::vector<double>& reference)
{
  if (candidate.size() != reference.size())
  {
    throw std::runtime_error(
      "relative_frobenius: size mismatch");
  }

  long double numerator = 0.0L;
  long double denominator = 0.0L;

  for (std::size_t index = 0;
       index < candidate.size();
       ++index)
  {
    const long double difference =
      (long double)candidate[index]
      - (long double)reference[index];
    const long double value =
      (long double)reference[index];

    numerator += difference * difference;
    denominator += value * value;
  }

  return (double)std::sqrt(
    numerator / std::max(denominator, 1.0e-300L));
}

template<int D>
void compare_plan(
  const jsimplex::RefSimplexPrecomp<D,double>& pre,
  const jsimplex::DSimplexGeom<D,double>& geom,
  const jsimplex::EllipticDegreeSpec& degrees,
  std::mt19937& generator,
  double tolerance)
{
  using Plan = jsimplex::EllipticPlan<D,double>;
  using Workspace = jsimplex::EllipticWorkspace<D,double>;
  using CoeffView =
    jsimplex::EllipticElementCoefficientsView<D,double>;

  Plan plan(pre, degrees);
  Workspace dense_workspace(plan);
  Workspace dag_workspace(plan);

  std::normal_distribution<double> normal(0.0, 0.15);

  const int Mp2 = plan.coefficient_size(2);
  const int Mp1 = plan.coefficient_size(1);
  const int Mp0 = plan.coefficient_size(0);

  std::vector<double> A(
    (std::size_t)D * (std::size_t)D
      * (std::size_t)Mp2,
    0.0);
  std::vector<double> b(
    (std::size_t)D * (std::size_t)Mp1,
    0.0);
  std::vector<double> c(
    (std::size_t)Mp0,
    0.0);

  if (degrees.p2 >= 0)
  {
    for (double& value : A)
    {
      value = normal(generator);
    }

    // Add a positive constant diagonal principal part.
    const double one_coeff =
      1.0 / plan.phi0_res;
    for (int axis = 0; axis < D; ++axis)
    {
      A[
        (
          (std::size_t)axis * (std::size_t)D
          + (std::size_t)axis
        ) * (std::size_t)Mp2] +=
        one_coeff;
    }
  }

  if (degrees.p1 >= 0)
  {
    for (double& value : b)
    {
      value = normal(generator);
    }
  }

  if (degrees.p0 >= 0)
  {
    for (double& value : c)
    {
      value = normal(generator);
    }
  }

  CoeffView coeffs;
  coeffs.A = degrees.p2 >= 0 ? A.data() : nullptr;
  coeffs.b = degrees.p1 >= 0 ? b.data() : nullptr;
  coeffs.c = degrees.p0 >= 0 ? c.data() : nullptr;

  std::vector<double> dense(
    (std::size_t)pre.m_int * (std::size_t)pre.M,
    0.0);
  std::vector<double> dag(
    (std::size_t)pre.m_int * (std::size_t)pre.M,
    0.0);

  jsimplex::jdsimplex_assemble_elliptic_L_int<D,double>(
    pre,
    geom,
    plan,
    coeffs,
    dense_workspace,
    dense.data());

  jsimplex::jdsimplex_assemble_elliptic_L_int_dag<D,double>(
    pre,
    geom,
    plan,
    coeffs,
    dag_workspace,
    dag.data());

  const double error =
    relative_frobenius(dag, dense);

  std::cout
    << "    degrees=("
    << degrees.p2 << ","
    << degrees.p1 << ","
    << degrees.p0 << ")"
    << ", relative error="
    << std::scientific
    << std::setprecision(3)
    << error
    << '\n';

  if (error > tolerance)
  {
    throw std::runtime_error(
      "DAG Leaf assembly compatibility failure");
  }
}

template<int D>
void run_dimension(
  int n,
  int q_pad,
  double tolerance,
  const std::string& stencil_folder,
  unsigned int seed)
{
  const auto kappa = make_kappa<D>();

  jsimplex::RefSimplexPrecomp<D,double> pre(
    n,
    q_pad,
    0,
    0,
    kappa.data(),
    stencil_folder);

  const auto vertices = make_vertices<D>();
  jsimplex::DSimplexGeom<D,double> geom;
  jsimplex::dsimplex_affine_from_verts<D,double>(
    vertices.data(),
    geom);

  if (!geom.valid)
  {
    throw std::runtime_error(
      "invalid test geometry");
  }

  std::mt19937 generator(seed + 1777u * D);

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", M=" << pre.M
    << ", m_int=" << pre.m_int
    << '\n';

  // Constant-coefficient Poisson/Laplace path used by the default Leaf.
  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{0, -1, -1},
    generator,
    tolerance);

  // General variable-coefficient path with every differential order active.
  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{2, 1, 2},
    generator,
    tolerance);

  // Lower-order-only path.
  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{-1, 2, 1},
    generator,
    tolerance);

  std::cout << "D=" << D << ": PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  const int n =
    argc > 1 ? std::stoi(argv[1]) : 5;
  const int q_pad =
    argc > 2 ? std::stoi(argv[2]) : 2;
  const double tolerance =
    argc > 3 ? std::stod(argv[3]) : 8.0e-11;
  const std::string stencil_folder =
    argc > 4 ? argv[4] : "stencils";

  if (n < 2)
  {
    std::cerr << "n must be at least 2\n";
    return 2;
  }

  try
  {
    constexpr unsigned int seed = 184729u;

    run_dimension<1>(
      n, q_pad, tolerance, stencil_folder, seed);
    run_dimension<2>(
      n, q_pad, tolerance, stencil_folder, seed);
    run_dimension<3>(
      n, q_pad, tolerance, stencil_folder, seed);
    run_dimension<4>(
      n, q_pad, tolerance, stencil_folder, seed);
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }

  std::cout
    << "all DAG Leaf elliptic assembly compatibility tests passed\n";
  return 0;
}
