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
#include <type_traits>
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

  for (int row = 0; row < D; ++row)
  {
    vertices[(std::size_t)row] =
      0.07 * (row + 1);
  }

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

double relative_vector_error(
  const std::vector<double>& candidate,
  const std::vector<double>& reference)
{
  if (candidate.size() != reference.size())
  {
    throw std::runtime_error(
      "relative_vector_error: size mismatch");
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

double relative_scalar_error(
  long double left,
  long double right)
{
  return (double)(
    std::abs(left - right)
    / std::max(
        {std::abs(left), std::abs(right), 1.0e-300L}));
}

void dense_apply(
  int rows,
  int cols,
  const std::vector<double>& matrix,
  const std::vector<double>& x,
  std::vector<double>& y)
{
  std::fill(y.begin(), y.end(), 0.0);

  for (int column = 0; column < cols; ++column)
  {
    const double x_column =
      x[(std::size_t)column];
    const double* matrix_column =
      matrix.data()
      + (std::size_t)rows * (std::size_t)column;

    for (int row = 0; row < rows; ++row)
    {
      y[(std::size_t)row] +=
        matrix_column[row] * x_column;
    }
  }
}

void dense_apply_transpose(
  int rows,
  int cols,
  const std::vector<double>& matrix,
  const std::vector<double>& y,
  std::vector<double>& x)
{
  std::fill(x.begin(), x.end(), 0.0);

  for (int column = 0; column < cols; ++column)
  {
    const double* matrix_column =
      matrix.data()
      + (std::size_t)rows * (std::size_t)column;
    double sum = 0.0;

    for (int row = 0; row < rows; ++row)
    {
      sum +=
        matrix_column[row] * y[(std::size_t)row];
    }

    x[(std::size_t)column] = sum;
  }
}

template<int D>
void verify_workspace_types()
{
  using ActionWorkspace =
    jsimplex::EllipticActionWorkspace<D,double>;
  using DenseWorkspace =
    jsimplex::EllipticDenseWorkspace<D,double>;
  using LegacyWorkspace =
    jsimplex::EllipticWorkspace<D,double>;

  static_assert(
    !ActionWorkspace::has_dense_assembly_storage,
    "action workspace must not contain dense assembly storage");

  static_assert(
    DenseWorkspace::has_dense_assembly_storage,
    "dense workspace must advertise dense assembly storage");

  static_assert(
    std::is_base_of_v<ActionWorkspace, DenseWorkspace>,
    "dense workspace should extend the action workspace");

  static_assert(
    std::is_same_v<LegacyWorkspace, DenseWorkspace>,
    "legacy EllipticWorkspace alias must preserve dense callers");
}

template<int D>
void compare_plan(
  const jsimplex::RefSimplexPrecomp<D,double>& pre,
  const jsimplex::DSimplexGeom<D,double>& geom,
  const jsimplex::EllipticDegreeSpec& degrees,
  std::mt19937& generator,
  double tolerance)
{
  using Plan =
    jsimplex::EllipticPlan<D,double>;
  using ActionWorkspace =
    jsimplex::EllipticActionWorkspace<D,double>;
  using DenseWorkspace =
    jsimplex::EllipticDenseWorkspace<D,double>;
  using CoeffView =
    jsimplex::EllipticElementCoefficientsView<D,double>;

  verify_workspace_types<D>();

  Plan plan(pre, degrees);

  // The dense path owns its explicit assembly storage.
  DenseWorkspace dense_workspace(plan);

  // Forward and transpose actions each get their own action-only workspace.
  ActionWorkspace forward_workspace(plan);
  ActionWorkspace transpose_workspace(plan);

  if (!dense_workspace.compatible(plan) ||
      !forward_workspace.compatible(plan) ||
      !transpose_workspace.compatible(plan))
  {
    throw std::runtime_error(
      "workspace/plan compatibility check failed");
  }

  std::normal_distribution<double>
    coefficient_normal(0.0, 0.15);
  std::normal_distribution<double>
    vector_normal(0.0, 1.0);

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
      value = coefficient_normal(generator);
    }

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
      value = coefficient_normal(generator);
    }
  }

  if (degrees.p0 >= 0)
  {
    for (double& value : c)
    {
      value = coefficient_normal(generator);
    }
  }

  CoeffView coeffs;
  coeffs.A =
    degrees.p2 >= 0 ? A.data() : nullptr;
  coeffs.b =
    degrees.p1 >= 0 ? b.data() : nullptr;
  coeffs.c =
    degrees.p0 >= 0 ? c.data() : nullptr;

  std::vector<double> dense_matrix(
    (std::size_t)pre.m_int
      * (std::size_t)pre.M,
    0.0);

  jsimplex::jdsimplex_assemble_elliptic_L_int_dag<D,double>(
    pre,
    geom,
    plan,
    coeffs,
    dense_workspace,
    dense_matrix.data());

  std::vector<double> x(
    (std::size_t)pre.M,
    0.0);
  std::vector<double> y(
    (std::size_t)pre.m_int,
    0.0);

  for (double& value : x)
  {
    value = vector_normal(generator);
  }
  for (double& value : y)
  {
    value = vector_normal(generator);
  }

  std::vector<double> dense_forward(
    (std::size_t)pre.m_int,
    0.0);
  std::vector<double> action_forward(
    (std::size_t)pre.m_int,
    0.0);

  dense_apply(
    pre.m_int,
    pre.M,
    dense_matrix,
    x,
    dense_forward);

  jsimplex::jdsimplex_apply_elliptic<D,double>(
    pre,
    geom,
    plan,
    coeffs,
    forward_workspace,
    x.data(),
    action_forward.data());

  std::vector<double> dense_transpose(
    (std::size_t)pre.M,
    0.0);
  std::vector<double> action_transpose(
    (std::size_t)pre.M,
    0.0);

  dense_apply_transpose(
    pre.m_int,
    pre.M,
    dense_matrix,
    y,
    dense_transpose);

  jsimplex::jdsimplex_apply_elliptic_transpose<D,double>(
    pre,
    geom,
    plan,
    coeffs,
    transpose_workspace,
    y.data(),
    action_transpose.data());

  long double left = 0.0L;
  for (int row = 0;
       row < pre.m_int;
       ++row)
  {
    left +=
      (long double)y[(std::size_t)row]
      * (long double)action_forward[
          (std::size_t)row];
  }

  long double right = 0.0L;
  for (int column = 0;
       column < pre.M;
       ++column)
  {
    right +=
      (long double)x[(std::size_t)column]
      * (long double)action_transpose[
          (std::size_t)column];
  }

  const double forward_error =
    relative_vector_error(
      action_forward,
      dense_forward);
  const double transpose_error =
    relative_vector_error(
      action_transpose,
      dense_transpose);
  const double adjoint_error =
    relative_scalar_error(left, right);

  std::cout
    << "    degrees=("
    << degrees.p2 << ","
    << degrees.p1 << ","
    << degrees.p0 << ")"
    << ", forward=" << std::scientific
    << std::setprecision(3)
    << forward_error
    << ", transpose=" << transpose_error
    << ", adjoint=" << adjoint_error
    << '\n';

  if (forward_error > tolerance)
  {
    throw std::runtime_error(
      "separated-workspace forward action mismatch");
  }
  if (transpose_error > tolerance)
  {
    throw std::runtime_error(
      "separated-workspace transpose action mismatch");
  }
  if (adjoint_error > tolerance)
  {
    throw std::runtime_error(
      "separated-workspace adjoint identity mismatch");
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

  std::mt19937 generator(
    seed + 1777u * D);

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", M=" << pre.M
    << ", m_int=" << pre.m_int
    << '\n';

  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{
      0, -1, -1},
    generator,
    tolerance);

  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{
      2, 1, 2},
    generator,
    tolerance);

  compare_plan<D>(
    pre,
    geom,
    jsimplex::EllipticDegreeSpec{
      -1, 2, 1},
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
    constexpr unsigned int seed = 917231u;

    run_dimension<1>(
      n,
      q_pad,
      tolerance,
      stencil_folder,
      seed);
    run_dimension<2>(
      n,
      q_pad,
      tolerance,
      stencil_folder,
      seed);
    run_dimension<3>(
      n,
      q_pad,
      tolerance,
      stencil_folder,
      seed);
    run_dimension<4>(
      n,
      q_pad,
      tolerance,
      stencil_folder,
      seed);
  }
  catch (const std::exception& error)
  {
    std::cerr
      << "FAIL: "
      << error.what()
      << '\n';
    return 1;
  }

  std::cout
    << "all separated jelliptic workspace tests passed\n";
  return 0;
}
