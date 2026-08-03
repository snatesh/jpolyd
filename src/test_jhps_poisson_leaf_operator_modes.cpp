#include <jhps_c.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

struct SolveResult
{
  int rc = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int root_nb = 0;
  int interface_nb = 0;

  double root_residual = 0.0;
  double interface_residual = 0.0;
  double parent_residual = 0.0;

  std::vector<double> coefficients;
};

double relative_error(
  const std::vector<double>& candidate,
  const std::vector<double>& reference)
{
  if (candidate.size() != reference.size())
  {
    throw std::runtime_error(
      "relative_error: size mismatch");
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
    numerator
    / std::max(
        denominator,
        1.0e-300L));
}

SolveResult solve_legacy_dense()
{
  constexpr int D = 1;
  constexpr int n = 4;
  constexpr int nelem = 2;
  constexpr int M = n + 1;
  constexpr int m_int = n - 1;

  const std::array<double, D + 1> kappa{
    0.5, 0.5};

  const std::array<int, 3> vertex_ids{
    0, 1, 2};
  const std::array<double, 3> coordinates{
    0.0, 0.5, 1.0};
  const std::array<int, nelem * (D + 1)> simplices{
    0, 1,
    1, 2};
  const std::array<int, 2> merge_pairs{
    0, 1};

  const std::array<double, nelem * m_int> source{
    0.10, -0.04, 0.03,
    -0.07, 0.05, 0.02};
  const std::array<int, 2> boundary_keys{
    0, 2};
  const std::array<double, 2> boundary_g{
    0.25, 1.15};

  SolveResult result;
  result.coefficients.assign(
    (std::size_t)nelem * M,
    0.0);

  result.rc = jhps_poisson_mesh_tree_solve(
    D,
    n,
    2,
    0,
    0,
    kappa.data(),
    (int)vertex_ids.size(),
    vertex_ids.data(),
    coordinates.data(),
    nelem,
    simplices.data(),
    1,
    merge_pairs.data(),
    source.data(),
    2,
    boundary_keys.data(),
    boundary_g.data(),
    10.0,
    1.0,
    0.0,
    0,
    result.coefficients.data(),
    &result.root_residual,
    &result.interface_residual,
    &result.parent_residual,
    &result.M,
    &result.m_int,
    &result.kf,
    &result.root_nb,
    &result.interface_nb);

  return result;
}

SolveResult solve_mode(
  int mode,
  int solver = JHPS_LEAF_LS_AUTO)
{
  constexpr int D = 1;
  constexpr int n = 4;
  constexpr int nelem = 2;
  constexpr int M = n + 1;
  constexpr int m_int = n - 1;

  const std::array<double, D + 1> kappa{
    0.5, 0.5};

  const std::array<int, 3> vertex_ids{
    0, 1, 2};
  const std::array<double, 3> coordinates{
    0.0, 0.5, 1.0};
  const std::array<int, nelem * (D + 1)> simplices{
    0, 1,
    1, 2};
  const std::array<int, 2> merge_pairs{
    0, 1};

  const std::array<double, nelem * m_int> source{
    0.10, -0.04, 0.03,
    -0.07, 0.05, 0.02};
  const std::array<int, 2> boundary_keys{
    0, 2};
  const std::array<double, 2> boundary_g{
    0.25, 1.15};

  SolveResult result;
  result.coefficients.assign(
    (std::size_t)nelem * M,
    0.0);

  result.rc =
    jhps_poisson_mesh_tree_solve_with_leaf_options(
      D,
      n,
      2,
      0,
      0,
      kappa.data(),
      (int)vertex_ids.size(),
      vertex_ids.data(),
      coordinates.data(),
      nelem,
      simplices.data(),
      1,
      merge_pairs.data(),
      source.data(),
      2,
      boundary_keys.data(),
      boundary_g.data(),
      10.0,
      1.0,
      0.0,
      0,
      mode,
      solver,
      1.0e-8,
      1,
      result.coefficients.data(),
      &result.root_residual,
      &result.interface_residual,
      &result.parent_residual,
      &result.M,
      &result.m_int,
      &result.kf,
      &result.root_nb,
      &result.interface_nb);

  return result;
}

void require_success(
  const SolveResult& result,
  const char* name)
{
  if (result.rc != 0)
  {
    throw std::runtime_error(
      std::string(name)
      + " returned "
      + std::to_string(result.rc));
  }

  if (result.M != 5
      || result.m_int != 3
      || result.kf != 1
      || result.root_nb != 2
      || result.interface_nb != 1)
  {
    throw std::runtime_error(
      std::string(name)
      + " returned unexpected dimensions");
  }
}

void compare(
  const SolveResult& candidate,
  const SolveResult& reference,
  const char* name,
  double tolerance)
{
  require_success(candidate, name);

  const double coefficient_error =
    relative_error(
      candidate.coefficients,
      reference.coefficients);

  const double residual_difference =
    std::max(
      {
        std::abs(
          candidate.root_residual
          - reference.root_residual),
        std::abs(
          candidate.interface_residual
          - reference.interface_residual),
        std::abs(
          candidate.parent_residual
          - reference.parent_residual)
      });

  std::cout
    << name
    << ": coefficient_error="
    << coefficient_error
    << ", residual_difference="
    << residual_difference
    << '\n';

  if (coefficient_error > tolerance
      || residual_difference > tolerance)
  {
    throw std::runtime_error(
      std::string(name)
      + " differs from legacy Dense");
  }
}

} // namespace

int main()
{
  try
  {
    constexpr double tolerance = 2.0e-8;

    const SolveResult legacy =
      solve_legacy_dense();
    require_success(legacy, "legacy Dense");

    const SolveResult explicit_dense =
      solve_mode(JHPS_LEAF_OPERATOR_DENSE);
    const SolveResult dense_lsmr =
      solve_mode(
        JHPS_LEAF_OPERATOR_DENSE,
        JHPS_LEAF_LS_LSMR);
    const SolveResult dense_qr =
      solve_mode(
        JHPS_LEAF_OPERATOR_DENSE,
        JHPS_LEAF_LS_DENSE_QR);
    const SolveResult dense_sparse =
      solve_mode(JHPS_LEAF_OPERATOR_DENSE_SPARSE);
    const SolveResult matrix_free =
      solve_mode(JHPS_LEAF_OPERATOR_MATRIX_FREE);
    const SolveResult verify =
      solve_mode(JHPS_LEAF_OPERATOR_VERIFY);

    compare(
      explicit_dense,
      legacy,
      "explicit Dense",
      tolerance);
    compare(
      dense_lsmr,
      legacy,
      "Dense/LSMR",
      tolerance);
    compare(
      dense_qr,
      legacy,
      "Dense/QR",
      tolerance);
    compare(
      dense_sparse,
      legacy,
      "DenseSparse",
      tolerance);
    compare(
      matrix_free,
      legacy,
      "MatrixFree",
      tolerance);
    compare(
      verify,
      legacy,
      "Verify",
      tolerance);

    std::cout
      << "all C wrapper leaf-mode tests passed\n";
  }
  catch (const std::exception& error)
  {
    std::cerr
      << "FAIL: "
      << error.what()
      << '\n';
    return 1;
  }

  return 0;
}
