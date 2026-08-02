#include <jprecomp.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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
    kappa[static_cast<std::size_t>(parameter)] =
      0.08 + 0.11 * parameter +
      0.025 * ((parameter + 1) % 2);
  }
  return kappa;
}

double relative_scalar_error(
  long double left,
  long double right)
{
  return static_cast<double>(
    std::abs(left - right) /
    std::max(
      {std::abs(left), std::abs(right), 1.0e-300L}));
}

template<int D>
void run_dimension(
  int n,
  int q_pad,
  double tolerance,
  const std::string& stencil_folder,
  unsigned int seed)
{
  using Precomp = jsimplex::RefSimplexPrecomp<D, double>;

  const auto kappa = make_kappa<D>();
  Precomp precomp(
    n,
    q_pad,
    0,
    0,
    kappa.data(),
    stencil_folder);

  const typename Precomp::DAGCompatibilityReport report =
    precomp.dag_compatibility_report();

  std::mt19937 generator(seed + 1009u * D);
  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> x(
    static_cast<std::size_t>(precomp.M),
    0.0);
  std::vector<double> channel_adjoint(
    precomp.partial_value_count,
    0.0);
  for (double& value : x)
  {
    value = normal(generator);
  }
  for (double& value : channel_adjoint)
  {
    value = normal(generator);
  }

  std::vector<double> channel_values(
    precomp.partial_value_count,
    0.0);
  std::vector<double> x_adjoint(
    static_cast<std::size_t>(precomp.M),
    0.0);

  typename Precomp::PartialWorkspace forward_workspace;
  typename Precomp::PartialWorkspace transpose_workspace;

  precomp.apply_partials(
    x.data(),
    channel_values.data(),
    forward_workspace);
  precomp.apply_partials_transpose(
    channel_adjoint.data(),
    x_adjoint.data(),
    transpose_workspace);

  long double left = 0.0L;
  for (std::size_t index = 0;
       index < channel_values.size();
       ++index)
  {
    left +=
      static_cast<long double>(channel_values[index]) *
      static_cast<long double>(channel_adjoint[index]);
  }

  long double right = 0.0L;
  for (int index = 0; index < precomp.M; ++index)
  {
    right +=
      static_cast<long double>(
        x[static_cast<std::size_t>(index)]) *
      static_cast<long double>(
        x_adjoint[static_cast<std::size_t>(index)]);
  }

  const double adjoint_error =
    relative_scalar_error(left, right);

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", M=" << precomp.M
    << ", partials=" << precomp.partials.size()
    << ", D factors=" << precomp.dag_d_factors.size()
    << ", K factors=" << precomp.dag_k_factors.size()
    << ", D batches=" << precomp.derivative_batches.size()
    << ", K forward batches="
    << precomp.promotion_forward_batches.size()
    << '\n';

  std::cout
    << std::scientific
    << std::setprecision(3)
    << "  materialization rel errors: "
    << "L0=" << report.L0_relative
    << ", Li=" << report.Li_relative
    << ", Lij=" << report.Lij_relative
    << '\n'
    << "  stacked forward/transpose error: "
    << adjoint_error
    << '\n';

  if (report.maximum() > tolerance)
  {
    throw std::runtime_error(
      "DAG compatibility materialization exceeded tolerance");
  }
  if (adjoint_error > tolerance)
  {
    throw std::runtime_error(
      "DAG forward/transpose identity exceeded tolerance");
  }

  std::cout << "D=" << D << ": PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  const int n = (argc > 1) ? std::stoi(argv[1]) : 5;
  const int q_pad = (argc > 2) ? std::stoi(argv[2]) : 2;
  const double tolerance =
    (argc > 3) ? std::stod(argv[3]) : 5.0e-11;
  const std::string stencil_folder =
    (argc > 4) ? argv[4] : "stencils";
  const unsigned int seed = 830271u;

  if (n < 2)
  {
    std::cerr << "n must be at least 2\n";
    return 2;
  }

  try
  {
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
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }

  std::cout
    << "all jprecomp DAG compatibility tests passed\n";
  return 0;
}
