#include <jelliptic.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

template<int D>
void run_dimension(int n, int p, unsigned int seed)
{
  using Real = double;

  std::array<Real, D + 1> kappa{};
  for (int i = 0; i <= D; ++i)
    kappa[(std::size_t)i] = Real(0.63 + 0.11 * i);

  jsimplex::EllipticDegreeSpec degrees{p, p, p};
  jsimplex::EllipticPlan<D,Real> plan(
    n,
    kappa.data(),
    degrees,
    true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature,
    true);

  if (plan.residual_degree != n || plan.mR != plan.M)
    throw std::runtime_error("trial-degree residual policy mismatch");

  std::mt19937 generator(seed + 7919u * D);
  std::normal_distribution<Real> normal(Real(0), Real(1));

  for (int order = 0; order <= 2; ++order)
  {
    const int coefficient_degree = plan.coefficient_degree(order);
    const int input_degree = plan.derivative_degree(order);
    const int Mp = plan.coefficient_size(order);
    const int MN = jsimplex::Basis<D,Real>::dim_Pi(input_degree);
    const int MR = plan.mR;

    std::vector<Real> q((std::size_t)Mp, Real(0));
    for (Real& value : q)
      value = normal(generator);

    std::vector<Real> quadrature_matrix(
      (std::size_t)MR * MN,
      Real(0));
    std::vector<Real> clenshaw_matrix(
      (std::size_t)MR * MN,
      Real(0));

    jsimplex::MultByQQuadratureWorkspace<Real> quadrature_work;
    jsimplex::assemble_restricted_mult_quadrature<D,Real>(
      plan.quadrature_plan(),
      q.data(),
      coefficient_degree,
      input_degree,
      plan.residual_degree,
      quadrature_work,
      quadrature_matrix.data());

    const int plan_id = plan.plan_id_for_order(order);
    const auto& clenshaw_plan = plan.entry(plan_id);
    jsimplex::MultByQClenshawWorkspace<D,Real> clenshaw_work;
    if (!clenshaw_plan.scalar_only)
      clenshaw_work.init(clenshaw_plan.p, clenshaw_plan.MK);

    std::vector<Real> input(
      (std::size_t)clenshaw_plan.MK,
      Real(0));
    std::vector<Real> output(
      (std::size_t)clenshaw_plan.MK,
      Real(0));

    for (int column = 0; column < MN; ++column)
    {
      std::fill(input.begin(), input.end(), Real(0));
      input[(std::size_t)column] = Real(1);
      clenshaw_plan.mult.apply(
        q.data(),
        input.data(),
        output.data(),
        clenshaw_work);

      const int active_rows = std::min(MR, clenshaw_plan.MK);
      for (int row = 0; row < active_rows; ++row)
      {
        clenshaw_matrix[
          (std::size_t)row
          + (std::size_t)MR * (std::size_t)column] =
          output[(std::size_t)row];
      }
    }

    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (std::size_t entry = 0;
         entry < quadrature_matrix.size();
         ++entry)
    {
      const long double difference =
        (long double)quadrature_matrix[entry]
        - (long double)clenshaw_matrix[entry];
      const long double reference =
        (long double)clenshaw_matrix[entry];
      numerator += difference * difference;
      denominator += reference * reference;
    }

    const double relative_error =
      std::sqrt((double)(
        numerator / std::max(denominator, 1.0e-300L)));

    std::cout
      << "D=" << D
      << " n=" << n
      << " order=" << order
      << " q_mult=" << plan.q_mult
      << " relative=" << relative_error
      << '\n';

    if (relative_error > 1.0e-10)
      throw std::runtime_error(
        "quadrature/Clenshaw multiplication matrix mismatch");
  }
}

} // namespace

int main()
{
  try
  {
    run_dimension<1>(4, 4, 1001u);
    run_dimension<2>(4, 4, 1002u);
    run_dimension<3>(4, 4, 1003u);
    run_dimension<4>(4, 4, 1004u);
    std::cout << "all quadrature multiplication-matrix tests passed\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
