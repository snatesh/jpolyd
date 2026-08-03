#include <jelliptic.hh>
#include <jgeom.hh>
#include <jleaf.hh>
#include <jprecomp.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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
  for (int parameter = 0;
       parameter <= D;
       ++parameter)
  {
    kappa[(std::size_t)parameter] =
      0.11
      + 0.07 * parameter
      + 0.02 * ((parameter + 1) % 2);
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
      0.04 * (row + 1);
  }

  for (int column = 0;
       column < D;
       ++column)
  {
    for (int row = 0; row < D; ++row)
    {
      double value = 0.0;

      if (row == column)
      {
        value =
          0.9 + 0.16 * (column + 1);
      }
      else if (row > column)
      {
        value =
          0.035 * (row + column + 1);
      }
      else
      {
        value =
          -0.02 * (row + 1);
      }

      vertices[
        (std::size_t)row
        + (std::size_t)D
          * (std::size_t)(column + 1)] =
        vertices[(std::size_t)row]
        + value;
    }
  }

  return vertices;
}

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

double scaled_difference_error(
  const std::vector<double>& candidate,
  const std::vector<double>& reference,
  const std::vector<double>& physical_scale)
{
  if (candidate.size() != reference.size()
      || candidate.size() != physical_scale.size())
  {
    throw std::runtime_error(
      "scaled_difference_error: size mismatch");
  }

  long double numerator = 0.0L;
  long double reference_norm = 0.0L;
  long double scale_norm = 0.0L;

  for (std::size_t index = 0;
       index < candidate.size();
       ++index)
  {
    const long double difference =
      (long double)candidate[index]
      - (long double)reference[index];
    const long double reference_value =
      (long double)reference[index];
    const long double scale_value =
      (long double)physical_scale[index];

    numerator += difference * difference;
    reference_norm +=
      reference_value * reference_value;
    scale_norm += scale_value * scale_value;
  }

  const long double denominator =
    std::max(
      {reference_norm,
       scale_norm,
       (long double)candidate.size()});

  return (double)std::sqrt(
    numerator / denominator);
}

template<int D>
struct ApplyResult
{
  std::vector<double> c;
  std::vector<double> trace;
  std::vector<double> raw_flux;
  std::vector<double> augmented_flux;
  std::vector<double> mismatch;
  std::vector<double> residual;
};

template<int D>
ApplyResult<D> apply_leaf(
  const jsimplex::Leaf<D,double>& leaf,
  const std::vector<double>& lambda,
  const std::vector<double>& source)
{
  ApplyResult<D> result;
  result.c.assign((std::size_t)leaf.M, 0.0);
  result.trace.assign((std::size_t)leaf.nb, 0.0);
  result.raw_flux.assign((std::size_t)leaf.nb, 0.0);
  result.augmented_flux.assign((std::size_t)leaf.nb, 0.0);
  result.mismatch.assign((std::size_t)leaf.nb, 0.0);
  result.residual.assign((std::size_t)leaf.m_int, 0.0);

  leaf.apply(
    lambda.data(),
    source.data(),
    result.c.data(),
    result.trace.data(),
    result.raw_flux.data(),
    result.augmented_flux.data(),
    result.mismatch.data(),
    result.residual.data());

  return result;
}

template<int D>
void compare_result(
  const ApplyResult<D>& candidate,
  const ApplyResult<D>& reference,
  const std::vector<double>& lambda,
  const std::vector<double>& source,
  double tolerance,
  const char* label)
{
  const double c_error =
    relative_error(candidate.c, reference.c);
  const double trace_error =
    relative_error(candidate.trace, reference.trace);
  const double flux_error =
    relative_error(
      candidate.raw_flux,
      reference.raw_flux);
  const double augmented_error =
    relative_error(
      candidate.augmented_flux,
      reference.augmented_flux);
  // Mismatch and PDE residual are expected to be very small. A pure
  // relative error with the dense residual in the denominator is therefore
  // ill-conditioned and can report O(1) even when the absolute difference is
  // at roundoff. Scale them by the corresponding physical data instead.
  const double mismatch_error =
    scaled_difference_error(
      candidate.mismatch,
      reference.mismatch,
      lambda);
  const double residual_error =
    scaled_difference_error(
      candidate.residual,
      reference.residual,
      source);

  std::cout
    << "    " << label
    << ": c=" << std::scientific
    << std::setprecision(3)
    << c_error
    << ", T=" << trace_error
    << ", F=" << flux_error
    << ", aug=" << augmented_error
    << ", mismatch=" << mismatch_error
    << ", residual=" << residual_error
    << '\n';

  const double maximum =
    std::max(
      {c_error,
       trace_error,
       flux_error,
       augmented_error,
       mismatch_error,
       residual_error});

  if (maximum > tolerance)
  {
    throw std::runtime_error(
      std::string(label)
      + " leaf result mismatch");
  }
}

template<int D>
void test_batched_boundary_maps(
  const jsimplex::Leaf<D,double>& dense_leaf,
  const jsimplex::Leaf<D,double>& matrix_free_leaf,
  std::mt19937& generator,
  double tolerance)
{
  constexpr int nrhs = 3;
  std::normal_distribution<double>
    normal(0.0, 1.0);

  std::vector<double> X(
    (std::size_t)dense_leaf.M * nrhs,
    0.0);
  for (double& value : X)
  {
    value = normal(generator);
  }

  std::vector<double> trace_dense(
    (std::size_t)dense_leaf.nb * nrhs,
    0.0);
  std::vector<double> trace_sparse(
    (std::size_t)dense_leaf.nb * nrhs,
    0.0);
  std::vector<double> flux_dense(
    (std::size_t)dense_leaf.nb * nrhs,
    0.0);
  std::vector<double> flux_sparse(
    (std::size_t)dense_leaf.nb * nrhs,
    0.0);

  dense_leaf.apply_trace_columns(
    X.data(),
    dense_leaf.M,
    nrhs,
    trace_dense.data(),
    dense_leaf.nb);
  matrix_free_leaf.apply_trace_columns(
    X.data(),
    matrix_free_leaf.M,
    nrhs,
    trace_sparse.data(),
    matrix_free_leaf.nb);

  dense_leaf.apply_flux_columns(
    X.data(),
    dense_leaf.M,
    nrhs,
    flux_dense.data(),
    dense_leaf.nb);
  matrix_free_leaf.apply_flux_columns(
    X.data(),
    matrix_free_leaf.M,
    nrhs,
    flux_sparse.data(),
    matrix_free_leaf.nb);

  const double trace_error =
    relative_error(trace_sparse, trace_dense);
  const double flux_error =
    relative_error(flux_sparse, flux_dense);

  std::cout
    << "    CSC x dense boundary maps:"
    << " T=" << std::scientific
    << std::setprecision(3)
    << trace_error
    << ", F=" << flux_error
    << '\n';

  if (trace_error > tolerance
      || flux_error > tolerance)
  {
    throw std::runtime_error(
      "batched CSC boundary map mismatch");
  }
}


template<int D>
void test_response_maps(
  const jsimplex::Leaf<D,double>& qr_leaf,
  const jsimplex::Leaf<D,double>& lsmr_leaf,
  double tolerance)
{
  const int nrhs = qr_leaf.nb + qr_leaf.m_int;
  std::vector<double> qr_maps(
    (std::size_t)qr_leaf.M * (std::size_t)nrhs,
    0.0);
  std::vector<double> lsmr_maps(
    (std::size_t)lsmr_leaf.M * (std::size_t)nrhs,
    0.0);

  typename jsimplex::Leaf<D,double>::SolveWorkspace qr_workspace(qr_leaf);
  typename jsimplex::Leaf<D,double>::SolveWorkspace lsmr_workspace(lsmr_leaf);
  long long qr_iterations = -1;
  long long lsmr_iterations = 0;

  qr_leaf.solve_response_maps(
    qr_maps.data(),
    qr_leaf.M,
    qr_workspace,
    &qr_iterations);
  lsmr_leaf.solve_response_maps(
    lsmr_maps.data(),
    lsmr_leaf.M,
    lsmr_workspace,
    &lsmr_iterations);

  const double map_error =
    relative_error(qr_maps, lsmr_maps);

  std::cout
    << "    reusable [Ulam Uf]: error="
    << std::scientific
    << std::setprecision(3)
    << map_error
    << ", QR iterations=" << qr_iterations
    << ", LSMR iterations=" << lsmr_iterations
    << '\n';

  if (qr_iterations != 0
      || lsmr_iterations <= 0
      || map_error > tolerance)
  {
    throw std::runtime_error(
      "DenseQR/LSMR response-map mismatch");
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
  using Precomp =
    jsimplex::RefSimplexPrecomp<D,double>;
  using Plan =
    jsimplex::EllipticPlan<D,double>;
  using DenseWorkspace =
    jsimplex::EllipticDenseWorkspace<D,double>;
  using Leaf =
    jsimplex::Leaf<D,double>;

  const auto kappa = make_kappa<D>();
  Precomp pre(
    n,
    q_pad,
    0,
    0,
    kappa.data(),
    stencil_folder);

  const auto vertices = make_vertices<D>();
  std::array<int, D + 1> global_vertices{};
  for (int vertex = 0;
       vertex <= D;
       ++vertex)
  {
    global_vertices[(std::size_t)vertex] =
      100 * D + vertex;
  }

  const jsimplex::EllipticDegreeSpec degrees{
    2, 1, 2};
  Plan plan(pre, degrees);
  DenseWorkspace legacy_dense_workspace(plan);
  DenseWorkspace dense_lsmr_workspace(plan);

  std::mt19937 generator(
    seed + 4099u * D);
  std::normal_distribution<double>
    coefficient_normal(0.0, 0.12);
  std::normal_distribution<double>
    vector_normal(0.0, 1.0);

  const int Mp2 = plan.coefficient_size(2);
  const int Mp1 = plan.coefficient_size(1);
  const int Mp0 = plan.coefficient_size(0);

  std::vector<double> A(
    (std::size_t)D * D * Mp2,
    0.0);
  std::vector<double> b(
    (std::size_t)D * Mp1,
    0.0);
  std::vector<double> c(
    (std::size_t)Mp0,
    0.0);

  for (double& value : A)
  {
    value = coefficient_normal(generator);
  }
  for (double& value : b)
  {
    value = coefficient_normal(generator);
  }
  for (double& value : c)
  {
    value = coefficient_normal(generator);
  }

  const double one_coeff =
    1.0 / plan.phi0_res;
  for (int axis = 0;
       axis < D;
       ++axis)
  {
    A[
      (
        (std::size_t)axis * D
        + (std::size_t)axis
      ) * (std::size_t)Mp2] +=
      one_coeff;
  }

  jsimplex::EllipticElementCoefficientsView<D,double>
    coeffs;
  coeffs.A = A.data();
  coeffs.b = b.data();
  coeffs.c = c.data();

  typename Leaf::LsmrOptions lsmr_options;
  lsmr_options.atol = 1.0e-13;
  lsmr_options.btol = 1.0e-13;
  lsmr_options.itnlim = 3000;
  lsmr_options.nout = -1;

  // Legacy call shape: no LeafOptions argument, so it must remain Dense.
  Leaf dense_leaf(
    pre,
    vertices.data(),
    global_vertices.data(),
    plan,
    coeffs,
    legacy_dense_workspace,
    1.0,
    lsmr_options);

  typename Leaf::Options dense_lsmr_options;
  dense_lsmr_options.operator_mode =
    jsimplex::LeafOperatorMode::Dense;
  dense_lsmr_options.least_squares_solver =
    jsimplex::LeafLeastSquaresSolver::LSMR;
  dense_lsmr_options.interior_scale_override =
    dense_leaf.sL;

  Leaf dense_lsmr_leaf(
    pre,
    vertices.data(),
    global_vertices.data(),
    plan,
    coeffs,
    dense_lsmr_workspace,
    1.0,
    lsmr_options,
    dense_lsmr_options);

  typename Leaf::Options matrix_free_options;
  matrix_free_options.operator_mode =
    jsimplex::LeafOperatorMode::MatrixFree;

  // Force exactly the same interior scaling as the dense leaf.
  matrix_free_options.interior_scale_override =
    dense_leaf.sL;

  Leaf matrix_free_leaf(
    pre,
    vertices.data(),
    global_vertices.data(),
    plan,
    coeffs,
    1.0,
    lsmr_options,
    matrix_free_options);

  typename Leaf::Options dense_sparse_options;
  dense_sparse_options.operator_mode =
    jsimplex::LeafOperatorMode::DenseSparse;
  dense_sparse_options.interior_scale_override =
    dense_leaf.sL;

  Leaf dense_sparse_leaf(
    pre,
    vertices.data(),
    global_vertices.data(),
    plan,
    coeffs,
    1.0,
    lsmr_options,
    dense_sparse_options);

  typename Leaf::Options verify_options;
  verify_options.operator_mode =
    jsimplex::LeafOperatorMode::Verify;
  verify_options.verify_tolerance =
    tolerance;
  verify_options.verify_each_solve = true;

  // Force exactly the same interior scaling as the dense leaf.
  verify_options.interior_scale_override =
    dense_leaf.sL;

  Leaf verify_leaf(
    pre,
    vertices.data(),
    global_vertices.data(),
    plan,
    coeffs,
    1.0,
    lsmr_options,
    verify_options);

  const double expected_tau_row_factor =
    static_cast<double>(plan.mR)
    / static_cast<double>(plan.m2);
  const double expected_tau_C_effective =
    expected_tau_row_factor;
  const double tau_scale_tolerance = 64.0
    * std::numeric_limits<double>::epsilon()
    * std::max(1.0, expected_tau_C_effective);

  if (std::abs(dense_leaf.tau_C_base - 1.0)
        > tau_scale_tolerance
      || std::abs(
           dense_leaf.tau_residual_row_factor
           - expected_tau_row_factor)
        > tau_scale_tolerance
      || std::abs(
           dense_leaf.tau_C
           - expected_tau_C_effective)
        > tau_scale_tolerance)
  {
    throw std::runtime_error(
      "elliptic tau base/residual-row rescaling mismatch");
  }

  if (dense_leaf.operator_mode
        != jsimplex::LeafOperatorMode::Dense
      || dense_leaf.least_squares_solver
        != jsimplex::LeafLeastSquaresSolver::DenseQR
      || !dense_leaf.has_dense_local_operator())
  {
    throw std::runtime_error(
      "legacy leaf call did not remain Dense");
  }

  if (matrix_free_leaf.has_dense_local_operator()
      || !matrix_free_leaf.L.empty()
      || !matrix_free_leaf.T.empty()
      || !matrix_free_leaf.F.empty()
      || !matrix_free_leaf.A_tau.empty())
  {
    throw std::runtime_error(
      "MatrixFree leaf retained dense local matrices");
  }

  if (!dense_sparse_leaf.has_dense_interior_operator()
      || dense_sparse_leaf.has_dense_local_operator()
      || !dense_sparse_leaf.T.empty()
      || !dense_sparse_leaf.F.empty()
      || !dense_sparse_leaf.A_tau.empty())
  {
    throw std::runtime_error(
      "DenseSparse leaf storage is inconsistent");
  }

  if (!verify_leaf.has_dense_local_operator())
  {
    throw std::runtime_error(
      "Verify leaf did not retain dense fallback");
  }

  std::vector<double> lambda(
    (std::size_t)dense_leaf.nb,
    0.0);
  std::vector<double> source(
    (std::size_t)dense_leaf.m_int,
    0.0);

  for (double& value : lambda)
  {
    value = vector_normal(generator);
  }
  for (double& value : source)
  {
    value = vector_normal(generator);
  }

  const ApplyResult<D> dense_result =
    apply_leaf(dense_leaf, lambda, source);
  const ApplyResult<D> dense_lsmr_result =
    apply_leaf(dense_lsmr_leaf, lambda, source);
  const ApplyResult<D> matrix_free_result =
    apply_leaf(matrix_free_leaf, lambda, source);
  const ApplyResult<D> dense_sparse_result =
    apply_leaf(dense_sparse_leaf, lambda, source);
  const ApplyResult<D> verify_result =
    apply_leaf(verify_leaf, lambda, source);

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", M=" << pre.M
    << ", rows=" << dense_leaf.ntau_rows
    << ", sL dense=" << std::scientific
    << std::setprecision(3)
    << dense_leaf.sL
    << ", sL mf=" << matrix_free_leaf.sL
    << '\n';

  compare_result(
    dense_lsmr_result,
    dense_result,
    lambda,
    source,
    tolerance,
    "Dense/LSMR");
  test_response_maps(
    dense_leaf,
    dense_lsmr_leaf,
    tolerance);

  compare_result(
    matrix_free_result,
    dense_result,
    lambda,
    source,
    tolerance,
    "MatrixFree");
  compare_result(
    dense_sparse_result,
    dense_result,
    lambda,
    source,
    tolerance,
    "DenseSparse");
  compare_result(
    verify_result,
    dense_result,
    lambda,
    source,
    tolerance,
    "Verify");

  test_batched_boundary_maps(
    dense_leaf,
    dense_sparse_leaf,
    generator,
    tolerance);

  const double scale_error =
    std::abs(
      matrix_free_leaf.sL - dense_leaf.sL)
    / std::max(
        std::abs(dense_leaf.sL),
        1.0e-300);

  std::cout
    << "    interior scale error="
    << scale_error
    << '\n';

  if (scale_error > tolerance)
  {
    throw std::runtime_error(
      "matrix-free row-RMS scale mismatch");
  }

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
    argc > 3 ? std::stod(argv[3]) : 2.0e-8;
  const std::string stencil_folder =
    argc > 4 ? argv[4] : "stencils";

  if (n < 2)
  {
    std::cerr
      << "n must be at least 2\n";
    return 2;
  }

  try
  {
    constexpr unsigned int seed = 731927u;

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
    std::cerr
      << "FAIL: "
      << error.what()
      << '\n';
    return 1;
  }

  std::cout
    << "all Leaf operator-mode tests passed\n";
  return 0;
}
