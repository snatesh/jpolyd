#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

extern "C"
{

void lsmr_c_set_options(
  const double* atol,
  const double* btol,
  const double* conlim,
  const int* itnlim,
  const int* nout,
  const int* localsize,
  const int* ctest);

void lsmr_c_step(
  const int* m,
  const int* n,
  int* action,
  double* u,
  double* v,
  const double* b,
  const double* damp,
  double* x,
  int* istop,
  int* itn,
  int* stat,
  double* normr,
  double* normA,
  double* condA,
  double* normb,
  double* normx,
  double* normAr);

} // extern "C"

namespace
{

struct SolveResult
{
  bool passed = true;
  int invalid_actions = 0;
  int failed_solves = 0;
  int maximum_iterations = 0;
  double maximum_solution_error = 0.0;
  double maximum_residual_error = 0.0;
};

void apply_A_colmajor(
  int m,
  int n,
  const std::vector<double>& A,
  const std::vector<double>& x,
  std::vector<double>& y)
{
  for (int column = 0; column < n; ++column)
  {
    const double x_column = x[static_cast<std::size_t>(column)];
    const double* A_column =
      A.data() + static_cast<std::ptrdiff_t>(m) * column;

    for (int row = 0; row < m; ++row)
    {
      y[static_cast<std::size_t>(row)] +=
        A_column[row] * x_column;
    }
  }
}

void apply_AT_colmajor(
  int m,
  int n,
  const std::vector<double>& A,
  const std::vector<double>& y,
  std::vector<double>& x)
{
  for (int column = 0; column < n; ++column)
  {
    const double* A_column =
      A.data() + static_cast<std::ptrdiff_t>(m) * column;
    double sum = 0.0;

    for (int row = 0; row < m; ++row)
    {
      sum +=
        A_column[row] * y[static_cast<std::size_t>(row)];
    }

    x[static_cast<std::size_t>(column)] += sum;
  }
}

double relative_vector_error(
  const std::vector<double>& candidate,
  const std::vector<double>& reference)
{
  long double numerator = 0.0L;
  long double denominator = 0.0L;

  for (std::size_t index = 0; index < candidate.size(); ++index)
  {
    const long double difference =
      static_cast<long double>(candidate[index]) -
      static_cast<long double>(reference[index]);
    const long double value =
      static_cast<long double>(reference[index]);

    numerator += difference * difference;
    denominator += value * value;
  }

  return static_cast<double>(
    std::sqrt(
      numerator / std::max(denominator, 1.0e-300L)));
}

void build_problem(
  int thread_index,
  int round,
  int& m,
  int& n,
  std::vector<double>& A,
  std::vector<double>& x_exact,
  std::vector<double>& b)
{
  // Deliberately use different dimensions on different threads. A singleton
  // reverse-communication state cannot safely support these simultaneous
  // solves, while THREADPRIVATE state can.
  n = 5 + (thread_index % 5);
  m = n + 6 + ((thread_index + round) % 4);

  A.assign(
    static_cast<std::size_t>(m) *
    static_cast<std::size_t>(n),
    0.0);
  x_exact.assign(static_cast<std::size_t>(n), 0.0);
  b.assign(static_cast<std::size_t>(m), 0.0);

  for (int column = 0; column < n; ++column)
  {
    x_exact[static_cast<std::size_t>(column)] =
      std::cos(
        0.31 * static_cast<double>(column + 1) +
        0.17 * static_cast<double>(thread_index + 1) +
        0.07 * static_cast<double>(round + 1));

    for (int row = 0; row < m; ++row)
    {
      double value =
        0.025 *
        std::sin(
          0.13 * static_cast<double>((row + 1) * (column + 2)) +
          0.19 * static_cast<double>(thread_index + 1) +
          0.11 * static_cast<double>(round + 1));

      if (row == column)
      {
        value +=
          1.75 +
          0.08 * static_cast<double>(thread_index) +
          0.03 * static_cast<double>(column);
      }

      A[
        static_cast<std::size_t>(row) +
        static_cast<std::size_t>(m) *
        static_cast<std::size_t>(column)] = value;
    }
  }

  apply_A_colmajor(m, n, A, x_exact, b);
}

void cleanup_state(
  int m,
  int n,
  std::vector<double>& u,
  std::vector<double>& v,
  const std::vector<double>& b,
  double damp,
  std::vector<double>& x,
  int& istop,
  int& itn,
  int& stat,
  double& normr,
  double& normA,
  double& condA,
  double& normb,
  double& normx,
  double& normAr)
{
  int action = 10;
  lsmr_c_step(
    &m,
    &n,
    &action,
    u.data(),
    v.data(),
    b.data(),
    &damp,
    x.data(),
    &istop,
    &itn,
    &stat,
    &normr,
    &normA,
    &condA,
    &normb,
    &normx,
    &normAr);
}

void run_one_solve(
  int thread_index,
  int round,
  SolveResult& result,
  double tolerance)
{
  int m = 0;
  int n = 0;
  std::vector<double> A;
  std::vector<double> x_exact;
  std::vector<double> b;
  build_problem(
    thread_index,
    round,
    m,
    n,
    A,
    x_exact,
    b);

  std::vector<double> u(static_cast<std::size_t>(m), 0.0);
  std::vector<double> v(static_cast<std::size_t>(n), 0.0);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  // Values differ slightly by thread, so this also exercises thread-local
  // options rather than only thread-local reverse-communication storage.
  const double atol =
    2.0e-13 * (1.0 + 0.05 * thread_index);
  const double btol =
    2.0e-13 * (1.0 + 0.03 * thread_index);
  const double conlim = 1.0e14;
  const int itnlim = 300 + 7 * thread_index;
  const int nout = -1;
  const int localsize = 0;
  const int ctest = 3;
  const double damp = 0.0;

  lsmr_c_set_options(
    &atol,
    &btol,
    &conlim,
    &itnlim,
    &nout,
    &localsize,
    &ctest);

  // Every OpenMP thread reaches this point before any thread initializes its
  // LSMR state. This makes accidental singleton-state sharing much easier to
  // detect than independent, unsynchronized solves.
  #pragma omp barrier

  int action = 0;
  int istop = 0;
  int itn = 0;
  int stat = 0;
  double normr = 0.0;
  double normA = 0.0;
  double condA = 0.0;
  double normb = 0.0;
  double normx = 0.0;
  double normAr = 0.0;

  lsmr_c_step(
    &m,
    &n,
    &action,
    u.data(),
    v.data(),
    b.data(),
    &damp,
    x.data(),
    &istop,
    &itn,
    &stat,
    &normr,
    &normA,
    &condA,
    &normb,
    &normx,
    &normAr);

  // Force all active solves to have initialized before any thread proceeds.
  #pragma omp barrier

  bool valid_action = true;
  while (action != 0)
  {
    if (action == 1)
    {
      apply_AT_colmajor(m, n, A, u, v);
    }
    else if (action == 2)
    {
      apply_A_colmajor(m, n, A, v, u);
    }
    else
    {
      valid_action = false;
      ++result.invalid_actions;
      break;
    }

    // Encourage interleaving of reverse-communication calls across threads.
    std::this_thread::yield();

    lsmr_c_step(
      &m,
      &n,
      &action,
      u.data(),
      v.data(),
      b.data(),
      &damp,
      x.data(),
      &istop,
      &itn,
      &stat,
      &normr,
      &normA,
      &condA,
      &normb,
      &normx,
      &normAr);
  }

  cleanup_state(
    m,
    n,
    u,
    v,
    b,
    damp,
    x,
    istop,
    itn,
    stat,
    normr,
    normA,
    condA,
    normb,
    normx,
    normAr);

  result.maximum_iterations =
    std::max(result.maximum_iterations, itn);

  if (!valid_action)
  {
    result.passed = false;
    ++result.failed_solves;
    return;
  }

  std::vector<double> residual = b;
  for (double& value : residual)
  {
    value = -value;
  }
  apply_A_colmajor(m, n, A, x, residual);

  const double solution_error =
    relative_vector_error(x, x_exact);

  // Scale the residual by the right-hand-side norm.
  long double residual_norm_squared = 0.0L;
  long double b_norm_squared = 0.0L;
  for (int row = 0; row < m; ++row)
  {
    const long double residual_value =
      static_cast<long double>(
        residual[static_cast<std::size_t>(row)]);
    const long double b_value =
      static_cast<long double>(
        b[static_cast<std::size_t>(row)]);
    residual_norm_squared += residual_value * residual_value;
    b_norm_squared += b_value * b_value;
  }
  const double scaled_residual =
    static_cast<double>(
      std::sqrt(
        residual_norm_squared /
        std::max(b_norm_squared, 1.0e-300L)));

  result.maximum_solution_error =
    std::max(result.maximum_solution_error, solution_error);
  result.maximum_residual_error =
    std::max(result.maximum_residual_error, scaled_residual);

  if (!std::isfinite(solution_error) ||
      !std::isfinite(scaled_residual) ||
      solution_error > tolerance ||
      scaled_residual > tolerance ||
      stat != 0)
  {
    result.passed = false;
    ++result.failed_solves;
  }
}

} // namespace

int main(int argc, char** argv)
{
  const int requested_threads =
    (argc > 1) ? std::max(2, std::stoi(argv[1])) : 4;
  const int rounds =
    (argc > 2) ? std::max(1, std::stoi(argv[2])) : 32;
  const double tolerance =
    (argc > 3) ? std::stod(argv[3]) : 2.0e-10;

  omp_set_dynamic(0);

  std::vector<SolveResult> results(
    static_cast<std::size_t>(requested_threads));
  int actual_threads = 0;

  #pragma omp parallel num_threads(requested_threads)
  {
    const int thread_index = omp_get_thread_num();

    #pragma omp single
    {
      actual_threads = omp_get_num_threads();
    }

    for (int round = 0; round < rounds; ++round)
    {
      run_one_solve(
        thread_index,
        round,
        results[static_cast<std::size_t>(thread_index)],
        tolerance);

      // Keep rounds synchronized so all thread-local states are repeatedly
      // created, exercised concurrently, and cleaned up together.
      #pragma omp barrier
    }
  }

  if (actual_threads < 2)
  {
    std::cerr
      << "FAIL: OpenMP created only "
      << actual_threads
      << " thread; this did not test concurrency.\n";
    return 2;
  }

  bool passed = true;
  std::cout
    << "LSMR shim thread-safety stress test\n"
    << "  threads: " << actual_threads << '\n'
    << "  rounds per thread: " << rounds << '\n'
    << std::scientific
    << std::setprecision(3);

  for (int thread_index = 0;
       thread_index < actual_threads;
       ++thread_index)
  {
    const SolveResult& result =
      results[static_cast<std::size_t>(thread_index)];
    passed = passed && result.passed;

    std::cout
      << "  thread " << thread_index
      << ": solution error <= "
      << result.maximum_solution_error
      << ", residual <= "
      << result.maximum_residual_error
      << ", max itn = "
      << result.maximum_iterations
      << ", failures = "
      << result.failed_solves
      << ", invalid actions = "
      << result.invalid_actions
      << '\n';
  }

  if (!passed)
  {
    std::cerr
      << "FAIL: concurrent LSMR solves were not independent.\n";
    return 1;
  }

  std::cout
    << "PASS: concurrent solves retained independent "
    << "LSMR state and options.\n";
  return 0;
}
