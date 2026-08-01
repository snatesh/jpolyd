#include <jkmat.hh>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

template<int D>
std::vector<std::array<double, D + 1>>
make_kappa_cases(int random_count, unsigned int seed)
{
  std::vector<std::array<double, D + 1>> cases;

  std::array<double, D + 1> generic{};
  std::array<double, D + 1> zero{};
  std::array<double, D + 1> repeated{};
  std::array<double, D + 1> alternating{};
  std::array<double, D + 1> shifted{};

  for (int r = 0; r < D + 1; ++r)
  {
    generic[r] = 0.17 + 0.11 * r + 0.025 * ((r + 1) % 2);
    zero[r] = 0.0;
    repeated[r] = 0.25;
    alternating[r] = (r % 2 == 0) ? 0.0 : 0.5;
    shifted[r] = generic[r] + static_cast<double>(r);
  }

  cases.push_back(generic);
  cases.push_back(zero);
  cases.push_back(repeated);
  cases.push_back(alternating);
  cases.push_back(shifted);

  std::mt19937 rng(seed + 1009u * static_cast<unsigned int>(D));
  std::uniform_real_distribution<double> distribution(-0.45, 2.0);

  for (int sample = 0; sample < random_count; ++sample)
  {
    std::array<double, D + 1> kappa{};
    for (int r = 0; r < D + 1; ++r)
    {
      // Jacobi admissibility requires kappa_r > -1.
      kappa[r] = distribution(rng);
    }
    cases.push_back(kappa);
  }

  return cases;
}

template<int D>
jsimplex::KMatStencil
extract_stencil_at_fixed_degree(
  int n_test,
  unsigned int q,
  const std::array<double, D + 1>& kappa_src,
  int promoted_parameter)
{
  using Real = double;
  using K = jsimplex::KMat<D, Real>;
  using Basis = jsimplex::Basis<D, Real>;

  if (n_test < 1)
  {
    throw std::invalid_argument("n_test must be positive");
  }
  if (promoted_parameter < 0 || promoted_parameter > D)
  {
    throw std::invalid_argument("invalid promoted parameter");
  }

  std::array<Real, D + 1> kappa_tgt = kappa_src;
  kappa_tgt[promoted_parameter] += Real(1);

  const int M = Basis::dim_Pi(n_test);
  std::vector<Real> dense(
    static_cast<std::size_t>(M) * static_cast<std::size_t>(M),
    Real(0));
  std::vector<int> alpha(
    static_cast<std::size_t>(M) * static_cast<std::size_t>(D),
    0);

  /*
   * Deliberately use the dense natural one-parameter builder here.
   * It does not call discover_stencil_stable(), so this test does not
   * assume the conclusion it is trying to verify.
   */
  K::build_tprod_natural_pruned_dense(
    n_test,
    q,
    kappa_src.data(),
    kappa_tgt.data(),
    dense.data());

  Basis::build_alpha_table(n_test, alpha.data());

  jsimplex::KMatStencil stencil;
  std::memset(&stencil, 0, sizeof(stencil));

  K::extract_deltas_from_block(
    n_test,
    dense.data(),
    alpha.data(),
    &stencil);

  return stencil;  // ownership of keys0/keysm1 moves to the caller
}

template<int D>
void print_stencil(
  const jsimplex::KMatStencil& stencil,
  const char* label)
{
  using K = jsimplex::KMat<D, double>;

  std::cout
    << "    " << label
    << ": ndelta0=" << stencil.ndelta0
    << ", ndeltam1=" << stencil.ndeltam1
    << '\n';

  int delta[8] = {};

  std::cout << "      delta0 :";
  for (int k = 0; k < stencil.ndelta0; ++k)
  {
    K::unpack_delta8(stencil.keys0[k], delta);
    std::cout << " (";
    for (int d = 0; d < D; ++d)
    {
      if (d) std::cout << ',';
      std::cout << delta[d];
    }
    std::cout << ')';
  }
  std::cout << '\n';

  std::cout << "      deltam1:";
  for (int k = 0; k < stencil.ndeltam1; ++k)
  {
    K::unpack_delta8(stencil.keysm1[k], delta);
    std::cout << " (";
    for (int d = 0; d < D; ++d)
    {
      if (d) std::cout << ',';
      std::cout << delta[d];
    }
    std::cout << ')';
  }
  std::cout << '\n';
}

template<int D>
void run_dimension(
  int n_test,
  unsigned int q,
  int random_count,
  unsigned int seed,
  bool verbose)
{
  using K = jsimplex::KMat<D, double>;

  const auto cases = make_kappa_cases<D>(random_count, seed);

  std::cout
    << "D=" << D
    << ", fixed extraction degree n_test=" << n_test
    << ", q=" << q
    << ", kappa cases=" << cases.size()
    << '\n';

  for (int promoted_parameter = 0;
       promoted_parameter < D + 1;
       ++promoted_parameter)
  {
    jsimplex::KMatStencil baseline =
      extract_stencil_at_fixed_degree<D>(
        n_test,
        q,
        cases.front(),
        promoted_parameter);

    if (verbose)
    {
      print_stencil<D>(baseline, "baseline");
    }

    for (std::size_t sample = 1; sample < cases.size(); ++sample)
    {
      jsimplex::KMatStencil current =
        extract_stencil_at_fixed_degree<D>(
          n_test,
          q,
          cases[sample],
          promoted_parameter);

      const bool equal = K::deltas_equal(baseline, current);
      current.clear();

      if (!equal)
      {
        baseline.clear();
        throw std::runtime_error(
          "KMat delta stencil depends on kappa: D="
          + std::to_string(D)
          + ", promoted_parameter="
          + std::to_string(promoted_parameter)
          + ", sample="
          + std::to_string(sample));
      }
    }

    std::cout
      << "  promoted_parameter=" << promoted_parameter
      << ": PASS; ndelta0=" << baseline.ndelta0
      << ", ndeltam1=" << baseline.ndeltam1
      << '\n';

    baseline.clear();
  }

  std::cout << "D=" << D << ": PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  int n_test = 6;
  int random_count = 20;
  unsigned int q = 0;
  unsigned int seed = 918273u;
  bool verbose = false;

  if (argc > 1) n_test = std::stoi(argv[1]);
  if (argc > 2) random_count = std::stoi(argv[2]);
  if (argc > 3) q = static_cast<unsigned int>(std::stoul(argv[3]));
  if (argc > 4) verbose = std::stoi(argv[4]) != 0;

  if (n_test < 1)
  {
    std::cerr << "n_test must be positive\n";
    return 2;
  }
  if (random_count < 0)
  {
    std::cerr << "random_count must be nonnegative\n";
    return 2;
  }
  if (q == 0)
  {
    q = static_cast<unsigned int>(n_test + 2);
  }

  try
  {
    std::cout
      << "KMat fixed-degree kappa-invariance test\n"
      << "This test bypasses discover_stencil_stable().\n";

    run_dimension<1>(n_test, q, random_count, seed, verbose);
    run_dimension<2>(n_test, q, random_count, seed, verbose);
    run_dimension<3>(n_test, q, random_count, seed, verbose);
    run_dimension<4>(n_test, q, random_count, seed, verbose);
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }

  std::cout << "all KMat fixed-degree kappa-invariance tests passed\n";
  return 0;
}
