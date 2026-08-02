#include <jmat.hh>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

template<int D>
std::array<double, D + 1> generic_kappa()
{
  std::array<double, D + 1> kappa{};
  for (int r = 0; r < D + 1; ++r)
  {
    kappa[r] = 0.17 + 0.09 * r + 0.025 * ((r + 1) % 2);
  }
  return kappa;
}

double relative_error(
  const std::vector<double>& a,
  const std::vector<double>& b)
{
  if (a.size() != b.size())
  {
    throw std::runtime_error("relative_error: size mismatch");
  }

  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    const double difference = a[i] - b[i];
    numerator += difference * difference;
    denominator += b[i] * b[i];
  }

  return std::sqrt(
    numerator / std::max(denominator, 1.0e-300));
}

template<int D>
void run_dimension(
  int n,
  unsigned int q,
  const std::filesystem::path& root,
  double tolerance)
{
  using JOp = jsimplex::JMat<D, double>;
  using Basis = jsimplex::Basis<D, double>;

  const auto folder = root / ("D" + std::to_string(D));
  std::filesystem::remove_all(folder);

  const auto kappa_generic = generic_kappa<D>();
  std::array<double, D + 1> kappa_zero{};

  const int M = Basis::dim_Pi(n);
  std::map<std::string, std::vector<int>> signature_to_coordinates;

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", q=" << q
    << ", cache=" << folder
    << '\n';

  for (int coord = 0; coord < D; ++coord)
  {
    jsimplex::JMatStencil discovered{};
    const bool loaded_first =
      JOp::load_or_discover_coordinate_stencil(
        q,
        kappa_generic.data(),
        coord,
        &discovered,
        folder.string());

    if (loaded_first)
    {
      throw std::runtime_error(
        "first JMat cache request unexpectedly loaded");
    }

    const auto cache_path =
      JOp::stencil_cache_path(coord, folder.string());
    if (!std::filesystem::exists(cache_path))
    {
      throw std::runtime_error("JMat cache file was not written");
    }

    const std::string signature =
      JOp::stencil_signature(discovered);
    signature_to_coordinates[signature].push_back(coord);

    jsimplex::JMatStencil loaded{};
    const bool loaded_second =
      JOp::load_or_discover_coordinate_stencil(
        q,
        kappa_zero.data(),
        coord,
        &loaded,
        folder.string());

    if (!loaded_second)
    {
      throw std::runtime_error(
        "second JMat cache request unexpectedly rediscovered");
    }
    if (!JOp::deltas_equal(discovered, loaded))
    {
      throw std::runtime_error(
        "loaded JMat stencil differs from discovered stencil");
    }

    int* colptr = nullptr;
    int* rowind = nullptr;
    const std::size_t nnz =
      JOp::build_csc_pattern_from_stencil(
        n,
        loaded,
        &colptr,
        &rowind);

    std::vector<double> values(nnz, 0.0);
    JOp::fill_pruned_csc_values(
      n,
      kappa_zero.data(),
      q,
      coord,
      colptr,
      rowind,
      values.data());

    int* colptr_combined = nullptr;
    int* rowind_combined = nullptr;
    double* values_combined = nullptr;
    const std::size_t nnz_combined =
      JOp::build_pruned_csc_from_stencil(
        n,
        kappa_zero.data(),
        q,
        coord,
        loaded,
        &colptr_combined,
        &rowind_combined,
        &values_combined);

    if (nnz != nnz_combined ||
        std::memcmp(
          colptr,
          colptr_combined,
          static_cast<std::size_t>(M + 1) * sizeof(int)) != 0 ||
        (nnz > 0 &&
         std::memcmp(
           rowind,
           rowind_combined,
           nnz * sizeof(int)) != 0))
    {
      throw std::runtime_error(
        "JMat split pattern/value path differs from combined path");
    }

    std::vector<double> combined_values(
      values_combined,
      values_combined + nnz);
    const double split_combined_error =
      relative_error(values, combined_values);
    if (split_combined_error > tolerance)
    {
      throw std::runtime_error(
        "JMat split/combined value mismatch");
    }

    std::vector<double> dense(
      static_cast<std::size_t>(M) *
      static_cast<std::size_t>(M),
      0.0);
    JOp::build_pruned_dense_coordinate(
      n,
      kappa_zero.data(),
      q,
      coord,
      dense.data());

    std::vector<double> sparse_dense(
      static_cast<std::size_t>(M) *
      static_cast<std::size_t>(M),
      0.0);
    for (int j = 0; j < M; ++j)
    {
      for (int p = colptr[j]; p < colptr[j + 1]; ++p)
      {
        sparse_dense[
          static_cast<std::size_t>(rowind[p]) *
          static_cast<std::size_t>(M) +
          static_cast<std::size_t>(j)] =
          values[static_cast<std::size_t>(p)];
      }
    }

    const double dense_error =
      relative_error(sparse_dense, dense);
    if (dense_error > tolerance)
    {
      throw std::runtime_error(
        "JMat fixed-stencil CSC differs from dense builder");
    }

    std::free(colptr);
    std::free(rowind);
    std::free(colptr_combined);
    std::free(rowind_combined);
    std::free(values_combined);
    discovered.clear();
    loaded.clear();

    std::cout
      << "  coordinate=" << coord
      << ": nnz=" << nnz
      << ", dense error=" << dense_error
      << ", cache reload PASS\n";
  }

  std::cout << "  JMat signature interning groups:\n";
  for (const auto& entry : signature_to_coordinates)
  {
    std::cout << "    coordinates";
    for (const int coord : entry.second)
    {
      std::cout << ' ' << coord;
    }
    std::cout << " share one signature\n";
  }

  std::cout << "D=" << D << ": PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  const int n = (argc > 1) ? std::stoi(argv[1]) : 7;
  const unsigned int q =
    (argc > 2)
      ? static_cast<unsigned int>(std::stoul(argv[2]))
      : static_cast<unsigned int>(n + 2);
  const std::filesystem::path root =
    (argc > 3)
      ? std::filesystem::path(argv[3])
      : std::filesystem::path("jmat_stencil_cache_test");
  const double tolerance =
    (argc > 4) ? std::stod(argv[4]) : 5.0e-11;

  try
  {
    std::filesystem::remove_all(root);
    run_dimension<1>(n, q, root, tolerance);
    run_dimension<2>(n, q, root, tolerance);
    run_dimension<3>(n, q, root, tolerance);
    run_dimension<4>(n, q, root, tolerance);
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }

  std::cout
    << "all JMat stencil cache and split-fill tests passed\n";
  return 0;
}
