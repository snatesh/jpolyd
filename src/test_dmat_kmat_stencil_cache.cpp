#include <jdmat.hh>
#include <jkmat.hh>

#include <algorithm>
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
    const double d = a[i] - b[i];
    numerator += d * d;
    denominator += b[i] * b[i];
  }
  return std::sqrt(numerator / std::max(denominator, 1.0e-300));
}

template<int D>
void test_dmat(
  int n,
  unsigned int q,
  const std::filesystem::path& cache_folder,
  double tolerance)
{
  using DOp = jsimplex::DMat<D, double>;
  using Basis = jsimplex::Basis<D, double>;

  const auto kappa_generic = generic_kappa<D>();
  std::array<double, D + 1> kappa_zero{};
  const int ncol = Basis::dim_Pi(n);
  const int nrow = Basis::dim_Pi(n - 1);

  std::map<std::string, std::vector<int>> signature_to_axes;

  for (int axis = 0; axis < D; ++axis)
  {
    jsimplex::DMatStencil discovered{};
    const bool loaded_first = DOp::load_or_discover_natural_stencil(
      q,
      kappa_generic.data(),
      axis,
      &discovered,
      cache_folder.string());

    if (loaded_first)
    {
      throw std::runtime_error(
        "DMat first cache request unexpectedly loaded an existing file");
    }

    const auto cache_path =
      DOp::stencil_cache_path(axis, cache_folder.string());
    if (!std::filesystem::exists(cache_path))
    {
      throw std::runtime_error("DMat cache file was not written");
    }

    const std::string signature = DOp::stencil_signature(discovered);
    signature_to_axes[signature].push_back(axis);

    jsimplex::DMatStencil loaded{};
    const bool loaded_second = DOp::load_or_discover_natural_stencil(
      q,
      kappa_zero.data(),
      axis,
      &loaded,
      cache_folder.string());

    if (!loaded_second)
    {
      throw std::runtime_error(
        "DMat second cache request unexpectedly rediscovered");
    }
    if (!DOp::deltas_equal(discovered, loaded))
    {
      throw std::runtime_error(
        "DMat loaded stencil differs from discovered stencil");
    }

    int* colptr = nullptr;
    int* rowind = nullptr;
    const std::size_t nnz =
      DOp::build_natural_csc_pattern_from_stencil(
        n,
        loaded,
        &colptr,
        &rowind);

    std::vector<double> values(nnz, 0.0);
    DOp::fill_tprod_natural_csc_values(
      n,
      q,
      kappa_zero.data(),
      axis,
      colptr,
      rowind,
      values.data());

    int* colptr_combined = nullptr;
    int* rowind_combined = nullptr;
    double* values_combined = nullptr;
    const std::size_t nnz_combined =
      DOp::build_tprod_natural_pruned_csc_from_stencil(
        n,
        q,
        kappa_zero.data(),
        axis,
        loaded,
        &colptr_combined,
        &rowind_combined,
        &values_combined);

    if (nnz != nnz_combined ||
        std::memcmp(
          colptr,
          colptr_combined,
          static_cast<std::size_t>(ncol + 1) * sizeof(int)) != 0 ||
        (nnz > 0 &&
         std::memcmp(
           rowind,
           rowind_combined,
           nnz * sizeof(int)) != 0))
    {
      throw std::runtime_error(
        "DMat split pattern/value path differs from combined path");
    }

    std::vector<double> combined_values(
      values_combined,
      values_combined + nnz);
    const double split_combined_error =
      relative_error(values, combined_values);
    if (split_combined_error > tolerance)
    {
      throw std::runtime_error(
        "DMat split/combined value mismatch");
    }

    std::vector<double> dense(
      static_cast<std::size_t>(ncol) * ncol,
      0.0);
    DOp::build_tprod_natural_pruned_dense(
      n,
      q,
      kappa_zero.data(),
      axis,
      dense.data());

    std::vector<double> sparse_dense(
      static_cast<std::size_t>(nrow) * ncol,
      0.0);
    for (int j = 0; j < ncol; ++j)
    {
      for (int p = colptr[j]; p < colptr[j + 1]; ++p)
      {
        sparse_dense[
          static_cast<std::size_t>(rowind[p]) * ncol + j] =
          values[static_cast<std::size_t>(p)];
      }
    }

    std::vector<double> dense_active(
      static_cast<std::size_t>(nrow) * ncol,
      0.0);
    for (int i = 0; i < nrow; ++i)
    {
      std::memcpy(
        dense_active.data() + static_cast<std::size_t>(i) * ncol,
        dense.data() + static_cast<std::size_t>(i) * ncol,
        static_cast<std::size_t>(ncol) * sizeof(double));
    }

    const double dense_error =
      relative_error(sparse_dense, dense_active);
    if (dense_error > tolerance)
    {
      throw std::runtime_error(
        "DMat fixed-stencil CSC differs from dense natural builder");
    }

    std::free(colptr);
    std::free(rowind);
    std::free(colptr_combined);
    std::free(rowind_combined);
    std::free(values_combined);
    discovered.clear();
    loaded.clear();

    std::cout
      << "  DMat axis=" << axis
      << ": nnz=" << nnz
      << ", dense error=" << dense_error
      << ", cache reload PASS\n";
  }

  std::cout << "  DMat signature interning groups:\n";
  for (const auto& entry : signature_to_axes)
  {
    std::cout << "    axes";
    for (const int axis : entry.second)
    {
      std::cout << ' ' << axis;
    }
    std::cout << " share one signature\n";
  }
}

template<int D>
void test_kmat(
  int n,
  unsigned int q,
  const std::filesystem::path& cache_folder,
  double tolerance)
{
  using KOp = jsimplex::KMat<D, double>;
  using Basis = jsimplex::Basis<D, double>;

  const auto kappa_generic = generic_kappa<D>();
  std::array<double, D + 1> kappa_zero{};
  const int M = Basis::dim_Pi(n);

  std::map<std::string, std::vector<int>> signature_to_parameters;

  for (int parameter = 0; parameter < D + 1; ++parameter)
  {
    jsimplex::KMatStencil discovered{};
    const bool loaded_first = KOp::load_or_discover_natural_stencil(
      q,
      kappa_generic.data(),
      parameter,
      &discovered,
      cache_folder.string());

    if (loaded_first)
    {
      throw std::runtime_error(
        "KMat first cache request unexpectedly loaded an existing file");
    }

    const auto cache_path =
      KOp::stencil_cache_path(parameter, cache_folder.string());
    if (!std::filesystem::exists(cache_path))
    {
      throw std::runtime_error("KMat cache file was not written");
    }

    const std::string signature = KOp::stencil_signature(discovered);
    signature_to_parameters[signature].push_back(parameter);

    jsimplex::KMatStencil loaded{};
    const bool loaded_second = KOp::load_or_discover_natural_stencil(
      q,
      kappa_zero.data(),
      parameter,
      &loaded,
      cache_folder.string());

    if (!loaded_second)
    {
      throw std::runtime_error(
        "KMat second cache request unexpectedly rediscovered");
    }
    if (!KOp::deltas_equal(discovered, loaded))
    {
      throw std::runtime_error(
        "KMat loaded stencil differs from discovered stencil");
    }

    int* colptr = nullptr;
    int* rowind = nullptr;
    const std::size_t nnz =
      KOp::build_natural_csc_pattern_from_stencil(
        n,
        loaded,
        &colptr,
        &rowind);

    std::vector<double> values(nnz, 0.0);
    KOp::fill_tprod_natural_csc_values(
      n,
      q,
      kappa_zero.data(),
      parameter,
      colptr,
      rowind,
      values.data());

    int* colptr_combined = nullptr;
    int* rowind_combined = nullptr;
    double* values_combined = nullptr;
    const std::size_t nnz_combined =
      KOp::build_tprod_natural_pruned_csc_from_stencil(
        n,
        q,
        kappa_zero.data(),
        parameter,
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
        "KMat split pattern/value path differs from combined path");
    }

    std::vector<double> combined_values(
      values_combined,
      values_combined + nnz);
    const double split_combined_error =
      relative_error(values, combined_values);
    if (split_combined_error > tolerance)
    {
      throw std::runtime_error(
        "KMat split/combined value mismatch");
    }

    std::array<double, D + 1> kappa_target = kappa_zero;
    kappa_target[parameter] += 1.0;

    std::vector<double> dense(
      static_cast<std::size_t>(M) * M,
      0.0);
    KOp::build_tprod_natural_pruned_dense(
      n,
      q,
      kappa_zero.data(),
      kappa_target.data(),
      dense.data());

    std::vector<double> sparse_dense(
      static_cast<std::size_t>(M) * M,
      0.0);
    for (int j = 0; j < M; ++j)
    {
      for (int p = colptr[j]; p < colptr[j + 1]; ++p)
      {
        sparse_dense[
          static_cast<std::size_t>(rowind[p]) * M + j] =
          values[static_cast<std::size_t>(p)];
      }
    }

    const double dense_error =
      relative_error(sparse_dense, dense);
    if (dense_error > tolerance)
    {
      throw std::runtime_error(
        "KMat fixed-stencil CSC differs from dense natural builder");
    }

    std::free(colptr);
    std::free(rowind);
    std::free(colptr_combined);
    std::free(rowind_combined);
    std::free(values_combined);
    discovered.clear();
    loaded.clear();

    std::cout
      << "  KMat parameter=" << parameter
      << ": nnz=" << nnz
      << ", dense error=" << dense_error
      << ", cache reload PASS\n";
  }

  std::cout << "  KMat signature interning groups:\n";
  for (const auto& entry : signature_to_parameters)
  {
    std::cout << "    parameters";
    for (const int parameter : entry.second)
    {
      std::cout << ' ' << parameter;
    }
    std::cout << " share one signature\n";
  }
}

template<int D>
void run_dimension(
  int n,
  unsigned int q,
  const std::filesystem::path& root,
  double tolerance)
{
  const auto folder = root / ("D" + std::to_string(D));
  std::filesystem::remove_all(folder);

  std::cout
    << "D=" << D
    << ", n=" << n
    << ", q=" << q
    << ", cache=" << folder
    << '\n';

  test_dmat<D>(n, q, folder, tolerance);
  test_kmat<D>(n, q, folder, tolerance);
  std::cout << "D=" << D << ": PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  const int n = (argc > 1) ? std::stoi(argv[1]) : 6;
  const unsigned int q =
    (argc > 2)
      ? static_cast<unsigned int>(std::stoul(argv[2]))
      : static_cast<unsigned int>(n + 2);
  const std::filesystem::path root =
    (argc > 3)
      ? std::filesystem::path(argv[3])
      : std::filesystem::path("stencil_cache_test");
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
    << "all DMat/KMat stencil cache and split-fill tests passed\n";
  return 0;
}
