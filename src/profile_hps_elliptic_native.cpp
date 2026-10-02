#include <jhps_c.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options
{
  fs::path case_dir;
  int warmup = 2;
  int reps = 300;
  int rhs_cases = 4;
};

void usage(const char* argv0)
{
  std::cerr
    << "usage: " << argv0
    << " CASE_DIR [--warmup N] [--reps N] [--rhs-cases N]\n";
}

Options parse_args(int argc, char** argv)
{
  if (argc < 2)
  {
    usage(argv[0]);
    throw std::invalid_argument("missing CASE_DIR");
  }

  Options options;
  options.case_dir = argv[1];

  for (int i = 2; i < argc; ++i)
  {
    const std::string arg = argv[i];

    auto require_value = [&](const char* name) -> int
    {
      if (i + 1 >= argc)
        throw std::invalid_argument(std::string("missing value for ") + name);
      return std::stoi(argv[++i]);
    };

    if (arg == "--warmup")
      options.warmup = require_value("--warmup");
    else if (arg == "--reps")
      options.reps = require_value("--reps");
    else if (arg == "--rhs-cases")
      options.rhs_cases = require_value("--rhs-cases");
    else
      throw std::invalid_argument("unknown argument: " + arg);
  }

  if (options.warmup < 0)
    throw std::invalid_argument("--warmup must be nonnegative");
  if (options.reps < 1)
    throw std::invalid_argument("--reps must be positive");
  if (options.rhs_cases < 1)
    throw std::invalid_argument("--rhs-cases must be positive");

  return options;
}

std::unordered_map<std::string, std::string>
read_metadata(const fs::path& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open metadata file: " + path.string());

  std::unordered_map<std::string, std::string> values;
  std::string key;
  std::string value;

  while (stream >> key >> value)
    values[key] = value;

  return values;
}

const std::string& require_key(
  const std::unordered_map<std::string, std::string>& meta,
  const std::string& key)
{
  const auto it = meta.find(key);
  if (it == meta.end())
    throw std::runtime_error("missing metadata key: " + key);
  return it->second;
}

int meta_int(
  const std::unordered_map<std::string, std::string>& meta,
  const std::string& key)
{
  return std::stoi(require_key(meta, key));
}

double meta_double(
  const std::unordered_map<std::string, std::string>& meta,
  const std::string& key)
{
  return std::stod(require_key(meta, key));
}

template<class T>
std::vector<T> read_binary(const fs::path& path, std::size_t count)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open binary file: " + path.string());

  std::vector<T> values(count);
  stream.read(
    reinterpret_cast<char*>(values.data()),
    static_cast<std::streamsize>(count * sizeof(T)));

  if (stream.gcount() !=
      static_cast<std::streamsize>(count * sizeof(T)))
  {
    throw std::runtime_error(
      "unexpected byte count in binary file: " + path.string());
  }

  char extra = 0;
  if (stream.read(&extra, 1))
  {
    throw std::runtime_error(
      "binary file is larger than expected: " + path.string());
  }

  return values;
}

const char* env_or_unset(const char* name)
{
  const char* value = std::getenv(name);
  return value ? value : "<unset>";
}

class SolverHandle
{
public:
  SolverHandle() = default;

  ~SolverHandle()
  {
    reset();
  }

  SolverHandle(const SolverHandle&) = delete;
  SolverHandle& operator=(const SolverHandle&) = delete;

  void** out_ptr()
  {
    reset();
    return &handle_;
  }

  void* get() const
  {
    return handle_;
  }

private:
  void reset()
  {
    if (handle_)
    {
      jhps_elliptic_solver_destroy(handle_);
      handle_ = nullptr;
    }
  }

  void* handle_ = nullptr;
};

} // namespace


int main(int argc, char** argv)
{
  try
  {
    const Options options = parse_args(argc, argv);
    const auto meta = read_metadata(options.case_dir / "meta.txt");

    const int format_version = meta_int(meta, "format_version");
    if (format_version != 1)
      throw std::runtime_error("unsupported case format version");

    const int D = meta_int(meta, "D");
    const int n = meta_int(meta, "n");
    const int q_pad = meta_int(meta, "q_pad");
    const int q_vol = meta_int(meta, "q_vol");
    const int q_face = meta_int(meta, "q_face");
    const int p2 = meta_int(meta, "p2");
    const int p1 = meta_int(meta, "p1");
    const int p0 = meta_int(meta, "p0");
    const int assume_symmetric = meta_int(meta, "assume_symmetric");
    const int nverts = meta_int(meta, "nverts");
    const int nelem = meta_int(meta, "nelem");
    const int nmerge = meta_int(meta, "nmerge");
    const int nboundary_faces = meta_int(meta, "nboundary_faces");

    const double tau_C = meta_double(meta, "tau_C");
    const double alpha = meta_double(meta, "alpha");
    const double beta = meta_double(meta, "beta");

    if (D < 1 || D > 5)
      throw std::runtime_error("case D must be in 1..5");
    if (n < 2)
      throw std::runtime_error("case n must be at least 2");
    if (nmerge != nelem - 1)
      throw std::runtime_error("case nmerge must equal nelem-1");

    const std::size_t A_count =
      static_cast<std::size_t>(meta_int(meta, "A_count"));
    const std::size_t b_count =
      static_cast<std::size_t>(meta_int(meta, "b_count"));
    const std::size_t c_count =
      static_cast<std::size_t>(meta_int(meta, "c_count"));
    const std::size_t f_count =
      static_cast<std::size_t>(meta_int(meta, "f_count"));
    const std::size_t g_count =
      static_cast<std::size_t>(meta_int(meta, "g_count"));

    const auto kappa = read_binary<double>(
      options.case_dir / "kappa.f64",
      static_cast<std::size_t>(D + 1));
    const auto vertex_ids = read_binary<std::int32_t>(
      options.case_dir / "vertex_ids.i32",
      static_cast<std::size_t>(nverts));
    const auto coords = read_binary<double>(
      options.case_dir / "coords.f64",
      static_cast<std::size_t>(nverts) * D);
    const auto simplices = read_binary<std::int32_t>(
      options.case_dir / "simplices.i32",
      static_cast<std::size_t>(nelem) * (D + 1));
    const auto merge_pairs = read_binary<std::int32_t>(
      options.case_dir / "merge_pairs.i32",
      static_cast<std::size_t>(nmerge) * 2);
    const auto boundary_keys = read_binary<std::int32_t>(
      options.case_dir / "boundary_keys.i32",
      static_cast<std::size_t>(nboundary_faces) * D);

    const auto A_coeffs = read_binary<double>(
      options.case_dir / "A_coeffs.f64", A_count);
    const auto b_coeffs = read_binary<double>(
      options.case_dir / "b_coeffs.f64", b_count);
    const auto c_coeffs = read_binary<double>(
      options.case_dir / "c_coeffs.f64", c_count);
    const auto f_int = read_binary<double>(
      options.case_dir / "f_int.f64", f_count);
    const auto boundary_g = read_binary<double>(
      options.case_dir / "boundary_g.f64", g_count);

    std::cout << "native persistent elliptic HPS profiling driver\n";
    std::cout << "  case directory        = " << options.case_dir << "\n";
    std::cout << "  D                     = " << D << "\n";
    std::cout << "  degree n              = " << n << "\n";
    std::cout << "  vertices              = " << nverts << "\n";
    std::cout << "  elements/leaves       = " << nelem << "\n";
    std::cout << "  merges                = " << nmerge << "\n";
    std::cout << "  boundary faces        = " << nboundary_faces << "\n";
    std::cout << "  OMP_NUM_THREADS       = "
              << env_or_unset("OMP_NUM_THREADS") << "\n";
    std::cout << "  OPENBLAS_NUM_THREADS  = "
              << env_or_unset("OPENBLAS_NUM_THREADS") << "\n";

    SolverHandle solver;
    int M = 0;
    int m_int = 0;
    int kf = 0;
    int root_nb = 0;
    int interface_nb = 0;
    int leaf_threads_used = 0;

    const auto precompute_begin = std::chrono::steady_clock::now();

    const int create_status = jhps_elliptic_solver_create(
      D,
      n,
      q_pad,
      q_vol,
      q_face,
      kappa.data(),
      p2,
      p1,
      p0,
      assume_symmetric,
      A_coeffs.data(),
      b_coeffs.empty() ? nullptr : b_coeffs.data(),
      c_coeffs.empty() ? nullptr : c_coeffs.data(),
      nverts,
      reinterpret_cast<const int*>(vertex_ids.data()),
      coords.data(),
      nelem,
      reinterpret_cast<const int*>(simplices.data()),
      nmerge,
      reinterpret_cast<const int*>(merge_pairs.data()),
      nboundary_faces,
      reinterpret_cast<const int*>(boundary_keys.data()),
      tau_C,
      alpha,
      beta,
      JHPS_LEAF_OPERATOR_DENSE,
      JHPS_LEAF_LS_DENSE_QR,
      0.0,
      0,
      solver.out_ptr(),
      &M,
      &m_int,
      &kf,
      &root_nb,
      &interface_nb,
      &leaf_threads_used);

    if (create_status != 0 || !solver.get())
      throw std::runtime_error("jhps_elliptic_solver_create failed");

    const auto precompute_end = std::chrono::steady_clock::now();
    const double precompute_seconds =
      std::chrono::duration<double>(
        precompute_end - precompute_begin).count();

    if (f_count != static_cast<std::size_t>(nelem) * m_int)
      throw std::runtime_error("source size does not match solver m_int");
    if (g_count != static_cast<std::size_t>(nboundary_faces) * kf)
      throw std::runtime_error("boundary data size does not match solver kf");

    std::vector<double> leaf_coeffs(
      static_cast<std::size_t>(nelem) * M, 0.0);

    std::cout << "  persistent precompute = "
              << std::fixed << std::setprecision(6)
              << precompute_seconds << " s\n";
    std::cout << "  M                     = " << M << "\n";
    std::cout << "  m_int                 = " << m_int << "\n";
    std::cout << "  kf                    = " << kf << "\n";
    std::cout << "  root trace dofs       = " << root_nb << "\n";
    std::cout << "  interface trace dofs  = " << interface_nb << "\n";
    std::cout << "  leaf threads used     = " << leaf_threads_used << "\n";

    double root_residual = 0.0;
    double interface_residual = 0.0;
    double parent_residual = 0.0;

    const int check_status = jhps_elliptic_solver_solve(
      solver.get(),
      f_int.data(),
      boundary_g.data(),
      1,
      leaf_coeffs.data(),
      &root_residual,
      &interface_residual,
      &parent_residual);

    if (check_status != 0)
      throw std::runtime_error("checked persistent solve failed");

    std::cout << "  checked solve:\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "    root residual       = " << root_residual << "\n";
    std::cout << "    interface residual  = " << interface_residual << "\n";
    std::cout << "    parent residual     = " << parent_residual << "\n";

    // Prebuild a few scaled RHS/boundary cases outside the timed solve loop.
    // The problem is linear, so this perturbs the data without changing the
    // operator/factorization and avoids adding case construction to the timer.
    std::vector<std::vector<double>> f_cases(
      static_cast<std::size_t>(options.rhs_cases));
    std::vector<std::vector<double>> g_cases(
      static_cast<std::size_t>(options.rhs_cases));

    for (int c = 0; c < options.rhs_cases; ++c)
    {
      const double scale = 1.0 + 0.01 * c;

      f_cases[static_cast<std::size_t>(c)].resize(f_int.size());
      std::transform(
        f_int.begin(),
        f_int.end(),
        f_cases[static_cast<std::size_t>(c)].begin(),
        [scale](double value) { return scale * value; });

      g_cases[static_cast<std::size_t>(c)].resize(boundary_g.size());
      std::transform(
        boundary_g.begin(),
        boundary_g.end(),
        g_cases[static_cast<std::size_t>(c)].begin(),
        [scale](double value) { return scale * value; });
    }

    for (int i = 0; i < options.warmup; ++i)
    {
      const int c = i % options.rhs_cases;
      const int status = jhps_elliptic_solver_solve(
        solver.get(),
        f_cases[static_cast<std::size_t>(c)].data(),
        g_cases[static_cast<std::size_t>(c)].data(),
        0,
        leaf_coeffs.data(),
        nullptr,
        nullptr,
        nullptr);

      if (status != 0)
        throw std::runtime_error("warmup persistent solve failed");
    }

    std::vector<double> times(
      static_cast<std::size_t>(options.reps), 0.0);
    double checksum = 0.0;

    for (int rep = 0; rep < options.reps; ++rep)
    {
      const int c = rep % options.rhs_cases;

      const auto begin = std::chrono::steady_clock::now();

      const int status = jhps_elliptic_solver_solve(
        solver.get(),
        f_cases[static_cast<std::size_t>(c)].data(),
        g_cases[static_cast<std::size_t>(c)].data(),
        0,
        leaf_coeffs.data(),
        nullptr,
        nullptr,
        nullptr);

      const auto end = std::chrono::steady_clock::now();

      if (status != 0)
        throw std::runtime_error("timed persistent solve failed");

      times[static_cast<std::size_t>(rep)] =
        std::chrono::duration<double>(end - begin).count();

      checksum += leaf_coeffs.empty() ? 0.0 : leaf_coeffs[0];
    }

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    const double min_time = sorted_times.front();
    const double max_time = sorted_times.back();
    const double median_time =
      sorted_times[sorted_times.size() / 2];
    const double mean_time =
      std::accumulate(times.begin(), times.end(), 0.0)
      / static_cast<double>(times.size());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "online persistent solve timing\n";
    std::cout << "  warmup solves         = " << options.warmup << "\n";
    std::cout << "  timed solves          = " << options.reps << "\n";
    std::cout << "  rhs cases             = " << options.rhs_cases << "\n";
    std::cout << "  min                   = " << min_time << " s\n";
    std::cout << "  median                = " << median_time << " s\n";
    std::cout << "  mean                  = " << mean_time << " s\n";
    std::cout << "  max                   = " << max_time << " s\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  leaves/s (median)     = "
              << static_cast<double>(nelem) / median_time << "\n";
    std::cout << std::setprecision(16);
    std::cout << "  checksum              = " << checksum << "\n";

    return 0;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "profile_hps_elliptic_native failed: "
              << ex.what() << "\n";
    return 1;
  }
}
