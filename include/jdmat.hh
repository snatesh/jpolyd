#ifndef JPOLYD_DMAT_HH
#define JPOLYD_DMAT_HH

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include "jbasis.hh"
#include "jquad_tprod.hh"   // QuadMapped<D,Real>

namespace jsimplex
{

/* stencil encoding non-zero coupling between P_j,P_{j-1} 
   in sparse differrentiation operators */
struct DMatStencil
{
  int axis;        // derivative axis
  int ndelta;      // number of unique delta keys
  uint64_t* keys;  // length ndelta, sorted unique packed deltas

  void clear()
  {
    if (keys) { std::free(keys); keys = nullptr; }
    axis = -1;
    ndelta = 0;
  }
};



template<int D, class Real>
struct DMat
{
  /* Build dense derivative projection matrix:
  
     Dout[i,j] = < phi_rng_i(kappa_rng), d/dx_axis phi_src_j(kappa_src) >_{w(kappa_rng)}
  
     using mapped tensor-product quadrature built for kappa_rng.
  
   Output:
     Dout is row-major MxM, where M = dim_Pi(n).
  
   Notes:
   - No pruning here (we’ll use this to corroborate which kappa_rng choices yield sparsity).
   - eval_all is called with a valid V pointer (not nullptr) because your implementation
     likely expects it. */
  static void build_tprod(int n,
                          unsigned int q,
                          const Real* kappa_src,   // length D+1
                          const Real* kappa_rng,   // length D+1
                          int axis,                // 0..D-1
                          Real* Dout)              // (M*M) row-major
  {
    if (!kappa_src || !kappa_rng || !Dout)
    {
      std::cerr << "DMat::build_tprod: null input\n";
      std::exit(1);
    }
    if (n < 1)
    {
      std::cerr << "DMat::build_tprod: require n >= 1\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "DMat::build_tprod: require q >= 1\n";
      std::exit(1);
    }
    if (axis < 0 || axis >= D)
    {
      std::cerr << "DMat::build_tprod: axis out of range\n";
      std::exit(1);
    }

    const int M = Basis<D,Real>::dim_Pi(n);
    std::memset(Dout, 0, (std::size_t)M * (std::size_t)M * sizeof(Real));

    // ---- Build κ-aware mapped quadrature for the range weight ----
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;

    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::cerr << "DMat::build_tprod: malloc quad failed\n";
      std::exit(1);
    }

    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_rng, X, wq);
    if (built != npts)
    {
      std::cerr << "DMat::build_tprod: build_kappa failed\n";
      std::exit(1);
    }

    // Normalize weights (matches your convention; helps conditioning)
    Real sw = Real(0);
    for (int p = 0; p < npts; ++p)
    {
      sw += wq[p];
    }
    if (sw != Real(0))
    {
      const Real inv_sw = Real(1) / sw;
      for (int p = 0; p < npts; ++p)
      {
        wq[p] *= inv_sw;
      }
    }

    // ---- Basis tables for degree n ----
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* invh_src   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* invh_rng   = (Real*) std::malloc((std::size_t)M * sizeof(Real));

    if (!alpha_table || !tail_deg || !invh_src || !invh_rng)
    {
      std::cerr << "DMat::build_tprod: malloc tables failed\n";
      std::exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);

    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + m * D;
      invh_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      invh_rng[m] = Basis<D,Real>::inv_h_alpha(a, kappa_rng);
    }

    // ---- Evaluate values and gradients ----
    // Layout:
    //   V[p + m*ldV], ldV = npts
    //   dV[(p + m*ldV)*D + ell], ell=0..D-1
    const int ldV = npts;

    Real* Vrng  = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vsrc  = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* dVsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * (std::size_t)D * sizeof(Real));

    if (!Vrng || !Vsrc || !dVsrc)
    {
      std::cerr << "DMat::build_tprod: malloc V/dV failed\n";
      std::exit(1);
    }

    // Range values
    Basis<D,Real>::eval_all(
      X,
      D, 1,     // ld_point, ld_dim for AoS X[p*D + j]
      npts,
      kappa_rng,
      n,
      alpha_table,
      tail_deg,
      invh_rng,
      Vrng,
      ldV,
      nullptr
    );

    // Source values + analytic gradients
    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa_src,
      n,
      alpha_table,
      tail_deg,
      invh_src,
      Vsrc,
      ldV,
      dVsrc
    );

    // ---- Assemble: Dout = Vrng^T * diag(wq) * (d/dx_axis Vsrc) ----
    // dVsrc axis slice: dVsrc[(p + j*ldV)*D + axis]
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
    {
      const Real* vi = Vrng + (std::size_t)i * (std::size_t)ldV;

      for (int j = 0; j < M; ++j)
      {
        const Real* dVj = dVsrc + (std::size_t)j * (std::size_t)ldV * (std::size_t)D;

        Real s = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          s += vi[p] * wq[p] * dVj[(std::size_t)p * (std::size_t)D + (std::size_t)axis];
        }

        Dout[(std::size_t)i * (std::size_t)M + (std::size_t)j] = s;
      }
    }

    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_rng);
    std::free(Vrng);
    std::free(Vsrc);
    std::free(dVsrc);
  }
  
  /* Natural parameter shift for coordinate derivative:
       kappa_rng = kappa_src + e_axis + e_D  (where last index is D)
    
     Builds dense Dout via build_tprod(...), then:
       (1) enforces the exact degree-drop nullspace: rows for total degree n are set to 0
           (i.e. rows i >= dim_Pi(n-1) are identically zero for a first derivative)
       (2) applies row-relative pruning ONLY on the active rows:
           |D_ij| < (100*eps*||row_i||_2) => 0 */
  static void build_tprod_natural_pruned_dense(int n,
                                               unsigned int q,
                                               const Real* kappa_src,  // length D+1
                                               int axis,               // 0..D-1
                                               Real* Dout)             // (M*M) row-major
  {
    // to avoid flickering deltas in pattern determination
    // probably not necessary, but for safety.
    if (!kappa_src || !Dout)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: null input\n";
      std::exit(1);
    }
    if (n < 1)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: require n >= 1\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: require q >= 1\n";
      std::exit(1);
    }
    if (axis < 0 || axis >= D)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: axis out of range\n";
      std::exit(1);
    }
  
    // Build kappa_rng = kappa_src + e_axis + e_last
    Real kappa_rng[D + 1];
    for (int i = 0; i < D + 1; ++i)
    {
      kappa_rng[i] = kappa_src[i];
    }
    kappa_rng[axis] += Real(1);
    kappa_rng[D]    += Real(1);
  
    build_tprod(n, q, kappa_src, kappa_rng, axis, Dout);
  
    const int M     = Basis<D,Real>::dim_Pi(n);
    const int M_nm1 = Basis<D,Real>::dim_Pi(n - 1);
  
    // (1) Enforce exact degree-drop nullspace: rows for total degree n are zero.
    //#pragma omp parallel for schedule(static)
    for (int i = M_nm1; i < M; ++i)
    {
      Real* row = Dout + (std::size_t)i * (std::size_t)M;
      for (int j = 0; j < M; ++j)
      {
        row[j] = Real(0);
      }
    }
  
    // Compute a global scale for absolute pruning.
    // (Infinity norm of matrix; cheap enough at these sizes.)
    Real max_abs = Real(0);
    for (int idx = 0; idx < M * M; ++idx)
    {
      const Real a = std::abs(Dout[idx]);
      if (a > max_abs) max_abs = a;
    }
  
    const Real eps = std::numeric_limits<Real>::epsilon();
  
    // Row-relative factor (your choice)
    const Real rel_factor = Real(1000) * eps;
  
    // Absolute floor: scale-aware, but prevents 1e-13 dust from surviving on tiny rows.
    // You can tune the multiplier; this is conservative.
    const Real abs_factor = Real(1e3) * eps;
    const Real abs_floor  = abs_factor * (max_abs > Real(0) ? max_abs : Real(1));
  
    // (2) Prune active rows.
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < M_nm1; ++i)
    {
      Real* row = Dout + (std::size_t)i * (std::size_t)M;
  
      Real n2 = Real(0);
      for (int j = 0; j < M; ++j)
      {
        const Real v = row[j];
        n2 += v * v;
      }
  
      const Real row_norm = std::sqrt(n2);
      if (row_norm == Real(0))
      {
        continue;
      }
  
      const Real tol = std::max(rel_factor * row_norm, abs_floor);
  
      for (int j = 0; j < M; ++j)
      {
        if (std::abs(row[j]) < tol)
        {
          row[j] = Real(0);
        }
      }
    }
  }

  /* Natural parameter shift for coordinate derivative:
       kappa_rng = kappa_src + e_axis + e_D  (where last index is D)
   
     if n < stencil_min 
     Builds dense Dout via build_tprod(...), then:
       (1) enforces the exact degree-drop nullspace: rows for total degree n are set to 0
           (i.e. rows i >= dim_Pi(n-1) are identically zero for a first derivative)
       (2) applies row-relative pruning ONLY on the active rows:
           |D_ij| < (100*eps*||row_i||_2) => 0 
     otherwise
       Compute stable stencil pattern for n between stencil_min,stencil_max 
       Then evaluate dmat entries only where indicated by non-zero stencil */
  static void build_tprod_natural_pruned(int n,
                                         unsigned int q,
                                         const Real* kappa_src,
                                         int axis,
                                         Real* Dout)
  {
    build_tprod_natural_pruned(
      n, q, kappa_src, axis, Dout, default_stencil_folder());
  }

  static void build_tprod_natural_pruned(int n,
                                         unsigned int q,
                                         const Real* kappa_src,
                                         int axis,
                                         Real* Dout,
                                         const std::string& stencil_folder)
  {
    if (!kappa_src || !Dout)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: null input\n";
      std::exit(1);
    }
    if (n < 0)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: n must be non-negative\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: require q >= 1\n";
      std::exit(1);
    }
    if (axis < 0 || axis >= D)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: axis out of range\n";
      std::exit(1);
    }
    if (n == 0)
    {
      Dout[0] = Real(0);
      return;
    }

    DMatStencil S;
    std::memset(&S, 0, sizeof(S));
    load_or_discover_natural_stencil(
      q, kappa_src, axis, &S, stencil_folder);
    build_tprod_from_deltas(n, q, kappa_src, axis, S, Dout);
    S.clear();
  }

  
  /* Given total degree j and local index k in Hom(j)
     extract the corresponding row from alpha_table */
  static inline const int* hom_decode_ptr(int j,
                                          int k,
                                          const int* alpha_table) // (M x D)
  {
    const int m0 = Basis<D,Real>::dim_Pi(j - 1);
    const int m  = m0 + k;
    return alpha_table + (std::size_t)m * (std::size_t)D;
  }

  /* overflow safety for n choose k */ 
  static inline uint64_t choose_u64(int n, int k)
  {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
  
    // Exact integer computation using __int128 to reduce overflow risk.
    __int128 acc = 1;
    for (int i = 1; i <= k; ++i)
    {
      const int num = n - k + i;
      const int den = i;
      acc = acc * (__int128)num;
      acc = acc / (__int128)den;
    }
  
    if (acc < 0) return 0;
    if (acc > (__int128)std::numeric_limits<uint64_t>::max())
    {
      std::cerr << "choose_u64 overflow: n=" << n << " k=" << k << "\n";
      std::exit(1);
    }
  
    return (uint64_t)acc;
  }
  
  /* Given multi-index a with total degree j
     get its position k in Hom(j) 
    
     Fast rank alpha (sum=j) in Hom(j) under fill_degree_rec order:
     for each coord: a = rem..0 (descending), recurse. Last coord forced. */
  static inline int hom_encode_rank_fast(int j, const int* alpha) // length D
  {
    uint64_t rank = 0;
    int rem = j;
  
    for (int coord = 0; coord < D - 1; ++coord)
    {
      const int a = alpha[coord];
  
      // Skipped values at this coord are ap = rem, rem-1, ..., a+1.
      // Let k = number of skipped ap values = rem - a.
      // The contribution is:
      //   sum_{ap=a+1..rem} C((rem-ap) + p - 1, p - 1)
      // where p = parts_left = D-(coord+1).
      //
      // Using hockey-stick:
      //   sum_{t=0..(rem-a-1)} C(t + p - 1, p - 1) = C((rem-a-1)+p, p)
      const int p = D - (coord + 1);
      const int k = rem - a - 1;
  
      if (k >= 0)
      {
        rank += choose_u64(k + p, p);
      }
  
      rem -= a;
    }
  
    if (rank > (uint64_t)std::numeric_limits<int>::max())
    {
      std::cerr << "hom_encode_rank_fast: rank overflow int\n";
      std::exit(1);
    }
  
    return (int)rank;
  }
 

  static inline uint64_t pack_delta8(const int* dvec) // length D
  {
    // Works for D <= 8.
    uint64_t key = 0;
    for (int i = 0; i < D; ++i)
    {
      const int v = dvec[i] + 64; // bias to nonnegative
      // Optional hard check (recommended in debug)
      // if (v < 0 || v > 255) { std::cerr << "pack_delta8: out of range\n"; std::exit(1); }
  
      key = (key << 8) | (uint64_t)(v & 0xFF);
    }
    return key;
  }

  static inline void unpack_delta8(uint64_t key, int* dvec) // length D
  {
    // key stores D bytes, highest-order byte corresponds to dvec[0].
    for (int i = D - 1; i >= 0; --i)
    {
      const int v = (int)(key & 0xFF);
      dvec[i] = v - 64;
      key >>= 8;
    }
  }

  static inline bool deltas_equal(const DMatStencil& A, const DMatStencil& B)
  {
    if (A.ndelta != B.ndelta) return false;
    if (A.ndelta == 0) return true;
    return std::memcmp(A.keys, B.keys, (std::size_t)A.ndelta * sizeof(uint64_t)) == 0;
  }


  static constexpr int stencil_cache_version()
  {
    return 1;
  }

  static constexpr int default_stencil_min_degree()
  {
    return D + 1;
  }

  static constexpr int default_stencil_max_degree()
  {
    return 4 * D;
  }

  static std::string default_stencil_folder()
  {
    return "stencils";
  }

  /* Canonical structural signature used by jprecomp for interning.
     The derivative axis is intentionally omitted: two axes with identical
     delta lists should intern to one shared structural stencil. */
  static std::string stencil_signature(const DMatStencil& S)
  {
    std::ostringstream out;
    out << "DMat:D=" << D << ":ndelta=" << S.ndelta << ":keys=";
    out << std::hex << std::setfill('0');
    for (int k = 0; k < S.ndelta; ++k)
    {
      if (k) out << ',';
      out << std::setw(16) << S.keys[k];
    }
    return out.str();
  }

  static bool stencil_valid(const DMatStencil& S, int expected_axis = -1)
  {
    if (expected_axis >= 0 && S.axis != expected_axis) return false;
    if (S.axis < 0 || S.axis >= D) return false;
    if (S.ndelta < 0) return false;
    if (S.ndelta > 0 && !S.keys) return false;
    for (int k = 1; k < S.ndelta; ++k)
    {
      if (!(S.keys[k - 1] < S.keys[k])) return false;
    }
    return true;
  }

  static void copy_stencil(const DMatStencil& src, DMatStencil* dst)
  {
    if (!dst || !stencil_valid(src))
    {
      std::cerr << "DMat::copy_stencil: invalid input\n";
      std::exit(1);
    }
    dst->clear();
    dst->axis = src.axis;
    dst->ndelta = src.ndelta;
    if (src.ndelta == 0)
    {
      dst->keys = nullptr;
      return;
    }
    dst->keys = static_cast<uint64_t*>(
      std::malloc(static_cast<std::size_t>(src.ndelta) * sizeof(uint64_t)));
    if (!dst->keys)
    {
      std::cerr << "DMat::copy_stencil: allocation failed\n";
      std::exit(1);
    }
    std::memcpy(
      dst->keys,
      src.keys,
      static_cast<std::size_t>(src.ndelta) * sizeof(uint64_t));
  }

  static std::filesystem::path stencil_cache_path(
    int axis,
    const std::string& stencil_folder = "stencils")
  {
    if (axis < 0 || axis >= D)
    {
      std::cerr << "DMat::stencil_cache_path: axis out of range\n";
      std::exit(1);
    }
    const std::filesystem::path folder =
      stencil_folder.empty() ? std::filesystem::path(".")
                             : std::filesystem::path(stencil_folder);
    std::ostringstream name;
    name << "dmat_D" << D
         << "_axis" << axis
         << "_v" << stencil_cache_version()
         << ".stencil";
    return folder / name.str();
  }

  /* Return false only when the keyed file is absent.  A present but invalid
     file is a hard error so stale/corrupt cache data is never used silently. */
  static bool load_stencil_file(
    int axis,
    DMatStencil* S_out,
    const std::string& stencil_folder = "stencils")
  {
    if (!S_out)
    {
      std::cerr << "DMat::load_stencil_file: null output\n";
      std::exit(1);
    }
    const std::filesystem::path path =
      stencil_cache_path(axis, stencil_folder);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
      if (ec)
      {
        std::cerr << "DMat::load_stencil_file: exists failed for "
                  << path << ": " << ec.message() << '\n';
        std::exit(1);
      }
      return false;
    }

    std::ifstream in(path);
    std::string magic;
    std::string kind;
    std::string basis_tag;
    int version = 0;
    int file_D = -1;
    int file_axis = -1;
    int ndelta = -1;
    if (!(in >> magic >> version >> kind >> basis_tag
             >> file_D >> file_axis >> ndelta))
    {
      std::cerr << "DMat::load_stencil_file: malformed header in "
                << path << '\n';
      std::exit(1);
    }
    if (magic != "JPOLYD_STENCIL" ||
        version != stencil_cache_version() ||
        kind != "DMAT" ||
        basis_tag != "KAPPA_MINUS_HALF" ||
        file_D != D || file_axis != axis || ndelta < 0)
    {
      std::cerr << "DMat::load_stencil_file: incompatible cache file "
                << path << '\n';
      std::exit(1);
    }

    DMatStencil loaded;
    std::memset(&loaded, 0, sizeof(loaded));
    loaded.axis = axis;
    loaded.ndelta = ndelta;
    if (ndelta > 0)
    {
      loaded.keys = static_cast<uint64_t*>(
        std::malloc(static_cast<std::size_t>(ndelta) * sizeof(uint64_t)));
      if (!loaded.keys)
      {
        std::cerr << "DMat::load_stencil_file: allocation failed\n";
        std::exit(1);
      }
      for (int k = 0; k < ndelta; ++k)
      {
        if (!(in >> loaded.keys[k]))
        {
          loaded.clear();
          std::cerr << "DMat::load_stencil_file: truncated key list in "
                    << path << '\n';
          std::exit(1);
        }
      }
    }
    std::string end_token;
    if (!(in >> end_token) || end_token != "END" ||
        !stencil_valid(loaded, axis))
    {
      loaded.clear();
      std::cerr << "DMat::load_stencil_file: invalid stencil in "
                << path << '\n';
      std::exit(1);
    }

    S_out->clear();
    *S_out = loaded;
    std::memset(&loaded, 0, sizeof(loaded));
    return true;
  }

  static void save_stencil_file(
    int axis,
    const DMatStencil& S,
    const std::string& stencil_folder = "stencils")
  {
    if (!stencil_valid(S, axis))
    {
      std::cerr << "DMat::save_stencil_file: invalid stencil\n";
      std::exit(1);
    }
    const std::filesystem::path path =
      stencil_cache_path(axis, stencil_folder);
    const std::filesystem::path folder = path.parent_path();
    std::error_code ec;
    if (!folder.empty())
    {
      std::filesystem::create_directories(folder, ec);
      if (ec)
      {
        std::cerr << "DMat::save_stencil_file: cannot create "
                  << folder << ": " << ec.message() << '\n';
        std::exit(1);
      }
    }

    const std::filesystem::path temporary = path.string() + ".tmp";
    {
      std::ofstream out(temporary, std::ios::trunc);
      if (!out)
      {
        std::cerr << "DMat::save_stencil_file: cannot open "
                  << temporary << '\n';
        std::exit(1);
      }
      out << "JPOLYD_STENCIL " << stencil_cache_version()
          << " DMAT KAPPA_MINUS_HALF "
          << D << ' ' << axis << ' ' << S.ndelta << '\n';
      for (int k = 0; k < S.ndelta; ++k)
      {
        out << S.keys[k] << '\n';
      }
      out << "END\n";
      out.flush();
      if (!out)
      {
        std::cerr << "DMat::save_stencil_file: write failed for "
                  << temporary << '\n';
        std::exit(1);
      }
    }

    std::filesystem::rename(temporary, path, ec);
    if (ec)
    {
      std::error_code remove_ec;
      std::filesystem::remove(path, remove_ec);
      ec.clear();
      std::filesystem::rename(temporary, path, ec);
    }
    if (ec)
    {
      std::cerr << "DMat::save_stencil_file: rename failed for "
                << path << ": " << ec.message() << '\n';
      std::exit(1);
    }
  }

  /* Main persistent path.  Returns true when loaded and false when newly
     discovered and written. */
  static bool load_or_discover_natural_stencil(
    unsigned int q,
    const Real* kappa_src,
    int axis,
    DMatStencil* S_out,
    const std::string& stencil_folder = "stencils",
    int n_min = default_stencil_min_degree(),
    int n_max = default_stencil_max_degree())
  {
    if (!kappa_src || !S_out)
    {
      std::cerr << "DMat::load_or_discover_natural_stencil: null input\n";
      std::exit(1);
    }
    if (load_stencil_file(axis, S_out, stencil_folder)) return true;
    discover_stencil_stable(q, kappa_src, axis, n_min, n_max, S_out);
    save_stencil_file(axis, *S_out, stencil_folder);
    return false;
  }

  static int cmp_u64(const void* a, const void* b)
  {
    const uint64_t A = *(const uint64_t*)a;
    const uint64_t B = *(const uint64_t*)b;
    if (A < B) return -1;
    if (A > B) return 1;
    return 0;
  }
  
  static void extract_deltas_from_block(int n_test,
                                        int j_rep,
                                        int axis,
                                        const Real* Ddense,      // (M x M) row-major, pruned
                                        const int* alpha_table,  // (M x D)
                                        DMatStencil* S)
  {
    const int M = Basis<D,Real>::dim_Pi(n_test);
  
    if (!S || !Ddense || !alpha_table)
    {
      std::cerr << "extract_deltas_from_block: null input\n";
      std::exit(1);
    }
    if (j_rep < 1 || j_rep > n_test)
    {
      std::cerr << "extract_deltas_from_block: invalid j_rep\n";
      std::exit(1);
    }
  
    // Block starts/sizes in global indexing
    const int row0 = Basis<D,Real>::dim_Pi(j_rep - 2); // degree (j_rep-1)
    const int col0 = Basis<D,Real>::dim_Pi(j_rep - 1); // degree (j_rep)
    const int r    = Basis<D,Real>::dim_Hom(j_rep - 1);
    const int c    = Basis<D,Real>::dim_Hom(j_rep);
  
    // Worst-case nnz in the block is r*c (tiny at discovery degrees)
    uint64_t* tmp = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)c * sizeof(uint64_t));
    if (!tmp)
    {
      std::cerr << "extract_deltas_from_block: alloc failed\n";
      std::exit(1);
    }
  
    int ntmp = 0;
    int dvec[8]; // supports D up to 8 (your D<=7 target)
  
    for (int i_loc = 0; i_loc < r; ++i_loc)
    {
      const int i = row0 + i_loc;
  
      for (int j_loc = 0; j_loc < c; ++j_loc)
      {
        const int j = col0 + j_loc;
        const Real a = Ddense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* dst = hom_decode_ptr(j_rep - 1, i_loc, alpha_table); // degree j_rep-1
        const int* src = hom_decode_ptr(j_rep,     j_loc, alpha_table); // degree j_rep
  
        for (int d = 0; d < D; ++d) { dvec[d] = dst[d] - src[d]; }
  
        tmp[ntmp++] = pack_delta8(dvec);
      }
    }
  
    // Sort + unique
    if (ntmp > 1) { std::qsort(tmp, (std::size_t)ntmp, sizeof(uint64_t), cmp_u64); }
  
    int nu = 0;
    for (int k = 0; k < ntmp; ++k)
    {
      if (k == 0 || tmp[k] != tmp[k - 1]) { tmp[nu++] = tmp[k]; }
    }
  
    S->clear();
    S->axis = axis;
    S->ndelta = nu;
    if (nu == 0)
    {
      S->keys = nullptr;
      std::free(tmp);
      return;
    }
  
    S->keys = (uint64_t*) std::malloc((std::size_t)nu * sizeof(uint64_t));
    if (!S->keys)
    {
      std::cerr << "extract_deltas_from_block: alloc keys failed\n";
      std::exit(1);
    }
    std::memcpy(S->keys, tmp, (std::size_t)nu * sizeof(uint64_t));
  
    std::free(tmp);
  }

  static void discover_stencil_stable(unsigned int q,
                                      const Real* kappa_src,
                                      int axis,
                                      int n_min,
                                      int n_max,
                                      DMatStencil* S_out)
  {
    if (!S_out)
    {
      std::cerr << "discover_stencil_stable: null S_out\n";
      std::exit(1);
    }
  
    DMatStencil S_prev; std::memset(&S_prev, 0, sizeof(S_prev));
    DMatStencil S_cur;  std::memset(&S_cur,  0, sizeof(S_cur));
  
    for (int n_test = n_min; n_test <= n_max; ++n_test)
    {
      // Use second-to-last block: j_rep = n_test - 1
      const int j_rep = n_test - 1;
      if (j_rep < 1) continue;
  
      const int M = Basis<D,Real>::dim_Pi(n_test);
  
      // Dense pruned matrix
      Real* Ddense = (Real*) std::malloc((std::size_t)M * (std::size_t)M * sizeof(Real));
      int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      if (!Ddense || !alpha_table)
      {
        std::cerr << "discover_stencil_stable: alloc failed\n";
        std::exit(1);
      }
  
      // Build pruned dense (your existing routine)
      build_tprod_natural_pruned_dense(n_test, q, kappa_src, axis, Ddense); 
  
      // Build alpha_table for this n_test
      Basis<D,Real>::build_alpha_table(n_test, alpha_table);
  
      // Extract delta set from block (j_rep -> j_rep-1)
      extract_deltas_from_block(n_test, j_rep, axis, Ddense, alpha_table, &S_cur);
  
      std::free(Ddense);
      std::free(alpha_table);
  
      if (n_test > n_min && deltas_equal(S_cur, S_prev))
      {
        // stabilized
        S_prev.clear();
        S_out->clear();
        *S_out = S_cur;                // shallow move
        std::memset(&S_cur, 0, sizeof(S_cur));
        std::cout << "stabilized at n_test = " << n_test << std::endl;
        return;
      }
  
      S_prev.clear();
      S_prev = S_cur;                  // shallow move
      std::memset(&S_cur, 0, sizeof(S_cur));
    }
  
    std::cerr << "DMat: delta-stencil did not stabilize up to n_max\n";
    std::exit(1);
  }


  static void build_tprod_from_deltas(int n,
                                     unsigned int q,
                                     const Real* kappa_src,
                                     int axis,
                                     const DMatStencil& S,
                                     Real* Dout) // MxM row-major, dense for testing
  {
    if (!kappa_src || !Dout)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: null input\n";
      std::exit(1);
    }
    if (n < 1)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: require n >= 1\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: require q >= 1\n";
      std::exit(1);
    }
    if (axis < 0 || axis >= D)
    {
      std::cerr << "DMat::build_tprod_natural_pruned: axis out of range\n";
      std::exit(1);
    }
  
    // Build kappa_rng = kappa_src + e_axis + e_last
    Real kappa_rng[D + 1];
    for (int i = 0; i < D + 1; ++i)
    {
      kappa_rng[i] = kappa_src[i];
    }
    kappa_rng[axis] += Real(1);
    kappa_rng[D]    += Real(1);
    
    const int M = Basis<D,Real>::dim_Pi(n);
    std::memset(Dout, 0, (std::size_t)M * (std::size_t)M * sizeof(Real));

    // ---- Build κ-aware mapped quadrature for the range weight ----
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;

    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::cerr << "DMat::build_tprod: malloc quad failed\n";
      std::exit(1);
    }

    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_rng, X, wq);
    if (built != npts)
    {
      std::cerr << "DMat::build_tprod: build_kappa failed\n";
      std::exit(1);
    }

    Real sw = Real(0);
    for (int p = 0; p < npts; ++p)
    {
      sw += wq[p];
    }
    if (sw != Real(0))
    {
      const Real inv_sw = Real(1) / sw;
      for (int p = 0; p < npts; ++p)
      {
        wq[p] *= inv_sw;
      }
    }

    // ---- Basis tables for degree n ----
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* invh_src   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* invh_rng   = (Real*) std::malloc((std::size_t)M * sizeof(Real));

    if (!alpha_table || !tail_deg || !invh_src || !invh_rng)
    {
      std::cerr << "DMat::build_tprod: malloc tables failed\n";
      std::exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);

    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + m * D;
      invh_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      invh_rng[m] = Basis<D,Real>::inv_h_alpha(a, kappa_rng);
    }

    // ---- Evaluate values and gradients ----
    // Layout:
    //   V[p + m*ldV], ldV = npts
    //   dV[(p + m*ldV)*D + ell], ell=0..D-1
    const int ldV = npts;

    Real* Vrng  = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vsrc  = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* dVsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * (std::size_t)D * sizeof(Real));

    if (!Vrng || !Vsrc || !dVsrc)
    {
      std::cerr << "DMat::build_tprod: malloc V/dV failed\n";
      std::exit(1);
    }

    // Range values
    Basis<D,Real>::eval_all(
      X,
      D, 1,     // ld_point, ld_dim for AoS X[p*D + j]
      npts,
      kappa_rng,
      n,
      alpha_table,
      tail_deg,
      invh_rng,
      Vrng,
      ldV,
      nullptr
    );

    // Source values + analytic gradients
    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa_src,
      n,
      alpha_table,
      tail_deg,
      invh_src,
      Vsrc,
      ldV,
      dVsrc
    );
  
    // === Assemble using delta stencil ===
    int dvec[8];
    int dst_alpha[8];
  
    for (int jdeg = 1; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int c    = Basis<D,Real>::dim_Hom(jdeg);
  
      const int row0 = Basis<D,Real>::dim_Pi(jdeg - 2);
      const int r    = Basis<D,Real>::dim_Hom(jdeg - 1);
      
      // build inverse map for Hom(jdeg-1)
      //HomInvMap inv_dst;
      //build_hom_invmap(jdeg - 1, alpha_table, inv_dst);  
      for (int jloc = 0; jloc < c; ++jloc)
      {
        const int jg = col0 + jloc; // global column index
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
  
        for (int t = 0; t < S.ndelta; ++t)
        {
          unpack_delta8(S.keys[t], dvec);
  
          // dst_alpha = src + dvec
          bool ok = true;
          int sum = 0;
          for (int d = 0; d < D; ++d)
          {
            dst_alpha[d] = src[d] + dvec[d];
            if (dst_alpha[d] < 0) { ok = false; break; }
            sum += dst_alpha[d];
          }
          if (!ok) continue;
          if (sum != jdeg - 1) continue; // should hold for derivative deltas
          const int iloc = hom_encode_rank_fast(jdeg - 1, dst_alpha);
          if (iloc < 0) continue;        // shouldn't happen
          if (iloc >= r) continue;       // safety
  
          const int ig = row0 + iloc;    // global row index
  
          // Compute entry (ig, jg) using the same quadrature dot product as dense build_tprod:
          const Real* vi = Vrng + (std::size_t)ig * (std::size_t)ldV;
          const Real* dVj = dVsrc + (std::size_t)jg * (std::size_t)ldV * (std::size_t)D;
  
          Real s = Real(0);
          for (int p = 0; p < npts; ++p)
          {
            s += vi[p] * wq[p] * dVj[(std::size_t)p * (std::size_t)D + (std::size_t)axis];
          }
  
          Dout[(std::size_t)ig * (std::size_t)M + (std::size_t)jg] = s;
        }
      }
    }
    
    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_rng);
    std::free(Vrng);
    std::free(Vsrc);
    std::free(dVsrc);
  }

  /* Build only the finite-degree CSC structure obtained by truncating a
     degree-independent delta stencil.  No quadrature or parameter values are
     used. */
  static std::size_t build_natural_csc_pattern_from_stencil(
    int n,
    const DMatStencil& S,
    int** colptr_out,
    int** rowind_out)
  {
    if (!colptr_out || !rowind_out || !stencil_valid(S))
    {
      std::cerr << "DMat::build_natural_csc_pattern_from_stencil: invalid input\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    if (n < 0)
    {
      std::cerr << "DMat::build_natural_csc_pattern_from_stencil: n < 0\n";
      std::exit(1);
    }

    const int ncol = Basis<D,Real>::dim_Pi(n);
    const int nrow = (n == 0) ? 0 : Basis<D,Real>::dim_Pi(n - 1);
    int* colptr = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(ncol + 1) * sizeof(int)));
    if (!colptr)
    {
      std::cerr << "DMat::build_natural_csc_pattern_from_stencil: allocation failed\n";
      std::exit(1);
    }
    colptr[0] = 0;
    if (n == 0)
    {
      colptr[1] = 0;
      *colptr_out = colptr;
      return 0;
    }

    const int M = ncol;
    int* alpha_table = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(M) * D * sizeof(int)));
    if (!alpha_table)
    {
      std::free(colptr);
      std::cerr << "DMat::build_natural_csc_pattern_from_stencil: allocation failed\n";
      std::exit(1);
    }
    Basis<D,Real>::build_alpha_table(n, alpha_table);

    std::vector<int> rows;
    rows.reserve(static_cast<std::size_t>(ncol) *
                 static_cast<std::size_t>(S.ndelta));
    int dvec[8];
    int dst_alpha[8];

    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int cdeg = Basis<D,Real>::dim_Hom(jdeg);
      for (int jloc = 0; jloc < cdeg; ++jloc)
      {
        const int jg = col0 + jloc;
        if (jdeg >= 1)
        {
          const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
          const int row0 = Basis<D,Real>::dim_Pi(jdeg - 2);
          const int r = Basis<D,Real>::dim_Hom(jdeg - 1);
          for (int t = 0; t < S.ndelta; ++t)
          {
            unpack_delta8(S.keys[t], dvec);
            bool ok = true;
            int sum = 0;
            for (int dim = 0; dim < D; ++dim)
            {
              dst_alpha[dim] = src[dim] + dvec[dim];
              if (dst_alpha[dim] < 0) { ok = false; break; }
              sum += dst_alpha[dim];
            }
            if (!ok || sum != jdeg - 1) continue;
            const int iloc = hom_encode_rank_fast(jdeg - 1, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
            const int row = row0 + iloc;
            if (row >= 0 && row < nrow) rows.push_back(row);
          }
        }
        colptr[jg + 1] = static_cast<int>(rows.size());
      }
    }

    int* rowind = nullptr;
    if (!rows.empty())
    {
      rowind = static_cast<int*>(
        std::malloc(rows.size() * sizeof(int)));
      if (!rowind)
      {
        std::free(alpha_table);
        std::free(colptr);
        std::cerr << "DMat::build_natural_csc_pattern_from_stencil: allocation failed\n";
        std::exit(1);
      }
      std::memcpy(rowind, rows.data(), rows.size() * sizeof(int));
    }
    std::free(alpha_table);
    *colptr_out = colptr;
    *rowind_out = rowind;
    return rows.size();
  }

  /* Fill numerical values on an already supplied CSC structure.  The pattern
     may be an exact stencil or a union stencil; structurally absent entries
     simply receive their numerical (usually zero) quadrature value. */
  static void fill_tprod_natural_csc_values(
    int n,
    unsigned int q,
    const Real* kappa_src,
    int axis,
    const int* colptr,
    const int* rowind,
    Real* values)
  {
    if (!kappa_src || !colptr)
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: null input\n";
      std::exit(1);
    }
    if (n < 0 || q == 0 || axis < 0 || axis >= D)
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: invalid argument\n";
      std::exit(1);
    }
    const int ncol = Basis<D,Real>::dim_Pi(n);
    const int nrow = (n == 0) ? 0 : Basis<D,Real>::dim_Pi(n - 1);
    if (colptr[0] != 0)
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: colptr[0] != 0\n";
      std::exit(1);
    }
    for (int j = 0; j < ncol; ++j)
    {
      if (colptr[j + 1] < colptr[j])
      {
        std::cerr << "DMat::fill_tprod_natural_csc_values: invalid colptr\n";
        std::exit(1);
      }
    }
    const int nnz = colptr[ncol];
    if (nnz > 0 && (!rowind || !values))
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: null nnz arrays\n";
      std::exit(1);
    }
    for (int p = 0; p < nnz; ++p)
    {
      if (rowind[p] < 0 || rowind[p] >= nrow)
      {
        std::cerr << "DMat::fill_tprod_natural_csc_values: row out of range\n";
        std::exit(1);
      }
    }
    if (n == 0) return;

    Real kappa_rng[D + 1];
    for (int r = 0; r < D + 1; ++r) kappa_rng[r] = kappa_src[r];
    kappa_rng[axis] += Real(1);
    kappa_rng[D] += Real(1);

    const int M = ncol;
    const int npts = static_cast<int>(QuadMapped<D,Real>::npoints(q));
    Real* X = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * D * sizeof(Real)));
    Real* wq = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * sizeof(Real)));
    int* alpha_table = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(M) * D * sizeof(int)));
    int* tail_deg = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(M) * D * sizeof(int)));
    Real* invh_src = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(M) * sizeof(Real)));
    Real* invh_rng = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(M) * sizeof(Real)));
    Real* Vrng = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * M * sizeof(Real)));
    Real* Vsrc = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * M * sizeof(Real)));
    Real* dVsrc = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * M * D * sizeof(Real)));
    if (!X || !wq || !alpha_table || !tail_deg || !invh_src ||
        !invh_rng || !Vrng || !Vsrc || !dVsrc)
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: allocation failed\n";
      std::exit(1);
    }

    if (QuadMapped<D,Real>::build_kappa(q, kappa_rng, X, wq) != npts)
    {
      std::cerr << "DMat::fill_tprod_natural_csc_values: quadrature failed\n";
      std::exit(1);
    }
    Real sum_w = Real(0);
    for (int p = 0; p < npts; ++p) sum_w += wq[p];
    if (sum_w != Real(0))
    {
      const Real inv_sum_w = Real(1) / sum_w;
      for (int p = 0; p < npts; ++p) wq[p] *= inv_sum_w;
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);
    for (int m = 0; m < M; ++m)
    {
      const int* alpha = alpha_table + static_cast<std::size_t>(m) * D;
      invh_src[m] = Basis<D,Real>::inv_h_alpha(alpha, kappa_src);
      invh_rng[m] = Basis<D,Real>::inv_h_alpha(alpha, kappa_rng);
    }
    Basis<D,Real>::eval_all(
      X, D, 1, npts, kappa_rng, n, alpha_table, tail_deg,
      invh_rng, Vrng, npts, nullptr);
    Basis<D,Real>::eval_all(
      X, D, 1, npts, kappa_src, n, alpha_table, tail_deg,
      invh_src, Vsrc, npts, dVsrc);

    for (int j = 0; j < ncol; ++j)
    {
      const Real* dVj = dVsrc +
        static_cast<std::size_t>(j) * npts * D;
      for (int pos = colptr[j]; pos < colptr[j + 1]; ++pos)
      {
        const int i = rowind[pos];
        const Real* vi = Vrng + static_cast<std::size_t>(i) * npts;
        Real value = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          value += vi[p] * wq[p] *
            dVj[static_cast<std::size_t>(p) * D + axis];
        }
        values[pos] = value;
      }
    }

    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_rng);
    std::free(Vrng);
    std::free(Vsrc);
    std::free(dVsrc);
  }

  static std::size_t build_tprod_natural_pruned_csc_from_stencil(
    int n,
    unsigned int q,
    const Real* kappa_src,
    int axis,
    const DMatStencil& S,
    int** colptr_out,
    int** rowind_out,
    Real** values_out)
  {
    if (!values_out)
    {
      std::cerr << "DMat::build_tprod_natural_pruned_csc_from_stencil: null output\n";
      std::exit(1);
    }
    *values_out = nullptr;
    const std::size_t nnz = build_natural_csc_pattern_from_stencil(
      n, S, colptr_out, rowind_out);
    if (nnz > 0)
    {
      *values_out = static_cast<Real*>(std::malloc(nnz * sizeof(Real)));
      if (!*values_out)
      {
        std::free(*colptr_out);
        std::free(*rowind_out);
        *colptr_out = nullptr;
        *rowind_out = nullptr;
        std::cerr << "DMat::build_tprod_natural_pruned_csc_from_stencil: allocation failed\n";
        std::exit(1);
      }
    }
    fill_tprod_natural_csc_values(
      n, q, kappa_src, axis, *colptr_out, *rowind_out, *values_out);
    return nnz;
  }

  static std::size_t build_tprod_natural_pruned_csc(
    int n,
    unsigned int q,
    const Real* kappa_src,
    int axis,
    int** colptr_out,
    int** rowind_out,
    Real** values_out)
  {
    return build_tprod_natural_pruned_csc(
      n, q, kappa_src, axis, colptr_out, rowind_out, values_out,
      default_stencil_folder());
  }

  static std::size_t build_tprod_natural_pruned_csc(
    int n,
    unsigned int q,
    const Real* kappa_src,
    int axis,
    int** colptr_out,
    int** rowind_out,
    Real** values_out,
    const std::string& stencil_folder)
  {
    if (!kappa_src || !colptr_out || !rowind_out || !values_out)
    {
      std::cerr << "DMat::build_tprod_natural_pruned_csc: null input\n";
      std::exit(1);
    }
    DMatStencil S;
    std::memset(&S, 0, sizeof(S));
    load_or_discover_natural_stencil(
      q, kappa_src, axis, &S, stencil_folder);
    const std::size_t nnz = build_tprod_natural_pruned_csc_from_stencil(
      n, q, kappa_src, axis, S, colptr_out, rowind_out, values_out);
    S.clear();
    return nnz;
  }




};

} // namespace jsimplex

#endif

