#ifndef JPOLYD_DMAT_HH
#define JPOLYD_DMAT_HH

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <array>
#include "jbasis.hh"
#include "jquad_tprod.hh"   // QuadMapped<D,Real>

namespace jsimplex
{

/* stencil encoding non-zero coupling between P_j,P_{j-1} 
   in sparse differrentiation operators */
struct DMatStencil
{
  int Ddim;        // = D (redundant but nice for debugging)
  int axis;        // derivative axis
  int ndelta;      // number of unique delta keys
  uint64_t* keys;  // length ndelta, sorted unique packed deltas

  void clear()
  {
    if (keys) { std::free(keys); keys = nullptr; }
    Ddim = 0;
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
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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
                                         const Real* kappa_src,  // length D+1
                                         int axis,               // 0..D-1
                                         Real* Dout)             // (M*M) row-major
  {
    if (n < 0) { std::cerr << "n must be non-negative\n"; exit(1); }
    if (n == 0)
    {
      const int M = 1;//Basis<D,Real>::dim_pi(n);
      std::memset(Dout, 0, (std::size_t)M * (std::size_t)M * sizeof(Real));
      return;
    }

    const int stencil_min = D+1;
    const int stencil_max = 4*D;

    if (n < stencil_min)
    {
      build_tprod_natural_pruned_dense(n, q, kappa_src, axis, Dout);
      return;
    }

    // 1) Discover degree-invariant delta stencil (small-n exploration)
    DMatStencil S;
    std::memset(&S, 0, sizeof(S));
    discover_stencil_stable(q, kappa_src, axis, stencil_min, stencil_max, &S);

    // 2) Build using stencil (still dense output for testing)
    build_tprod_from_deltas(n, q, kappa_src, axis, S, Dout);
    // 4) Cleanup
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
  
  static inline bool stencil_equal(const DMatStencil& A, const DMatStencil& B)
  {
    if (A.axis != B.axis) return false;
    if (A.Ddim != B.Ddim) return false;
    return deltas_equal(A, B);
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
    S->Ddim = D;
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
        std::cout << "n_test = " << n_test << std::endl;
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

  static std::size_t build_tprod_natural_pruned_csc(int n,
                                                    unsigned int q,
                                                    const Real* kappa_src,
                                                    int axis,
                                                    int** colptr_out,
                                                    int** rowind_out,
                                                    Real** x_out)
  {
    const int stencil_min = 5;
    const int stencil_max = 12;
  
    if (!colptr_out || !rowind_out || !x_out)
    {
      std::cerr << "build_tprod_natural_pruned_csc: null output ptr\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *x_out = nullptr;
  
    if (n < 0)
    {
      std::cerr << "build_tprod_natural_pruned_csc: n < 0\n";
      std::exit(1);
    }
    if (n == 0)
    {
      // Operator Π0 -> Π-1 (empty range). Return a 0x1 matrix: nrow=0, ncol=1.
      const int ncol = Basis<D,Real>::dim_Pi(0); // 1
      int* colptr = (int*) std::malloc((std::size_t)(ncol + 1) * sizeof(int));
      if (!colptr) { std::cerr << "alloc failed\n"; std::exit(1); }
      colptr[0] = 0;
      colptr[1] = 0;
      *colptr_out = colptr;
      *rowind_out = nullptr;
      *x_out = nullptr;
      return 0;
    }
  
    const int ncol = Basis<D,Real>::dim_Pi(n);
    const int nrow = Basis<D,Real>::dim_Pi(n - 1);
  
    // ---- Discover delta stencil (pattern) ----
    DMatStencil S;
    std::memset(&S, 0, sizeof(S));
  
    if (n < stencil_min)
    {
      Real* Ddense = (Real*) std::malloc((std::size_t)ncol * (std::size_t)ncol * sizeof(Real));
      if (!Ddense) { std::cerr << "alloc failed\n"; std::exit(1); }
      build_tprod_natural_pruned_dense(n, q, kappa_src, axis, Ddense);
  
      // Compress dense -> CSC over active rows [0..nrow).
      // Count nnz per column first
      int* colnnz = (int*) std::malloc((std::size_t)ncol * sizeof(int));
      if (!colnnz) { std::cerr << "alloc failed\n"; std::exit(1); }
      for (int j = 0; j < ncol; ++j) colnnz[j] = 0;
  
      for (int j = 0; j < ncol; ++j)
      {
        int cnt = 0;
        for (int i = 0; i < nrow; ++i)
        {
          if (Ddense[(std::size_t)i * (std::size_t)ncol + (std::size_t)j] != Real(0)) ++cnt;
        }
        colnnz[j] = cnt;
      }
  
      int* colptr = (int*) std::malloc((std::size_t)(ncol + 1) * sizeof(int));
      if (!colptr) { std::cerr << "alloc failed\n"; std::exit(1); }
      colptr[0] = 0;
      for (int j = 0; j < ncol; ++j) colptr[j + 1] = colptr[j] + colnnz[j];
      const int nnz = colptr[ncol];
  
      int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
      Real* x = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
      if ((!rowind && nnz) || (!x && nnz)) { std::cerr << "alloc failed\n"; std::exit(1); }
  
      int* wpos = (int*) std::malloc((std::size_t)ncol * sizeof(int));
      if (!wpos) { std::cerr << "alloc failed\n"; std::exit(1); }
      for (int j = 0; j < ncol; ++j) wpos[j] = colptr[j];
  
      for (int j = 0; j < ncol; ++j)
      {
        for (int i = 0; i < nrow; ++i)
        {
          const Real a = Ddense[(std::size_t)i * (std::size_t)ncol + (std::size_t)j];
          if (a == Real(0)) continue;
          const int p = wpos[j]++;
          rowind[p] = i;
          x[p] = a;
        }
      }
  
      std::free(wpos);
      std::free(colnnz);
      std::free(Ddense);
  
      *colptr_out = colptr;
      *rowind_out = rowind;
      *x_out = x;
      return (std::size_t)nnz;
    }
  
    discover_stencil_stable(q, kappa_src, axis, stencil_min, stencil_max, &S);
  
    if (!kappa_src)
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
  

    Real* dVsrc_axis = (Real*) std::malloc((std::size_t)ncol * (std::size_t)npts * sizeof(Real));
    if (!dVsrc_axis) { std::cerr << "alloc failed\n"; std::exit(1); }
    
    for (int jg = 0; jg < ncol; ++jg)
    {
      for (int p = 0; p < npts; ++p)
      {
        dVsrc_axis[(std::size_t)jg * (std::size_t)npts + (std::size_t)p] =
          dVsrc[((std::size_t)jg * (std::size_t)npts + (std::size_t)p) * (std::size_t)D + (std::size_t)axis];
      }
    }
  
    // ---- Pass 1: count nnz per column ----
    int* colnnz = (int*) std::malloc((std::size_t)ncol * sizeof(int));
    if (!colnnz) { std::cerr << "alloc failed\n"; std::exit(1); }
    for (int j = 0; j < ncol; ++j) colnnz[j] = 0;
  
    int dvec[8];
    int dst_alpha[8];
  
    for (int jdeg = 1; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int cdeg = Basis<D,Real>::dim_Hom(jdeg);
  
      for (int jloc = 0; jloc < cdeg; ++jloc)
      {
        const int jg = col0 + jloc; // global column in Π_n
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
  
        int cnt = 0;
        for (int t = 0; t < S.ndelta; ++t)
        {
          unpack_delta8(S.keys[t], dvec);
  
          bool ok = true;
          int sum = 0;
          for (int d = 0; d < D; ++d)
          {
            dst_alpha[d] = src[d] + dvec[d];
            if (dst_alpha[d] < 0) { ok = false; break; }
            sum += dst_alpha[d];
          }
          if (!ok) continue;
          if (sum != jdeg - 1) continue;
  
          // local row in Hom(jdeg-1)
          const int iloc = hom_encode_rank_fast(jdeg - 1, dst_alpha);
          // global row in Π_{n-1}
          const int ig = Basis<D,Real>::dim_Pi(jdeg - 2) + iloc;
  
          if ((unsigned)ig < (unsigned)nrow) { ++cnt; }
        }
  
        colnnz[jg] = cnt;
      }
    }
  
    // prefix sum -> colptr
    int* colptr = (int*) std::malloc((std::size_t)(ncol + 1) * sizeof(int));
    if (!colptr) { std::cerr << "alloc failed\n"; std::exit(1); }
    colptr[0] = 0;
    for (int j = 0; j < ncol; ++j) colptr[j + 1] = colptr[j] + colnnz[j];
    const int nnz = colptr[ncol];
  
    int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
    Real* x = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
    if ((!rowind && nnz) || (!x && nnz)) { std::cerr << "alloc failed\n"; std::exit(1); }
  
    // working write positions
    int* wpos = (int*) std::malloc((std::size_t)ncol * sizeof(int));
    if (!wpos) { std::cerr << "alloc failed\n"; std::exit(1); }
    for (int j = 0; j < ncol; ++j) wpos[j] = colptr[j];
  
    // ---- Pass 2: fill rowind and x ----
    for (int jdeg = 1; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int cdeg = Basis<D,Real>::dim_Hom(jdeg);
      const int row0 = Basis<D,Real>::dim_Pi(jdeg - 2);
  
      for (int jloc = 0; jloc < cdeg; ++jloc)
      {
        const int jg = col0 + jloc;
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
  
        // pointer to derivative samples for this column jg
        const Real* dVj = dVsrc_axis + (std::size_t)jg * (std::size_t)npts;
  
        for (int t = 0; t < S.ndelta; ++t)
        {
          unpack_delta8(S.keys[t], dvec);
  
          bool ok = true;
          int sum = 0;
          for (int d = 0; d < D; ++d)
          {
            dst_alpha[d] = src[d] + dvec[d];
            if (dst_alpha[d] < 0) { ok = false; break; }
            sum += dst_alpha[d];
          }
          if (!ok) continue;
          if (sum != jdeg - 1) continue;
  
          const int iloc = hom_encode_rank_fast(jdeg - 1, dst_alpha);
          const int ig = row0 + iloc;
          if ((unsigned)ig >= (unsigned)nrow) continue;
  
          // Compute entry: ∫ phi_i * (∂axis phi_j) w
          const Real* vi = Vrng + (std::size_t)ig * (std::size_t)npts;
  
          Real s = Real(0);
          for (int p = 0; p < npts; ++p)
          {
            s += vi[p] * wq[p] * dVj[p];
          }
  
          // (Optional) prune here? Usually you prune after building the full operator.
          // For now, store raw; you can apply row-relative prune later if desired.
          const int pos = wpos[jg]++;
          rowind[pos] = ig;
          x[pos] = s;
        }
      }
    }
  
    // sanity: wpos[j] should end at colptr[j+1]
    #ifdef JPOLY_DEBUG
    for (int j = 0; j < ncol; ++j)
    {
      if (wpos[j] != colptr[j + 1])
      {
        std::cerr << "CSC fill mismatch at col " << j << "\n";
        std::exit(1);
      }
    }
    #endif
  
  
    std::free(wpos);
    std::free(colnnz);
    std::free(alpha_table);
    S.clear();
    std::free(X);
    std::free(wq);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_rng);
    std::free(Vrng);
    std::free(Vsrc);
    std::free(dVsrc);
    std::free(dVsrc_axis);  
    *colptr_out = colptr;
    *rowind_out = rowind;
    *x_out = x;
    return (std::size_t)nnz;
  }




};

} // namespace jsimplex

#endif

