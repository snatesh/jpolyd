#ifndef JKMAT_H
#define JKMAT_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <cmath>
#include <jbasis.hh>
#include <jquad_tprod.hh>

namespace jsimplex
{



/* Stencil encoding non-zero coupling types for KMat promotion:
     (1) Hom(j)   <- Hom(j)     (same-degree)
     (2) Hom(j)   <- Hom(j+1)   (one-degree-down in rows)
   Each list stores packed multi-index deltas Δ = dst - src.
*/
struct KMatStencil
{
  int j_rep;          // representative ROW degree used for extraction (optional but useful)

  int ndelta0;        // # unique deltas for Hom(j) <- Hom(j)
  uint64_t* keys0;    // length ndelta0, sorted unique

  int ndeltam1;       // # unique deltas for Hom(j) <- Hom(j+1)
  uint64_t* keysm1;   // length ndeltam1, sorted unique

  void clear()
  {
    if (keys0)  { std::free(keys0);  keys0  = nullptr; }
    if (keysm1) { std::free(keysm1); keysm1 = nullptr; }

    j_rep = 0;
    ndelta0 = 0;
    ndeltam1 = 0;
  }
};

  
template<int D, class Real>
struct KMat
{
  /* full dense builder, no pruning */
  static void build_tprod(int n,
                          unsigned int q,          // 1D points per axis
                          const Real* kappa_src,   // length D+1
                          const Real* kappa_tgt,   // length D+1
                          Real* K)                 // row-major MxM
  {
    if (!kappa_src || !kappa_tgt || !K || n < 0 || q == 0)
    {
      std::cerr << "kmat::build_tprod: null input\n";
      std::exit(1);
    }

    const int M = Basis<D,Real>::dim_Pi(n);
    const std::size_t MM = (std::size_t)M * (std::size_t)M;
    std::memset(K, 0, MM * sizeof(Real));

    // Build alpha_table/tail_deg (same for src/tgt). inv_h differs.
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* inv_h_src  = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* inv_h_tgt  = (Real*) std::malloc((std::size_t)M * sizeof(Real));

    if (!alpha_table || !tail_deg || !inv_h_src || !inv_h_tgt)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);

    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + m * D;
      inv_h_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      inv_h_tgt[m] = Basis<D,Real>::inv_h_alpha(a, kappa_tgt);
    }

    // Build κ-aware mapped quadrature for target weight.
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;

    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::free(X);
      std::free(wq);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_tgt, X, wq);
    if (built != npts)
    {
      std::cerr << "KMat::build_tprod: build_kappa failed\n";
      std::exit(1);
    }

    // Evaluate basis at quadrature points.
    // Store V[p + m*npts] (ldV = npts)
    Real* Vsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vtgt = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    if (!Vsrc || !Vtgt)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::free(X);
      std::free(wq);
      std::free(Vsrc);
      std::free(Vtgt);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    Basis<D,Real>::eval_all(
      X,
      D, 1,       // ld_point, ld_dim for AoS X[p*D + j]
      npts,
      kappa_src,
      n,
      alpha_table,
      tail_deg,
      inv_h_src,
      Vsrc,
      npts,
      nullptr
    );

    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa_tgt,
      n,
      alpha_table,
      tail_deg,
      inv_h_tgt,
      Vtgt,
      npts,
      nullptr
    );

    // K = Vtgt^T * diag(wq) * Vsrc
    // K[i,j] = sum_p Vtgt[p+i*npts] * wq[p] * Vsrc[p+j*npts]
    
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
    {
      const Real* vt_i = Vtgt + (std::size_t)i * (std::size_t)npts;
    
      for (int j = 0; j < M; ++j)
      {
        const Real* vs_j = Vsrc + (std::size_t)j * (std::size_t)npts;
    
        Real s = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          s += vt_i[p] * wq[p] * vs_j[p];
        }
    
        K[(std::size_t)i * (std::size_t)M + (std::size_t)j] = s;
      }
    }


    std::free(alpha_table);
    std::free(tail_deg);
    std::free(inv_h_src);
    std::free(inv_h_tgt);
    std::free(X);
    std::free(wq);
    std::free(Vsrc);
    std::free(Vtgt);
  }

  /* full dense builder with pruning, no restriction
     on source and target params */
  static void build_tprod_pruned_dense(int n,
                                       unsigned int q,          // 1D points per axis
                                       const Real* kappa_src,   // length D+1
                                       const Real* kappa_tgt,   // length D+1
                                       Real* K)                 // row-major MxM
  {
    if (!kappa_src || !kappa_tgt || !K || n < 0 || q == 0)
    {
      std::cerr << "kmat::build_tprod: null input\n";
      std::exit(1);
    }

    const int M = Basis<D,Real>::dim_Pi(n);
    const std::size_t MM = (std::size_t)M * (std::size_t)M;
    std::memset(K, 0, MM * sizeof(Real));

    // Build alpha_table/tail_deg (same for src/tgt). inv_h differs.
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* inv_h_src  = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* inv_h_tgt  = (Real*) std::malloc((std::size_t)M * sizeof(Real));

    if (!alpha_table || !tail_deg || !inv_h_src || !inv_h_tgt)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);

    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + m * D;
      inv_h_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      inv_h_tgt[m] = Basis<D,Real>::inv_h_alpha(a, kappa_tgt);
    }

    // Build κ-aware mapped quadrature for target weight.
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;

    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::free(X);
      std::free(wq);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_tgt, X, wq);
    if (built != npts)
    {
      std::cerr << "KMat::build_tprod: build_kappa failed\n";
      std::exit(1);
    }

    // Evaluate basis at quadrature points.
    // Store V[p + m*npts] (ldV = npts)
    Real* Vsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vtgt = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    if (!Vsrc || !Vtgt)
    {
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h_src);
      std::free(inv_h_tgt);
      std::free(X);
      std::free(wq);
      std::free(Vsrc);
      std::free(Vtgt);
      std::cerr << "failed alloc\n"; 
      exit(1);
    }

    Basis<D,Real>::eval_all(
      X,
      D, 1,       // ld_point, ld_dim for AoS X[p*D + j]
      npts,
      kappa_src,
      n,
      alpha_table,
      tail_deg,
      inv_h_src,
      Vsrc,
      npts,
      nullptr
    );

    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa_tgt,
      n,
      alpha_table,
      tail_deg,
      inv_h_tgt,
      Vtgt,
      npts,
      nullptr
    );

    // K = Vtgt^T * diag(wq) * Vsrc
    // K[i,j] = sum_p Vtgt[p+i*npts] * wq[p] * Vsrc[p+j*npts]
    constexpr Real KMAT_PRUNE_TOL =
      Real(1000) * std::numeric_limits<Real>::epsilon();
    
    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
    {
      const Real* vt_i = Vtgt + (std::size_t)i * (std::size_t)npts;
    
      for (int j = 0; j < M; ++j)
      {
        const Real* vs_j = Vsrc + (std::size_t)j * (std::size_t)npts;
    
        Real s = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          s += vt_i[p] * wq[p] * vs_j[p];
        }
    
        // Prune quadrature / roundoff noise
        if (std::abs(s) <= KMAT_PRUNE_TOL)
        {
          s = Real(0);
        }
    
        K[(std::size_t)i * (std::size_t)M + (std::size_t)j] = s;
      }
    }


    std::free(alpha_table);
    std::free(tail_deg);
    std::free(inv_h_src);
    std::free(inv_h_tgt);
    std::free(X);
    std::free(wq);
    std::free(Vsrc);
    std::free(Vtgt);
  }

  /* Dense builder with pruning and natural restriction
     on source and target kappa 
 
     requires that dk = kappa_src-kappa_tgt = ej in R^{d+1} 
     which is the natural sparse promotion by exactly 1 in exactly one of 
     the parameters - doing so essentially lifts the input
     polynomial space by one degree corresponding to multiplication
     by the extra barycentric factor which pops out via the promoted 
     weight function */
  static void build_tprod_natural_pruned_dense(int n,
                                               unsigned int q,          // 1D points per axis
                                               const Real* kappa_src,   // length D+1
                                               const Real* kappa_tgt,   // length D+1
                                               Real* K)                 // row-major MxM
  {
    if (!kappa_src || !kappa_tgt || !K || n < 0 || q == 0)
    {
      std::cerr << "kmat::build_tprod: null input\n";
      std::exit(1);
    }

    const int M = Basis<D,Real>::dim_Pi(n);

    // Default to safe output on failure.
    std::memset(K, 0, (std::size_t)M * (std::size_t)M * sizeof(Real));

    // Enforce: kappa_tgt = kappa_src + e_j for exactly one j in {0..D}
    const Real tol = Real(1e-14);
    
    int idx_plus1 = -1;
    
    for (int i = 0; i < D + 1; ++i)
    {
      const Real dk = kappa_tgt[i] - kappa_src[i];
    
      if (std::abs(dk) <= tol)
      {
        // ok: zero shift
        continue;
      }
    
      if (std::abs(dk - Real(1)) <= tol)
      {
        // found +1 shift
        if (idx_plus1 != -1)
        {
          std::cerr << "kmat::build_tprod_natural: kappa_tgt-kappa_src != ej";
          exit(1);
        }
        idx_plus1 = i;
        continue;
      }
    
      // any other shift (negative, >1, non-integer) is invalid
      std::cerr << "kmat::build_tprod_natural: kappa_tgt-kappa_src != ej";
      exit(1);
    }
    
    // Must have exactly one promoted parameter
    if (idx_plus1 == -1)
    {
      std::cerr << "kmat::build_tprod_natural: kappa_tgt-kappa_src != ej";
      exit(1);
    }

    
    build_tprod_pruned_dense(n, q, kappa_src, kappa_tgt, K);
    
  }
  
  /* Natural promoted sparse ops mapping P_kappa to P_kappa+ej
     using stencil to assemble only nnz in dense K
   
     if n < stencil_min 
     Builds dense K via build_tprod(...), then:

     otherwise
       Compute stable stencil pattern for n between stencil_min,stencil_max 
       Then evaluate kmat entries only where indicated by non-zero stencil */
  static void build_tprod_natural_pruned(int n,
                                         unsigned int q,
                                         const Real* kappa_src,  // length D+1
                                         const Real* kappa_tgt,
                                         Real* K)             // (M*M) row-major
  {
    if (n < 0) { std::cerr << "n must be non-negative\n"; exit(1); }

    const int stencil_min = D+1;
    const int stencil_max = 4*D;

    if (n < stencil_min)
    {
      build_tprod_natural_pruned_dense(n, q, kappa_src, kappa_tgt, K);
      return;
    }
    std::cout << "USING STENCIL\n" << std::endl;
    // 1) Discover degree-invariant delta stencil (small-n exploration)
    KMatStencil S;
    std::memset(&S, 0, sizeof(S));
    discover_stencil_stable(q, kappa_src, kappa_tgt, stencil_min, stencil_max, &S);

    // 2) Build using stencil (still dense output for testing)
    build_tprod_from_deltas(n, q, kappa_src, kappa_tgt, S, K);
    // 4) Cleanup
    S.clear();
  }


  static std::size_t build_tprod_pruned_csc(int n,
                                            unsigned int q,
                                            const Real* kappa_src,
                                            const Real* kappa_tgt,
                                            int** colptr_out,
                                            int** rowind_out,
                                            Real** x_out)
  {
    if (!kappa_src || !kappa_tgt || !colptr_out || !rowind_out || !x_out)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: null input\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *x_out      = nullptr;
  
    if (n < 0)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: require n >= 0\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: require q >= 1\n";
      std::exit(1);
    }
  
    const int M = Basis<D,Real>::dim_Pi(n);
    const int stencil_min = D + 1;
    const int stencil_max = 4 * D;
    if (n < stencil_min)
    {
      Real* Kdense = (Real*) std::malloc((std::size_t)M * (std::size_t)M * sizeof(Real));
      if (!Kdense) { std::cerr << "alloc failed\n"; std::exit(1); }
      build_tprod_natural_pruned_dense(n, q, kappa_src, kappa_tgt, Kdense);
      const std::size_t nnz = detail::compress_dense_to_csc(M, Kdense, colptr_out, rowind_out, x_out);
      free(Kdense);
      return nnz;
    } 
    // ---- Enforce: kappa_tgt = kappa_src + e_j (exactly one +1) ----
    {
      const Real tol = Real(1e-14);
      int idx_plus1 = -1;
      for (int i = 0; i < D + 1; ++i)
      {
        const Real dk = kappa_tgt[i] - kappa_src[i];
        if (std::abs(dk) <= tol) continue;
  
        if (std::abs(dk - Real(1)) <= tol)
        {
          if (idx_plus1 != -1)
          {
            std::cerr << "KMat::build_tprod_pruned_csc: dk != e_j (multiple +1)\n";
            std::exit(1);
          }
          idx_plus1 = i;
          continue;
        }
  
        std::cerr << "KMat::build_tprod_pruned_csc: dk != e_j\n";
        std::exit(1);
      }
      if (idx_plus1 == -1)
      {
        std::cerr << "KMat::build_tprod_pruned_csc: dk != e_j (no +1)\n";
        std::exit(1);
      }
    }
  

    // ---- Discover stable stencil (two key sets) ----
    KMatStencil S;
    std::memset(&S, 0, sizeof(S));
    discover_stencil_stable(q, kappa_src, kappa_tgt, stencil_min, stencil_max, &S);
  
    // ---- Quadrature (target weight) ----
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;
    const int ldV = npts;
  
    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: malloc quad failed\n";
      std::exit(1);
    }
  
    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_tgt, X, wq);
    if (built != npts)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: build_kappa failed\n";
      std::exit(1);
    }
  
    // ---- Basis tables ----
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* invh_src   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* invh_tgt   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    if (!alpha_table || !tail_deg || !invh_src || !invh_tgt)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: malloc tables failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);
  
    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + (std::size_t)m * (std::size_t)D;
      invh_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      invh_tgt[m] = Basis<D,Real>::inv_h_alpha(a, kappa_tgt);
    }
  
    // ---- Evaluate basis values ----
    Real* Vsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vtgt = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    if (!Vsrc || !Vtgt)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: malloc V failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::eval_all(
      X, D, 1,
      npts,
      kappa_src,
      n,
      alpha_table,
      tail_deg,
      invh_src,
      Vsrc,
      ldV,
      nullptr
    );
  
    Basis<D,Real>::eval_all(
      X, D, 1,
      npts,
      kappa_tgt,
      n,
      alpha_table,
      tail_deg,
      invh_tgt,
      Vtgt,
      ldV,
      nullptr
    );
  
    // ---- Pass 1: count nnz per column ----
    int* colnnz = (int*) std::malloc((std::size_t)M * sizeof(int));
    if (!colnnz)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: alloc colnnz failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) colnnz[j] = 0;
  
    int dvec[8];
    int dst_alpha[8];
  
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int c    = Basis<D,Real>::dim_Hom(jdeg);
  
      for (int jloc = 0; jloc < c; ++jloc)
      {
        const int jg = col0 + jloc;
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
  
        int cnt = 0;
  
        // keys0: target degree = jdeg
        for (int t = 0; t < S.ndelta0; ++t)
        {
          unpack_delta8(S.keys0[t], dvec);
  
          bool ok = true;
          int sum = 0;
          for (int d = 0; d < D; ++d)
          {
            dst_alpha[d] = src[d] + dvec[d];
            if (dst_alpha[d] < 0) { ok = false; break; }
            sum += dst_alpha[d];
          }
          if (!ok) continue;
          if (sum != jdeg) continue;
  
          const int iloc = hom_encode_rank_fast(jdeg, dst_alpha);
          const int r    = Basis<D,Real>::dim_Hom(jdeg);
          if (iloc < 0 || iloc >= r) continue;
  
          ++cnt;
        }
  
        // keysm1: target degree = jdeg-1 (if jdeg>=1)
        if (jdeg >= 1)
        {
          const int ideg = jdeg - 1;
          const int r    = Basis<D,Real>::dim_Hom(ideg);
  
          for (int t = 0; t < S.ndeltam1; ++t)
          {
            unpack_delta8(S.keysm1[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              dst_alpha[d] = src[d] + dvec[d];
              if (dst_alpha[d] < 0) { ok = false; break; }
              sum += dst_alpha[d];
            }
            if (!ok) continue;
            if (sum != ideg) continue;
  
            const int iloc = hom_encode_rank_fast(ideg, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
  
            ++cnt;
          }
        }
  
        colnnz[jg] = cnt;
      }
    }
  
    int* colptr = (int*) std::malloc((std::size_t)(M + 1) * sizeof(int));
    if (!colptr)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: alloc colptr failed\n";
      std::exit(1);
    }
  
    colptr[0] = 0;
    for (int j = 0; j < M; ++j)
    {
      colptr[j + 1] = colptr[j] + colnnz[j];
    }
    const int nnz = colptr[M];
  
    int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
    Real* x     = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
    if ((!rowind && nnz) || (!x && nnz))
    {
      std::cerr << "KMat::build_tprod_pruned_csc: alloc nnz arrays failed\n";
      std::exit(1);
    }
  
    int* wpos = (int*) std::malloc((std::size_t)M * sizeof(int));
    if (!wpos)
    {
      std::cerr << "KMat::build_tprod_pruned_csc: alloc wpos failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) wpos[j] = colptr[j];
  
    // ---- Pass 2: fill CSC ----
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int c    = Basis<D,Real>::dim_Hom(jdeg);
  
      for (int jloc = 0; jloc < c; ++jloc)
      {
        const int jg = col0 + jloc;
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
        const Real* vj = Vsrc + (std::size_t)jg * (std::size_t)ldV;
  
        // keys0 -> target degree jdeg
        {
          const int row0 = Basis<D,Real>::dim_Pi(jdeg - 1);
          const int r    = Basis<D,Real>::dim_Hom(jdeg);
  
          for (int t = 0; t < S.ndelta0; ++t)
          {
            unpack_delta8(S.keys0[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              dst_alpha[d] = src[d] + dvec[d];
              if (dst_alpha[d] < 0) { ok = false; break; }
              sum += dst_alpha[d];
            }
            if (!ok) continue;
            if (sum != jdeg) continue;
  
            const int iloc = hom_encode_rank_fast(jdeg, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
  
            const int ig = row0 + iloc;
            const Real* vi = Vtgt + (std::size_t)ig * (std::size_t)ldV;
  
            Real s = Real(0);
            for (int p = 0; p < npts; ++p)
            {
              s += vi[p] * wq[p] * vj[p];
            }
  
            const int pos = wpos[jg]++;
            rowind[pos] = ig;
            x[pos] = s;
          }
        }
  
        // keysm1 -> target degree jdeg-1
        if (jdeg >= 1)
        {
          const int ideg = jdeg - 1;
          const int row0 = Basis<D,Real>::dim_Pi(ideg - 1);
          const int r    = Basis<D,Real>::dim_Hom(ideg);
  
          for (int t = 0; t < S.ndeltam1; ++t)
          {
            unpack_delta8(S.keysm1[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              dst_alpha[d] = src[d] + dvec[d];
              if (dst_alpha[d] < 0) { ok = false; break; }
              sum += dst_alpha[d];
            }
            if (!ok) continue;
            if (sum != ideg) continue;
  
            const int iloc = hom_encode_rank_fast(ideg, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
  
            const int ig = row0 + iloc;
            const Real* vi = Vtgt + (std::size_t)ig * (std::size_t)ldV;
  
            Real s = Real(0);
            for (int p = 0; p < npts; ++p)
            {
              s += vi[p] * wq[p] * vj[p];
            }
  
            const int pos = wpos[jg]++;
            rowind[pos] = ig;
            x[pos] = s;
          }
        }
      }
    }
  
    // ---- output ----
    *colptr_out = colptr;
    *rowind_out = rowind;
    *x_out      = x;
  
    // ---- cleanup temporaries ----
    S.clear();
    std::free(colnnz);
    std::free(wpos);
  
    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_tgt);
    std::free(Vsrc);
    std::free(Vtgt);
    return (std::size_t)nnz;
  }


  /* convert dense to CSC */
  /*static inline std::size_t compress_dense_to_csc(int M,
                                                  const Real* A,     // row-major MxM
                                                  int** colptr_out,
                                                  int** rowind_out,
                                                  Real** x_out)
  {
    int* colnnz = (int*) std::malloc((std::size_t)M * sizeof(int));
    if (!colnnz)
    {
      std::cerr << "compress_dense_to_csc: alloc colnnz failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) colnnz[j] = 0;
  
    // count nnz per column
    for (int j = 0; j < M; ++j)
    {
      int cnt = 0;
      for (int i = 0; i < M; ++i)
      {
        const Real v = A[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (v != Real(0)) ++cnt;
      }
      colnnz[j] = cnt;
    }
  
    int* colptr = (int*) std::malloc((std::size_t)(M + 1) * sizeof(int));
    if (!colptr)
    {
      std::cerr << "compress_dense_to_csc: alloc colptr failed\n";
      std::exit(1);
    }
  
    colptr[0] = 0;
    for (int j = 0; j < M; ++j)
    {
      colptr[j + 1] = colptr[j] + colnnz[j];
    }
  
    const int nnz = colptr[M];
  
    int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
    Real* x     = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
    if ((!rowind && nnz) || (!x && nnz))
    {
      std::cerr << "compress_dense_to_csc: alloc nnz arrays failed\n";
      std::exit(1);
    }
  
    int* wpos = (int*) std::malloc((std::size_t)M * sizeof(int));
    if (!wpos)
    {
      std::cerr << "compress_dense_to_csc: alloc wpos failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) wpos[j] = colptr[j];
  
    // fill (unsorted rows per column, but deterministic)
    for (int j = 0; j < M; ++j)
    {
      for (int i = 0; i < M; ++i)
      {
        const Real v = A[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (v == Real(0)) continue;
  
        const int pos = wpos[j]++;
        rowind[pos] = i;
        x[pos] = v;
      }
    }
  
    std::free(colnnz);
    std::free(wpos);
  
    *colptr_out = colptr;
    *rowind_out = rowind;
    *x_out      = x;
  
    return (std::size_t)nnz;
  }*/


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

  static inline bool deltas_equal(const KMatStencil& A, const KMatStencil& B)
  {
    if (A.ndelta0  != B.ndelta0)  return false;
    if (A.ndeltam1 != B.ndeltam1) return false;
  
    if (A.ndelta0)
    {
      if (std::memcmp(A.keys0,
                      B.keys0,
                      (std::size_t)A.ndelta0 * sizeof(uint64_t)) != 0)
      {
        return false;
      }
    }
  
    if (A.ndeltam1)
    {
      if (std::memcmp(A.keysm1,
                      B.keysm1,
                      (std::size_t)A.ndeltam1 * sizeof(uint64_t)) != 0)
      {
        return false;
      }
    }
  
    return true;
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
                                        const Real* Kdense,      // MxM row-major, already pruned (zeros OK)
                                        const int* alpha_table,  // (M x D)
                                        KMatStencil* S)
  {
    if (!S || !Kdense || !alpha_table)
    {
      std::cerr << "KMat::extract_deltas_from_block: null input\n";
      std::exit(1);
    }
  
    const int M = Basis<D,Real>::dim_Pi(n_test);
  
    // Representative ROW degree j. We need j+1 to exist for the (j+1)->j block.
    // Use something safely interior: j_rep = n_test - 1.
    int j_rep = n_test - 1;
    if (j_rep < 0) j_rep = 0;
  
    const int row0  = Basis<D,Real>::dim_Pi(j_rep - 1);
    const int r     = Basis<D,Real>::dim_Hom(j_rep);
  
    const int col0_0 = Basis<D,Real>::dim_Pi(j_rep - 1); // Hom(j_rep)
    const int c0     = Basis<D,Real>::dim_Hom(j_rep);
  
    const int col0_1 = Basis<D,Real>::dim_Pi(j_rep);     // Hom(j_rep+1)
    const int c1     = Basis<D,Real>::dim_Hom(j_rep + 1);
  
    // temp buffers
    uint64_t* tmp0  = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)c0 * sizeof(uint64_t));
    uint64_t* tmpm1 = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)c1 * sizeof(uint64_t));
    if ((!tmp0 && r*c0) || (!tmpm1 && r*c1))
    {
      std::cerr << "KMat::extract_deltas_from_block: alloc failed\n";
      std::exit(1);
    }
    int nt0 = 0;
    int ntm1 = 0;
  
    int dvec[8];
  
    // Loop rows in Hom(j_rep)
    for (int i_loc = 0; i_loc < r; ++i_loc)
    {
      const int i = row0 + i_loc;
      const int* dst = hom_decode_ptr(j_rep, i_loc, alpha_table); // degree j_rep
  
      // ---- Block A: Hom(j_rep) cols (same degree, s=0) ----
      for (int j_loc = 0; j_loc < c0; ++j_loc)
      {
        const int j = col0_0 + j_loc;
        const Real a = Kdense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* src = hom_decode_ptr(j_rep, j_loc, alpha_table); // degree j_rep
  
        for (int d = 0; d < D; ++d) dvec[d] = dst[d] - src[d];
        tmp0[nt0++] = pack_delta8(dvec);
      }
  
      // ---- Block B: Hom(j_rep+1) cols (degree downshift, s=-1) ----
      for (int j_loc = 0; j_loc < c1; ++j_loc)
      {
        const int j = col0_1 + j_loc;
        const Real a = Kdense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* src = hom_decode_ptr(j_rep + 1, j_loc, alpha_table); // degree j_rep+1
  
        for (int d = 0; d < D; ++d) dvec[d] = dst[d] - src[d];
        tmpm1[ntm1++] = pack_delta8(dvec);
      }
    }
  
    auto sort_unique = [&](uint64_t* tmp, int ntmp, int* nout, uint64_t** out)
    {
      if (ntmp == 0)
      {
        *nout = 0;
        *out  = nullptr;
        return;
      }
  
      std::qsort(tmp, (std::size_t)ntmp, sizeof(uint64_t), cmp_u64);
  
      int nunq = 1;
      for (int k = 1; k < ntmp; ++k)
      {
        if (tmp[k] != tmp[k - 1]) ++nunq;
      }
  
      uint64_t* keys = (uint64_t*) std::malloc((std::size_t)nunq * sizeof(uint64_t));
      if (!keys)
      {
        std::cerr << "KMat::extract_deltas_from_block: alloc keys failed\n";
        std::exit(1);
      }
  
      keys[0] = tmp[0];
      int w = 1;
      for (int k = 1; k < ntmp; ++k)
      {
        if (tmp[k] != tmp[k - 1]) keys[w++] = tmp[k];
      }
  
      *nout = nunq;
      *out  = keys;
    };
  
    // commit into S
    S->clear();
    S->j_rep = j_rep;
  
    sort_unique(tmp0,  nt0,  &S->ndelta0,  &S->keys0);
    sort_unique(tmpm1, ntm1, &S->ndeltam1, &S->keysm1);
  
    std::free(tmp0);
    std::free(tmpm1);
  }

  static void discover_stencil_stable(unsigned int q,
                                      const Real* kappa_src,
                                      const Real* kappa_tgt,
                                      int n_min,
                                      int n_max,
                                      KMatStencil* S_out)
  {
    if (!S_out)
    {
      std::cerr << "discover_stencil_stable: null S_out\n";
      std::exit(1);
    }
  
    KMatStencil S_prev; std::memset(&S_prev, 0, sizeof(S_prev));
    KMatStencil S_cur;  std::memset(&S_cur,  0, sizeof(S_cur));
  
    for (int n_test = n_min; n_test <= n_max; ++n_test)
    {
  
      const int M = Basis<D,Real>::dim_Pi(n_test);
  
      // Dense pruned matrix
      Real* Kdense = (Real*) std::malloc((std::size_t)M * (std::size_t)M * sizeof(Real));
      int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      if (!Kdense || !alpha_table)
      {
        std::cerr << "discover_stencil_stable: alloc failed\n";
        std::exit(1);
      }
  
      // Build pruned dense (your existing routine)
      build_tprod_natural_pruned_dense(n_test, q, kappa_src, kappa_tgt, Kdense); 
  
      // Build alpha_table for this n_test
      Basis<D,Real>::build_alpha_table(n_test, alpha_table);
  
      // Extract delta set from block 
      extract_deltas_from_block(n_test, Kdense, alpha_table, &S_cur);
  
      std::free(Kdense);
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
  
    std::cerr << "KMat: delta-stencil did not stabilize up to n_max\n";
    std::exit(1);
  }

  static void build_tprod_from_deltas(int n,
                                      unsigned int q,
                                      const Real* kappa_src,
                                      const Real* kappa_tgt,
                                      const KMatStencil& S,
                                      Real* K) // MxM row-major, dense for testing
  {
    
    if (!kappa_src || !kappa_tgt || !K)
    {
      std::cerr << "KMat::build_tprod_from_deltas: null input\n";
      std::exit(1);
    }
    if (n < 0)
    {
      std::cerr << "KMat::build_tprod_from_deltas: require n >= 0\n";
      std::exit(1);
    }
    if (q == 0)
    {
      std::cerr << "KMat::build_tprod_from_deltas: require q >= 1\n";
      std::exit(1);
    }
  
    const int M = Basis<D,Real>::dim_Pi(n);
    std::memset(K, 0, (std::size_t)M * (std::size_t)M * sizeof(Real));
  
    // ---- Build κ-aware mapped quadrature for the TARGET weight ----
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(q);
    const int npts = (int)npts_u;
  
    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::cerr << "KMat::build_tprod_from_deltas: malloc quad failed\n";
      std::exit(1);
    }
  
    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_tgt, X, wq);
    if (built != npts)
    {
      std::cerr << "KMat::build_tprod_from_deltas: build_kappa failed\n";
      std::exit(1);
    }
  
    // ---- Basis tables for degree n ----
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* invh_src   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    Real* invh_tgt   = (Real*) std::malloc((std::size_t)M * sizeof(Real));
  
    if (!alpha_table || !tail_deg || !invh_src || !invh_tgt)
    {
      std::cerr << "KMat::build_tprod_from_deltas: malloc tables failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);
  
    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + m * D;
      invh_src[m] = Basis<D,Real>::inv_h_alpha(a, kappa_src);
      invh_tgt[m] = Basis<D,Real>::inv_h_alpha(a, kappa_tgt);
    }
  
    // ---- Evaluate basis values at quadrature points ----
    const int ldV = npts;
  
    Real* Vtgt = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    Real* Vsrc = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    if (!Vtgt || !Vsrc)
    {
      std::cerr << "KMat::build_tprod_from_deltas: malloc V failed\n";
      std::exit(1);
    }
  
    // Target values (orthonormal in kappa_tgt)
    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa_tgt,
      n,
      alpha_table,
      tail_deg,
      invh_tgt,
      Vtgt,
      ldV,
      nullptr
    );
  
    // Source values (orthonormal in kappa_src)
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
      nullptr
    );
  
    // === Assemble using two delta stencils ===
    int dvec[8];
    int dst_alpha[8];
  
    // Loop over SOURCE blocks by degree jdeg
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int c    = Basis<D,Real>::dim_Hom(jdeg);
  
      for (int jloc = 0; jloc < c; ++jloc)
      {
        const int jg = col0 + jloc;            // global column index
        const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
        const Real* vj = Vsrc + (std::size_t)jg * (std::size_t)ldV;
  
        // ---- s = 0 block: rows Hom(jdeg) ----
        {
          const int row0 = Basis<D,Real>::dim_Pi(jdeg - 1);
          const int r    = Basis<D,Real>::dim_Hom(jdeg);
  
          for (int t = 0; t < S.ndelta0; ++t)
          {
            unpack_delta8(S.keys0[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              dst_alpha[d] = src[d] + dvec[d];
              if (dst_alpha[d] < 0) { ok = false; break; }
              sum += dst_alpha[d];
            }
            if (!ok) continue;
            if (sum != jdeg) continue;
  
            const int iloc = hom_encode_rank_fast(jdeg, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
  
            const int ig = row0 + iloc;
            const Real* vi = Vtgt + (std::size_t)ig * (std::size_t)ldV;
  
            Real s = Real(0);
            for (int p = 0; p < npts; ++p)
            {
              s += vi[p] * wq[p] * vj[p];
            }
  
            K[(std::size_t)ig * (std::size_t)M + (std::size_t)jg] = s;
          }
        }
  
        // ---- s = -1 block: rows Hom(jdeg-1) get cols Hom(jdeg) ----
        // This corresponds to your empirical: row block i couples to col blocks i and i+1.
        // Here, source degree is jdeg; target degree is jdeg-1.
        if (jdeg >= 1)
        {
          const int ideg = jdeg - 1;
          const int row0 = Basis<D,Real>::dim_Pi(ideg - 1);
          const int r    = Basis<D,Real>::dim_Hom(ideg);
  
          for (int t = 0; t < S.ndeltam1; ++t)
          {
            unpack_delta8(S.keysm1[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              dst_alpha[d] = src[d] + dvec[d];
              if (dst_alpha[d] < 0) { ok = false; break; }
              sum += dst_alpha[d];
            }
            if (!ok) continue;
            if (sum != ideg) continue;
  
            const int iloc = hom_encode_rank_fast(ideg, dst_alpha);
            if (iloc < 0 || iloc >= r) continue;
  
            const int ig = row0 + iloc;
            const Real* vi = Vtgt + (std::size_t)ig * (std::size_t)ldV;
  
            Real s = Real(0);
            for (int p = 0; p < npts; ++p)
            {
              s += vi[p] * wq[p] * vj[p];
            }
  
            K[(std::size_t)ig * (std::size_t)M + (std::size_t)jg] = s;
          }
        }
      }
    }
  
    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(invh_src);
    std::free(invh_tgt);
    std::free(Vtgt);
    std::free(Vsrc);
  }



};

} // namespace jsimplex

#endif // JKMAT_H

