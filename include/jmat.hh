#ifndef JMAT_H
#define JMAT_H

#include <cstdlib>      
#include <cmath>        
#include <cstddef>      
#include <cstring>
#include <type_traits>  
#include <jdetail.hh>     
#include <jbasis.hh>       
#include <jquad_tprod.hh>   


namespace jsimplex
{

/* Stencil encoding non-zero coupling types for JMat:
     (1) Hom(j)   <- Hom(j)     (same-degree)
     (2) Hom(j)   <- Hom(j+1)   (one-degree-down in rows)
     (3) Hom(j)   <- Hom(j-1)   (one-degree-up in rows)
   Each list stores packed multi-index deltas Δ = dst - src.
*/
struct JMatStencil
{
  int j_rep;          // representative ROW degree used for extraction (optional but useful)

  int ndelta0;        // # unique deltas for Hom(j) <- Hom(j)
  uint64_t* keys0;    // length ndelta0, sorted unique

  int ndeltam1;       // # unique deltas for Hom(j) <- Hom(j+1)
  uint64_t* keysm1;   // length ndeltam1, sorted unique
  
  int ndeltap1;       // # unique deltas for Hom(j) <- Hom(j-1)
  uint64_t* keysp1;   // length ndeltam1, sorted unique

  void clear()
  {
    if (keys0)  { std::free(keys0);  keys0  = nullptr; }
    if (keysm1) { std::free(keysm1); keysm1 = nullptr; }
    if (keysp1) { std::free(keysp1); keysp1 = nullptr; }

    j_rep = 0;
    ndelta0 = 0;
    ndeltam1 = 0;
    ndeltap1 = 0;
  }
};

/* JMat<D,Real>
   Unified builder for Jacobi matrices representing multiplication by coordinates
   on the D-simplex under Jacobi weight with parameters kappa[0..D].

   D = 1:
     build(n, kappa, nquad, J_all)
       - kappa has length 2 (kappa1,kappa2)
       - returns a single n x n dense symmetric matrix (row-major) for x on [0,1]
         with weight x^(kappa1-1/2) (1-x)^(kappa2-1/2), orthonormal basis.
       - nquad is ignored.

   D >= 2:
     build(n, kappa, nquad, J_all)
       - kappa has length D+1
       - builds D block-tridiagonal Jacobi matrices back-to-back in J_all,
         each of size N x N, row-major, where N = dim_Pi(n).
         Matrix i (0-based) corresponds to multiplication by x_i.
*/

template<int D, class Real>
struct JMat
{
  static void build(int n, const Real* kappa, unsigned int nquad, Real* J_all)
  {
    if (!kappa || !J_all || n < 0) { return; }

    if constexpr (D == 1)
    {
      // interpret n as maximum degree => N = n + 1 basis functions
      int N = n + 1;

      Real a = kappa[1] - Real(0.5);
      Real b = kappa[0] - Real(0.5);

      Real* d = (Real*) std::malloc((std::size_t)N * sizeof(Real));
      Real* e = (Real*) std::malloc((std::size_t)(N > 1 ? (N - 1) : 1) * sizeof(Real));
      if (!d || (!e && N > 1))
      {
        std::free(d);
        std::free(e);
        return;
      }

      // Jacobi on [-1,1] for degrees 0..n
      detail::jacobi_tridiag_ON<Real>(N, a, b, d, e);

      // Map to [0,1]: t = (u+1)/2
      for (int i = 0; i < N; ++i)
      {
        d[i] = (d[i] + Real(1.0)) * Real(0.5);
      }
      if (N > 1)
      {
        for (int i = 0; i < N - 1; ++i)
        {
          e[i] *= Real(0.5);
        }
      }

      const std::size_t NN = (std::size_t)N * (std::size_t)N;
      std::memset(J_all, 0, NN * sizeof(Real));
      for (int i = 0; i < N; ++i)
      {
        J_all[i + N * i] = d[i];
        if (i + 1 < N)
        {
          Real off = e[i];
          J_all[i + N * (i + 1)] = off;
          J_all[(i + 1) + N * i] = off;
        }
      }

      std::free(d);
      std::free(e);
    }
    else
    {
      // D>=2: assemble D block-tridiagonal matrices (size N x N each)
      const int N = Basis<D,Real>::dim_Pi(n);

      const std::size_t bytes = (std::size_t)D * (std::size_t)N * (std::size_t)N * sizeof(Real);
      std::memset(J_all, 0, bytes);

      // degree sizes and offsets
      int* r   = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
      int* off = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
      if (!r || !off)
      {
        std::free(r); std::free(off);
        return;
      }

      int acc = 0;
      for (int k = 0; k <= n; ++k)
      {
        r[k] = Basis<D,Real>::dim_R(k);
        off[k] = acc;
        acc += r[k];
      }

      // basis structures
      const int M = N;
      int* alphaTable = (int*)  std::malloc((std::size_t)M * D * sizeof(int));
      int* tailDeg    = (int*)  std::malloc((std::size_t)M * D * sizeof(int));
      Real* invH      = (Real*) std::malloc((std::size_t)M * sizeof(Real));
      if (!alphaTable || !tailDeg || !invH)
      {
        std::free(r); std::free(off);
        std::free(alphaTable); std::free(tailDeg); std::free(invH);
        return;
      }

      Basis<D,Real>::build_alpha_table(n, alphaTable);
      Basis<D,Real>::build_tail_deg(n, alphaTable, tailDeg);
      for (int m = 0; m < M; ++m)
      {
        invH[m] = Basis<D,Real>::inv_h_alpha(alphaTable + m * D, kappa);
      }

      // kappa-aware mapped rule: Q = nquad^D
      if (nquad == 0) { nquad = 1; }
      unsigned long long Qll = 1ULL;
      for (int i = 0; i < D; ++i) { Qll *= (unsigned long long)nquad; }
      const int Q = (int)Qll;

      Real* points  = (Real*) std::malloc((std::size_t)Q * D * sizeof(Real));
      Real* weights = (Real*) std::malloc((std::size_t)Q * sizeof(Real));
      if (!points || !weights)
      {
        std::free(r); std::free(off);
        std::free(alphaTable); std::free(tailDeg); std::free(invH);
        std::free(points); std::free(weights);
        return;
      }

      QuadMapped<D,Real>::build_kappa(nquad, kappa, points, weights);

      // basis values V: Q x M with ld_V = Q (column-major in m)
      Real* V = (Real*) std::malloc((std::size_t)Q * M * sizeof(Real));
      if (!V)
      {
        std::free(r); std::free(off);
        std::free(alphaTable); std::free(tailDeg); std::free(invH);
        std::free(points); std::free(weights);
        return;
      }

      Basis<D,Real>::eval_all(points, /*ld_point*/ D, /*ld_dim*/ 1, Q,
                              kappa, n, alphaTable, tailDeg, invH,
                              V, /*ld_V*/ Q);

      // assemble each coordinate matrix
      for (int coord = 0; coord < D; ++coord)
      {
        Real* Ji = J_all + (std::size_t)coord * (std::size_t)N * (std::size_t)N;

        Real* u = (Real*) std::malloc((std::size_t)Q * sizeof(Real));
        if (!u) { continue; }

        for (int p = 0; p < Q; ++p)
        {
          const Real xi = points[p * D + coord];
          u[p] = weights[p] * xi;
        }

        // stream over points
        for (int p = 0; p < Q; ++p)
        {
          const Real up = u[p];

          for (int k = 0; k <= n; ++k)
          {
            const int rk  = r[k];
            const int ok  = off[k];

            // B_{k,coord}
            for (int a = 0; a < rk; ++a)
            {
              const Real va = V[p + (ok + a) * Q];
              const Real sca = up * va;
              Real* rowJi = Ji + (ok + a) * N;

              for (int b = 0; b < rk; ++b)
              {
                const Real vb = V[p + (ok + b) * Q];
                rowJi[ ok + b ] += sca * vb;
              }
            }

            // A_{k,coord} with k+1 and symmetric counterpart
            if (k < n)
            {
              const int rkp1 = r[k + 1];
              const int okp1 = off[k + 1];

              for (int a = 0; a < rk; ++a)
              {
                const Real va = V[p + (ok + a) * Q];
                const Real sca = up * va;

                for (int b = 0; b < rkp1; ++b)
                {
                  const Real vb = V[p + (okp1 + b) * Q];
                  const Real contrib = sca * vb;

                  Ji[(ok + a)  * N + (okp1 + b)] += contrib; // upper
                  Ji[(okp1 + b)* N + (ok + a)]   += contrib; // lower
                }
              }
            }
          }
        }

        std::free(u);
      }

      // free temporaries
      std::free(V);
      std::free(points);
      std::free(weights);
      std::free(alphaTable);
      std::free(tailDeg);
      std::free(invH);
      std::free(r);
      std::free(off);
    }
  }
  
  static void build_pruned_dense(int n, const Real* kappa, unsigned int nquad, Real* J_all)
  {
    const int N = Basis<D,Real>::dim_Pi(n);
    // call dense builder and prune
    build(n, kappa, nquad, J_all);
    constexpr Real JMAT_PRUNE_TOL =
      Real(1000) * std::numeric_limits<Real>::epsilon();
    // degree sizes and offsets
    int* r   = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    int* off = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    if (!r || !off)
    {
      std::free(r); std::free(off);
      return;
    }

    int acc = 0;
    for (int k = 0; k <= n; ++k)
    {
      r[k] = Basis<D,Real>::dim_R(k);
      off[k] = acc;
      acc += r[k];
    }
    // assemble each coordinate matrix
    for (int coord = 0; coord < D; ++coord)
    {
      Real* Ji = J_all + (std::size_t)coord * (std::size_t)N * (std::size_t)N;
      for (int k = 0; k <= n; ++k)
      {
        const int rk  = r[k];
        const int ok  = off[k];
    
        // B_{k,coord}
        for (int a = 0; a < rk; ++a)
        {
          Real* rowJi = Ji + (ok + a) * N;
          for (int b = 0; b < rk; ++b)
          {
            if (std::abs(rowJi[ ok + b ]) <= JMAT_PRUNE_TOL)
            {
              rowJi[ ok + b ] = Real(0);
            }
          }
        }
        // A_{k,coord} with k+1 and symmetric counterpart
        if (k < n)
        {
          const int rkp1 = r[k + 1];
          const int okp1 = off[k + 1];
    
          for (int a = 0; a < rk; ++a)
          {
            for (int b = 0; b < rkp1; ++b)
            {
              
              if (std::abs(Ji[(ok + a)  * N + (okp1 + b)]) <= JMAT_PRUNE_TOL)
              {
                Ji[(ok + a)  * N + (okp1 + b)] = Real(0);  // upper
              }
              if (std::abs(Ji[(okp1 + b)* N + (ok + a)]) <= JMAT_PRUNE_TOL)
              {
                Ji[(okp1 + b)* N + (ok + a)] = Real(0); // lower
              }
            }
          }
        }
      }
    }
    std::free(r);
    std::free(off); 
  }
  
  /* Multiplication-by-coordinate sparse ops.
     Build J_all as concatenation [J_0 J_1 ... J_{D-1}] where each J_i is MxM.
  
     If n < stencil_min (or D==1): build dense then return.
     Else: discover a stable stencil per coordinate and assemble only those entries
           into dense Ji (for testing).
  */
  static void build_pruned(int n,
                           const Real* kappa,      // length D+1
                           unsigned int nquad,
                           Real* J_all)            // (D * M * M) row-major blocks
  {
    if (!kappa || !J_all)
    {
      std::cerr << "JMat::build_pruned: null input\n";
      std::exit(1);
    }
    if (n < 0)
    {
      std::cerr << "JMat::build_pruned: require n >= 0\n";
      std::exit(1);
    }
    if (nquad == 0) {nquad = 1;}

  
    const int stencil_min = D + 1;
    const int stencil_max = 4 * D;
  
    const int M = Basis<D,Real>::dim_Pi(n);
  
    if (D == 1 || n < stencil_min)
    {
      build_pruned_dense(n, kappa, nquad, J_all); // existing dense builder [J0..]
      return;
    }
  
    for (int coord = 0; coord < D; ++coord)
    {
      Real* Ji = J_all + (std::size_t)coord * (std::size_t)M * (std::size_t)M;
  
      JMatStencil S;
      std::memset(&S, 0, sizeof(S));
  
      // 1) Discover stencil for this coordinate
      discover_stencil_stable(nquad,
                              kappa,
                              coord,
                              stencil_min,
                              stencil_max,
                              &S);
  
      // 2) Assemble using stencil into dense (for testing)
      build_from_deltas(n,
                        kappa,
                        nquad,
                        coord,
                        S,
                        Ji);
  
      S.clear();
    }
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

  static inline bool deltas_equal(const JMatStencil& A, const JMatStencil& B)
  {
    if (A.ndelta0  != B.ndelta0)  return false;
    if (A.ndeltam1 != B.ndeltam1) return false;
    if (A.ndeltap1 != B.ndeltap1) return false;
  
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
    
    if (A.ndeltap1)
    {
      if (std::memcmp(A.keysp1,
                      B.keysp1,
                      (std::size_t)A.ndeltap1 * sizeof(uint64_t)) != 0)
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


  static void discover_stencil_stable(unsigned int nquad,
                                      const Real* kappa,
                                      int coord,
                                      int n_min,
                                      int n_max,
                                      JMatStencil* S_out)
  {
    if (!S_out)
    {
      std::cerr << "discover_stencil_stable: null S_out\n";
      std::exit(1);
    }
  
    JMatStencil S_prev; std::memset(&S_prev, 0, sizeof(S_prev));
    JMatStencil S_cur;  std::memset(&S_cur,  0, sizeof(S_cur));
  
    for (int n_test = n_min; n_test <= n_max; ++n_test)
    {
  
      const int M = Basis<D,Real>::dim_Pi(n_test);
  
      // Dense pruned matrix
      Real* Jdense = (Real*) std::malloc((std::size_t)M * (std::size_t)M * (std::size_t)D * sizeof(Real));
      int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      if (!Jdense || !alpha_table)
      {
        std::cerr << "discover_stencil_stable: alloc failed\n";
        std::exit(1);
      }
  
      // Build pruned dense (your existing routine)
      build_pruned_dense(n_test, kappa, nquad, Jdense); 
  
      // Build alpha_table for this n_test
      Basis<D,Real>::build_alpha_table(n_test, alpha_table);
 
      // select coord
      Real* Jdense_i = Jdense + (std::size_t)coord * (std::size_t)M* (std::size_t)M; 
 
      // Extract delta set from block 
      extract_deltas_from_block(n_test, Jdense_i, alpha_table, &S_cur);
  
      std::free(Jdense);
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
  
    std::cerr << "JMat: delta-stencil did not stabilize up to n_max\n";
    std::exit(1);
  }

  static void extract_deltas_from_block(int n_test,
                                        const Real* Jdense,      // MxM row-major, already pruned (zeros OK)
                                        const int* alpha_table,  // (M x D)
                                        JMatStencil* S)
  {
    if (!S || !Jdense || !alpha_table)
    {
      std::cerr << "JMat::extract_deltas_from_block: null input\n";
      std::exit(1);
    }
  
    const int M = Basis<D,Real>::dim_Pi(n_test);
  
    // Representative ROW degree j. We need j+1 to exist for the (j+1)->j block.
    // Use something safely interior: j_rep = n_test - 1.
    int j_rep = n_test - 1;
    if (j_rep < 0) j_rep = 0;
  
    const int row0  = Basis<D,Real>::dim_Pi(j_rep - 1);
    const int r     = Basis<D,Real>::dim_Hom(j_rep);
 
    const int col0_m1 = Basis<D,Real>::dim_Pi(j_rep - 2); // Hom(j_rep-1)
    const int cm1     = Basis<D,Real>::dim_Hom(j_rep - 1);
 
    const int col0_0 = Basis<D,Real>::dim_Pi(j_rep - 1); // Hom(j_rep)
    const int c0     = Basis<D,Real>::dim_Hom(j_rep);
  
    const int col0_1 = Basis<D,Real>::dim_Pi(j_rep);     // Hom(j_rep+1)
    const int c1     = Basis<D,Real>::dim_Hom(j_rep + 1);
  
    // temp buffers
    uint64_t* tmp0  = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)c0 * sizeof(uint64_t));
    uint64_t* tmpm1 = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)c1 * sizeof(uint64_t));
    uint64_t* tmpp1 = (uint64_t*) std::malloc((std::size_t)r * (std::size_t)cm1 * sizeof(uint64_t));
    if ((!tmp0 && r*c0) || (!tmpm1 && r*c1) || (!tmpp1 && r*cm1))
    {
      std::cerr << "JMat::extract_deltas_from_block: alloc failed\n";
      std::exit(1);
    }
    int nt0 = 0;
    int ntm1 = 0;
    int ntp1 = 0;
  
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
        const Real a = Jdense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* src = hom_decode_ptr(j_rep, j_loc, alpha_table); // degree j_rep
  
        for (int d = 0; d < D; ++d) dvec[d] = dst[d] - src[d];
        tmp0[nt0++] = pack_delta8(dvec);
      }
  
      // ---- Block B: Hom(j_rep+1) cols (degree downshift, s=-1) ----
      for (int j_loc = 0; j_loc < c1; ++j_loc)
      {
        const int j = col0_1 + j_loc;
        const Real a = Jdense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* src = hom_decode_ptr(j_rep + 1, j_loc, alpha_table); // degree j_rep+1
  
        for (int d = 0; d < D; ++d) dvec[d] = dst[d] - src[d];
        tmpm1[ntm1++] = pack_delta8(dvec);
      }
      // ---- Block C: Hom(j_rep-1) cols (degree upshift, s=+1) ----
      for (int j_loc = 0; j_loc < cm1; ++j_loc)
      {
        const int j = col0_m1 + j_loc;
        const Real a = Jdense[(std::size_t)i * (std::size_t)M + (std::size_t)j];
        if (a == Real(0)) continue;
  
        const int* src = hom_decode_ptr(j_rep - 1, j_loc, alpha_table); // degree j_rep-1
  
        for (int d = 0; d < D; ++d) dvec[d] = dst[d] - src[d];
        tmpp1[ntp1++] = pack_delta8(dvec);
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
        std::cerr << "JMat::extract_deltas_from_block: alloc keys failed\n";
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
    sort_unique(tmpp1, ntp1, &S->ndeltap1, &S->keysp1);
  
    std::free(tmp0);
    std::free(tmpm1);
    std::free(tmpp1);
  }

    
  static void build_from_deltas(int n,
                                const Real* kappa,
                                unsigned int nquad,
                                int coord,                  // 0..D-1
                                const JMatStencil& S,
                                Real* Ji)                   // NxN row-major, dense for testing
  {
    if (!kappa || !Ji)
    {
      std::cerr << "JMat::build_from_deltas: null input\n";
      std::exit(1);
    }
    if (n < 0)
    {
      std::cerr << "JMat::build_from_deltas: require n >= 0\n";
      std::exit(1);
    }
    if (nquad == 0) { nquad = 1; }
    if (coord < 0 || coord >= D)
    {
      std::cerr << "JMat::build_from_deltas: coord out of range\n";
      std::exit(1);
    }
  
    const int N = Basis<D,Real>::dim_Pi(n);
    std::memset(Ji, 0, (std::size_t)N * (std::size_t)N * sizeof(Real));
  
    // ---- degree sizes and offsets (same as dense builder) ----
    int* r   = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    int* off = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    if (!r || !off)
    {
      std::cerr << "JMat::build_from_deltas: alloc r/off failed\n";
      std::exit(1);
    }
  
    int acc = 0;
    for (int k = 0; k <= n; ++k)
    {
      r[k] = Basis<D,Real>::dim_R(k);
      off[k] = acc;
      acc += r[k];
    }
  
    // ---- basis structures ----
    const int M = N;
  
    int* alphaTable = (int*)  std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tailDeg    = (int*)  std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* invH      = (Real*) std::malloc((std::size_t)M * sizeof(Real));
  
    if (!alphaTable || !tailDeg || !invH)
    {
      std::cerr << "JMat::build_from_deltas: alloc tables failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::build_alpha_table(n, alphaTable);
    Basis<D,Real>::build_tail_deg(n, alphaTable, tailDeg);
    for (int m = 0; m < M; ++m)
    {
      invH[m] = Basis<D,Real>::inv_h_alpha(alphaTable + (std::size_t)m * (std::size_t)D, kappa);
    }
  
    // ---- quadrature ----
    unsigned long long Qll = 1ULL;
    for (int i = 0; i < D; ++i) Qll *= (unsigned long long)nquad;
    const int Q = (int)Qll;
  
    Real* points  = (Real*) std::malloc((std::size_t)Q * (std::size_t)D * sizeof(Real));
    Real* weights = (Real*) std::malloc((std::size_t)Q * sizeof(Real));
    if (!points || !weights)
    {
      std::cerr << "JMat::build_from_deltas: alloc quad failed\n";
      std::exit(1);
    }
  
    QuadMapped<D,Real>::build_kappa(nquad, kappa, points, weights);
  
    // ---- basis values V: Q x M with ld_V = Q (column-major in m) ----
    Real* V = (Real*) std::malloc((std::size_t)Q * (std::size_t)M * sizeof(Real));
    if (!V)
    {
      std::cerr << "JMat::build_from_deltas: alloc V failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::eval_all(points, /*ld_point*/ D, /*ld_dim*/ 1, Q,
                            kappa, n, alphaTable, tailDeg, invH,
                            V, /*ld_V*/ Q);
  
    // precompute u[p] = weights[p] * x_coord(points[p])
    Real* u = (Real*) std::malloc((std::size_t)Q * sizeof(Real));
    if (!u)
    {
      std::cerr << "JMat::build_from_deltas: alloc u failed\n";
      std::exit(1);
    }
    for (int p = 0; p < Q; ++p)
    {
      const Real xi = points[(std::size_t)p * (std::size_t)D + (std::size_t)coord];
      u[p] = weights[p] * xi;
    }
  
    // Helper buffers
    int dvec[8];
    int src_alpha[8];
  
    // === Assemble from delta sets ===
    //
    // We fill rows degree-by-degree:
    //   rows in Hom(j) with global row offset row0 = dim_Pi(j-1)
    //
    // For each row basis multi-index "dst", deltas encode:
    //   delta = dst - src  =>  src = dst - delta
    //
    // Bands:
    //   keys0 : rows Hom(j) <- cols Hom(j)
    //   keysm1: rows Hom(j) <- cols Hom(j+1)
    //   keysp1: rows Hom(j) <- cols Hom(j-1)
    //
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int row0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int rj   = Basis<D,Real>::dim_Hom(jdeg);
  
      // ---- band 0: cols degree jdeg ----
      {
        const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
        const int cj   = Basis<D,Real>::dim_Hom(jdeg);
  
        for (int iloc = 0; iloc < rj; ++iloc)
        {
          const int ig = row0 + iloc;
          const int* dst = hom_decode_ptr(jdeg, iloc, alphaTable);
  
          const Real* Vi = V + (std::size_t)ig * (std::size_t)Q;
  
          for (int t = 0; t < S.ndelta0; ++t)
          {
            unpack_delta8(S.keys0[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              src_alpha[d] = dst[d] - dvec[d];
              if (src_alpha[d] < 0) { ok = false; break; }
              sum += src_alpha[d];
            }
            if (!ok) continue;
            if (sum != jdeg) continue;
  
            const int jloc = hom_encode_rank_fast(jdeg, src_alpha);
            if (jloc < 0 || jloc >= cj) continue;
  
            const int jg = col0 + jloc;
            const Real* Vj = V + (std::size_t)jg * (std::size_t)Q;
  
            Real s = Real(0);
            for (int p = 0; p < Q; ++p) s += u[p] * Vi[p] * Vj[p];
  
            Ji[(std::size_t)ig * (std::size_t)N + (std::size_t)jg] = s;
          }
        }
      }
  
      // ---- band m1: cols degree jdeg+1 (rows <- higher degree cols) ----
      if (jdeg + 1 <= n)
      {
        const int sdeg = jdeg + 1;
        const int col0 = Basis<D,Real>::dim_Pi(sdeg - 1);
        const int cj   = Basis<D,Real>::dim_Hom(sdeg);
  
        for (int iloc = 0; iloc < rj; ++iloc)
        {
          const int ig = row0 + iloc;
          const int* dst = hom_decode_ptr(jdeg, iloc, alphaTable);
  
          const Real* Vi = V + (std::size_t)ig * (std::size_t)Q;
  
          for (int t = 0; t < S.ndeltam1; ++t)
          {
            unpack_delta8(S.keysm1[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              src_alpha[d] = dst[d] - dvec[d];
              if (src_alpha[d] < 0) { ok = false; break; }
              sum += src_alpha[d];
            }
            if (!ok) continue;
            if (sum != sdeg) continue;
  
            const int jloc = hom_encode_rank_fast(sdeg, src_alpha);
            if (jloc < 0 || jloc >= cj) continue;
  
            const int jg = col0 + jloc;
            const Real* Vj = V + (std::size_t)jg * (std::size_t)Q;
  
            Real s = Real(0);
            for (int p = 0; p < Q; ++p) s += u[p] * Vi[p] * Vj[p];
  
            Ji[(std::size_t)ig * (std::size_t)N + (std::size_t)jg] = s;
          }
        }
      }
  
      // ---- band p1: cols degree jdeg-1 (rows <- lower degree cols) ----
      if (jdeg - 1 >= 0)
      {
        const int sdeg = jdeg - 1;
        const int col0 = Basis<D,Real>::dim_Pi(sdeg - 1);
        const int cj   = Basis<D,Real>::dim_Hom(sdeg);
  
        for (int iloc = 0; iloc < rj; ++iloc)
        {
          const int ig = row0 + iloc;
          const int* dst = hom_decode_ptr(jdeg, iloc, alphaTable);
  
          const Real* Vi = V + (std::size_t)ig * (std::size_t)Q;
  
          for (int t = 0; t < S.ndeltap1; ++t)
          {
            unpack_delta8(S.keysp1[t], dvec);
  
            bool ok = true;
            int sum = 0;
            for (int d = 0; d < D; ++d)
            {
              src_alpha[d] = dst[d] - dvec[d];
              if (src_alpha[d] < 0) { ok = false; break; }
              sum += src_alpha[d];
            }
            if (!ok) continue;
            if (sum != sdeg) continue;
  
            const int jloc = hom_encode_rank_fast(sdeg, src_alpha);
            if (jloc < 0 || jloc >= cj) continue;
  
            const int jg = col0 + jloc;
            const Real* Vj = V + (std::size_t)jg * (std::size_t)Q;
  
            Real s = Real(0);
            for (int p = 0; p < Q; ++p) s += u[p] * Vi[p] * Vj[p];
  
            Ji[(std::size_t)ig * (std::size_t)N + (std::size_t)jg] = s;
          }
        }
      }
    }
  
    std::free(u);
    std::free(V);
    std::free(points);
    std::free(weights);
    std::free(alphaTable);
    std::free(tailDeg);
    std::free(invH);
    std::free(r);
    std::free(off);
  }

  // Build J_coord in CSC.
  //
  // Output arrays allocated with malloc:
  //   colptr : length M+1
  //   rowind : length nnz
  //   x      : length nnz
  //
  // Returns nnz.
  static std::size_t build_pruned_csc(int n,
                                      const Real* kappa,     // length D+1
                                      unsigned int nquad,
                                      int coord,             // 0..D-1
                                      int** colptr_out,
                                      int** rowind_out,
                                      Real** x_out)
  {
    if (!kappa || !colptr_out || !rowind_out || !x_out)
    {
      std::cerr << "JMat::build_pruned_csc: null input\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *x_out      = nullptr;
  
    if (n < 0)
    {
      std::cerr << "JMat::build_pruned_csc: require n >= 0\n";
      std::exit(1);
    }
    if (coord < 0 || coord >= D)
    {
      std::cerr << "JMat::build_pruned_csc: coord out of range\n";
      std::exit(1);
    }
    if (nquad == 0) { nquad = 1; }
  
    const int M = Basis<D,Real>::dim_Pi(n);
  
    // Stencil policy consistent with your dense build_pruned():
    const int stencil_min = D + 1;
    const int stencil_max = 4 * D;
  
    // --- Fallback: dense -> CSC (for small n or D==1) ---
    if (D == 1 || n < stencil_min)
    {
      Real* Jdense = (Real*) std::malloc((std::size_t)M * (std::size_t)M * sizeof(Real));
      if (!Jdense)
      {
        std::cerr << "JMat::build_pruned_csc: alloc dense failed\n";
        std::exit(1);
      }
  
      // Build only this coordinate in dense.
      // Your existing build_pruned_dense builds ALL coords into J_all; we can reuse it
      // by allocating D*M*M and slicing, OR (preferred) call build() and prune just this coord.
      //
      // Simplest reuse without refactor:
      Real* Jall = (Real*) std::malloc((std::size_t)D * (std::size_t)M * (std::size_t)M * sizeof(Real));
      if (!Jall)
      {
        std::free(Jdense);
        std::cerr << "JMat::build_pruned_csc: alloc Jall failed\n";
        std::exit(1);
      }
      build_pruned_dense(n, kappa, nquad, Jall);
      std::memcpy(Jdense,
                  Jall + (std::size_t)coord * (std::size_t)M * (std::size_t)M,
                  (std::size_t)M * (std::size_t)M * sizeof(Real));
      std::free(Jall);
  
      const std::size_t nnz =
        detail::compress_dense_to_csc(M, Jdense, colptr_out, rowind_out, x_out);
  
      std::free(Jdense);
      return nnz;
    }
  
    // --- Stencil path ---
    JMatStencil S;
    std::memset(&S, 0, sizeof(S));
  
    discover_stencil_stable(nquad,
                            kappa,
                            coord,
                            stencil_min,
                            stencil_max,
                            &S);
  
    // Build CSC directly using stencil (no dense intermediate).
    const std::size_t nnz =
      build_from_deltas_csc(n, kappa, nquad, coord, S, colptr_out, rowind_out, x_out);
  
    S.clear();
    return nnz;
  }

  static std::size_t build_from_deltas_csc(int n,
                                          const Real* kappa,
                                          unsigned int nquad,
                                          int coord,
                                          const JMatStencil& S,
                                          int** colptr_out,
                                          int** rowind_out,
                                          Real** x_out)
  {
    const int M = Basis<D,Real>::dim_Pi(n);
  
    // ---- Quadrature: weight w_kappa ----
    const unsigned int npts_u = QuadMapped<D,Real>::npoints(nquad);
    const int npts = (int)npts_u;
  
    Real* X  = (Real*) std::malloc((std::size_t)npts * (std::size_t)D * sizeof(Real));
    Real* wq = (Real*) std::malloc((std::size_t)npts * sizeof(Real));
    if (!X || !wq)
    {
      std::cerr << "JMat::build_from_deltas_csc: alloc quad failed\n";
      std::exit(1);
    }
    const int built = QuadMapped<D,Real>::build_kappa(nquad, kappa, X, wq);
    if (built != npts)
    {
      std::cerr << "JMat::build_from_deltas_csc: build_kappa failed\n";
      std::exit(1);
    }
  
    // normalize (matches your other builders; harmless if already normalized)
    Real sw = Real(0);
    for (int p = 0; p < npts; ++p) sw += wq[p];
    if (sw != Real(0))
    {
      const Real inv_sw = Real(1) / sw;
      for (int p = 0; p < npts; ++p) wq[p] *= inv_sw;
    }
  
    // ---- Basis tables for degree n ----
    int* alpha_table = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    int* tail_deg    = (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
    Real* inv_h      = (Real*) std::malloc((std::size_t)M * sizeof(Real));
    if (!alpha_table || !tail_deg || !inv_h)
    {
      std::cerr << "JMat::build_from_deltas_csc: alloc tables failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);
    for (int m = 0; m < M; ++m)
    {
      const int* a = alpha_table + (std::size_t)m * (std::size_t)D;
      inv_h[m] = Basis<D,Real>::inv_h_alpha(a, kappa);
    }
  
    // ---- Evaluate basis values V(x) ----
    const int ldV = npts;
    Real* V = (Real*) std::malloc((std::size_t)npts * (std::size_t)M * sizeof(Real));
    if (!V)
    {
      std::cerr << "JMat::build_from_deltas_csc: alloc V failed\n";
      std::exit(1);
    }
  
    Basis<D,Real>::eval_all(
      X,
      D, 1,
      npts,
      kappa,
      n,
      alpha_table,
      tail_deg,
      inv_h,
      V,
      ldV,
      nullptr
    );
  
    // ---- Pass 1: count nnz per column ----
    int* colnnz = (int*) std::malloc((std::size_t)M * sizeof(int));
    int* colptr = (int*) std::malloc((std::size_t)(M + 1) * sizeof(int));
    if (!colnnz || !colptr)
    {
      std::cerr << "JMat::build_from_deltas_csc: alloc colnnz/colptr failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) colnnz[j] = 0;
  
    int dvec[8];
    int dst_alpha[8];
  
    auto count_keys = [&](int jdeg, int jloc, int ideg, const uint64_t* keys, int nkey) -> int
    {
      if (ideg < 0 || ideg > n) return 0;
  
      const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
      int cnt = 0;
  
      for (int t = 0; t < nkey; ++t)
      {
        unpack_delta8(keys[t], dvec);
  
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
  
        // rank check (defensive)
        const int iloc = hom_encode_rank_fast(ideg, dst_alpha);
        const int r = Basis<D,Real>::dim_Hom(ideg);
        if (iloc < 0 || iloc >= r) continue;
  
        ++cnt;
      }
      return cnt;
    };
  
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int c    = Basis<D,Real>::dim_Hom(jdeg);
  
      for (int jloc = 0; jloc < c; ++jloc)
      {
        const int jg = col0 + jloc;
  
        int cnt = 0;
        // src j -> dst j
        cnt += count_keys(jdeg, jloc, jdeg,   S.keys0,  S.ndelta0);
        // src j -> dst j-1
        cnt += count_keys(jdeg, jloc, jdeg-1, S.keysm1, S.ndeltam1);
        // src j -> dst j+1
        cnt += count_keys(jdeg, jloc, jdeg+1, S.keysp1, S.ndeltap1);
  
        colnnz[jg] = cnt;
      }
    }
  
    colptr[0] = 0;
    for (int j = 0; j < M; ++j) colptr[j + 1] = colptr[j] + colnnz[j];
  
    const int nnz = colptr[M];
  
    int* rowind = (int*) std::malloc((std::size_t)nnz * sizeof(int));
    Real* x     = (Real*) std::malloc((std::size_t)nnz * sizeof(Real));
    int*  wpos  = (int*) std::malloc((std::size_t)M * sizeof(int));
    if ((!rowind && nnz) || (!x && nnz) || !wpos)
    {
      std::cerr << "JMat::build_from_deltas_csc: alloc nnz arrays failed\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j) wpos[j] = colptr[j];
  
    // ---- Pass 2: fill CSC ----
    auto fill_keys = [&](int jdeg, int jloc, int ideg, const uint64_t* keys, int nkey)
    {
      if (ideg < 0 || ideg > n) return;
  
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int jg = col0 + jloc;
  
      const int row0 = Basis<D,Real>::dim_Pi(ideg - 1);
  
      const int* src = hom_decode_ptr(jdeg, jloc, alpha_table);
      const Real* Vj = V + (std::size_t)jg * (std::size_t)ldV;
  
      for (int t = 0; t < nkey; ++t)
      {
        unpack_delta8(keys[t], dvec);
  
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
        const int r = Basis<D,Real>::dim_Hom(ideg);
        if (iloc < 0 || iloc >= r) continue;
  
        const int ig = row0 + iloc;
        const Real* Vi = V + (std::size_t)ig * (std::size_t)ldV;
  
        // s = <phi_i, x_coord * phi_j>_w
        Real s = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          const Real xp = X[(std::size_t)p * (std::size_t)D + (std::size_t)coord];
          s += Vi[p] * wq[p] * xp * Vj[p];
        }
  
        const int pos = wpos[jg]++;
        rowind[pos] = ig;
        x[pos]      = s;
      }
    };
  
    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int c = Basis<D,Real>::dim_Hom(jdeg);
      for (int jloc = 0; jloc < c; ++jloc)
      {
        fill_keys(jdeg, jloc, jdeg,   S.keys0,  S.ndelta0);
        fill_keys(jdeg, jloc, jdeg-1, S.keysm1, S.ndeltam1);
        fill_keys(jdeg, jloc, jdeg+1, S.keysp1, S.ndeltap1);
      }
    }
  
    // hand off outputs
    *colptr_out = colptr;
    *rowind_out = rowind;
    *x_out      = x;
  
    // free temporaries
    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(inv_h);
    std::free(V);
    std::free(colnnz);
    std::free(wpos);
  
    return (std::size_t)nnz;
  }



};

} // namespace jsimplex



#endif // JMAT_H
