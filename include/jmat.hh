#ifndef JMAT_H
#define JMAT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>
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
  
  /* Prune one already-assembled coordinate matrix in place.
     Keeping this separate lets the all-coordinate and one-coordinate builders
     share exactly the same pruning policy. */
  static void prune_dense_coordinate(int n, Real* Ji)
  {
    if (!Ji || n < 0) { return; }

    const int N = Basis<D,Real>::dim_Pi(n);
    constexpr Real JMAT_PRUNE_TOL =
      Real(1000) * std::numeric_limits<Real>::epsilon();

    int* r   = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    int* off = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
    if (!r || !off)
    {
      std::free(r);
      std::free(off);
      return;
    }

    int acc = 0;
    for (int k = 0; k <= n; ++k)
    {
      r[k] = Basis<D,Real>::dim_R(k);
      off[k] = acc;
      acc += r[k];
    }

    for (int k = 0; k <= n; ++k)
    {
      const int rk = r[k];
      const int ok = off[k];

      // B_{k,coord}
      for (int a = 0; a < rk; ++a)
      {
        Real* rowJi = Ji + (std::size_t)(ok + a) * (std::size_t)N;
        for (int b = 0; b < rk; ++b)
        {
          if (std::abs(rowJi[ok + b]) <= JMAT_PRUNE_TOL)
          {
            rowJi[ok + b] = Real(0);
          }
        }
      }

      // A_{k,coord} and its symmetric counterpart.
      if (k < n)
      {
        const int rkp1 = r[k + 1];
        const int okp1 = off[k + 1];

        for (int a = 0; a < rk; ++a)
        {
          for (int b = 0; b < rkp1; ++b)
          {
            Real& upper =
              Ji[(std::size_t)(ok + a) * (std::size_t)N
                 + (std::size_t)(okp1 + b)];
            Real& lower =
              Ji[(std::size_t)(okp1 + b) * (std::size_t)N
                 + (std::size_t)(ok + a)];

            if (std::abs(upper) <= JMAT_PRUNE_TOL) { upper = Real(0); }
            if (std::abs(lower) <= JMAT_PRUNE_TOL) { lower = Real(0); }
          }
        }
      }
    }

    std::free(r);
    std::free(off);
  }

  /* Build one coordinate Jacobi matrix only.

     The former stencil discovery called build_pruned_dense(), which builds all
     D coordinate matrices, and then retained only one block. Since the public
     CSC builder is itself called once per coordinate, that produced D^2 dense
     coordinate assemblies during stencil discovery. This helper preserves the
     same quadrature, basis ordering, accumulation order, and output layout for
     the requested coordinate while avoiding the unused D-1 matrices. */
  static void build_dense_coordinate(int n,
                                     const Real* kappa,
                                     unsigned int nquad,
                                     int coord,
                                     Real* Ji)
  {
    if (!kappa || !Ji || n < 0 || coord < 0 || coord >= D) { return; }

    if constexpr (D == 1)
    {
      build(n, kappa, nquad, Ji);
      return;
    }
    else
    {
      const int N = Basis<D,Real>::dim_Pi(n);
      std::memset(
        Ji,
        0,
        (std::size_t)N * (std::size_t)N * sizeof(Real)
      );

      int* r   = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
      int* off = (int*) std::malloc((std::size_t)(n + 1) * sizeof(int));
      if (!r || !off)
      {
        std::free(r);
        std::free(off);
        return;
      }

      int acc = 0;
      for (int k = 0; k <= n; ++k)
      {
        r[k] = Basis<D,Real>::dim_R(k);
        off[k] = acc;
        acc += r[k];
      }

      const int M = N;
      int* alphaTable =
        (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      int* tailDeg =
        (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      Real* invH =
        (Real*) std::malloc((std::size_t)M * sizeof(Real));
      if (!alphaTable || !tailDeg || !invH)
      {
        std::free(r);
        std::free(off);
        std::free(alphaTable);
        std::free(tailDeg);
        std::free(invH);
        return;
      }

      Basis<D,Real>::build_alpha_table(n, alphaTable);
      Basis<D,Real>::build_tail_deg(n, alphaTable, tailDeg);
      for (int m = 0; m < M; ++m)
      {
        invH[m] =
          Basis<D,Real>::inv_h_alpha(
            alphaTable + (std::size_t)m * (std::size_t)D,
            kappa
          );
      }

      if (nquad == 0) { nquad = 1; }
      unsigned long long Qll = 1ULL;
      for (int i = 0; i < D; ++i)
      {
        Qll *= (unsigned long long)nquad;
      }
      const int Q = (int)Qll;

      Real* points =
        (Real*) std::malloc((std::size_t)Q * (std::size_t)D * sizeof(Real));
      Real* weights =
        (Real*) std::malloc((std::size_t)Q * sizeof(Real));
      if (!points || !weights)
      {
        std::free(r);
        std::free(off);
        std::free(alphaTable);
        std::free(tailDeg);
        std::free(invH);
        std::free(points);
        std::free(weights);
        return;
      }

      QuadMapped<D,Real>::build_kappa(nquad, kappa, points, weights);

      Real* V =
        (Real*) std::malloc((std::size_t)Q * (std::size_t)M * sizeof(Real));
      Real* u =
        (Real*) std::malloc((std::size_t)Q * sizeof(Real));
      if (!V || !u)
      {
        std::free(r);
        std::free(off);
        std::free(alphaTable);
        std::free(tailDeg);
        std::free(invH);
        std::free(points);
        std::free(weights);
        std::free(V);
        std::free(u);
        return;
      }

      Basis<D,Real>::eval_all(
        points,
        D,
        1,
        Q,
        kappa,
        n,
        alphaTable,
        tailDeg,
        invH,
        V,
        Q
      );

      for (int p = 0; p < Q; ++p)
      {
        u[p] = weights[p] * points[(std::size_t)p * (std::size_t)D + coord];
      }

      for (int p = 0; p < Q; ++p)
      {
        const Real up = u[p];

        for (int k = 0; k <= n; ++k)
        {
          const int rk = r[k];
          const int ok = off[k];

          for (int a = 0; a < rk; ++a)
          {
            const Real va = V[p + (std::size_t)(ok + a) * (std::size_t)Q];
            const Real sca = up * va;
            Real* rowJi =
              Ji + (std::size_t)(ok + a) * (std::size_t)N;

            for (int b = 0; b < rk; ++b)
            {
              const Real vb =
                V[p + (std::size_t)(ok + b) * (std::size_t)Q];
              rowJi[ok + b] += sca * vb;
            }
          }

          if (k < n)
          {
            const int rkp1 = r[k + 1];
            const int okp1 = off[k + 1];

            for (int a = 0; a < rk; ++a)
            {
              const Real va =
                V[p + (std::size_t)(ok + a) * (std::size_t)Q];
              const Real sca = up * va;

              for (int b = 0; b < rkp1; ++b)
              {
                const Real vb =
                  V[p + (std::size_t)(okp1 + b) * (std::size_t)Q];
                const Real contrib = sca * vb;

                Ji[(std::size_t)(ok + a) * (std::size_t)N
                   + (std::size_t)(okp1 + b)] += contrib;
                Ji[(std::size_t)(okp1 + b) * (std::size_t)N
                   + (std::size_t)(ok + a)] += contrib;
              }
            }
          }
        }
      }

      std::free(V);
      std::free(u);
      std::free(points);
      std::free(weights);
      std::free(alphaTable);
      std::free(tailDeg);
      std::free(invH);
      std::free(r);
      std::free(off);
    }
  }

  static void build_pruned_dense_coordinate(int n,
                                            const Real* kappa,
                                            unsigned int nquad,
                                            int coord,
                                            Real* Ji)
  {
    build_dense_coordinate(n, kappa, nquad, coord, Ji);
    prune_dense_coordinate(n, Ji);
  }

  static void build_pruned_dense(int n, const Real* kappa, unsigned int nquad, Real* J_all)
  {
    const int N = Basis<D,Real>::dim_Pi(n);

    // Keep the efficient all-coordinate path when all D matrices are wanted:
    // quadrature and basis values are built only once.
    build(n, kappa, nquad, J_all);

    for (int coord = 0; coord < D; ++coord)
    {
      Real* Ji =
        J_all + (std::size_t)coord * (std::size_t)N * (std::size_t)N;
      prune_dense_coordinate(n, Ji);
    }
  }

  /* Multiplication-by-coordinate sparse ops.
     Build J_all as concatenation [J_0 J_1 ... J_{D-1}] where each J_i is MxM.

     Small degrees retain the dense fallback.  Otherwise each coordinate uses
     its persistent degree-independent stencil and only the indicated entries
     are assembled.
  */
  static void build_pruned(int n,
                           const Real* kappa,
                           unsigned int nquad,
                           Real* J_all)
  {
    build_pruned(
      n, kappa, nquad, J_all, default_stencil_folder());
  }

  static void build_pruned(int n,
                           const Real* kappa,
                           unsigned int nquad,
                           Real* J_all,
                           const std::string& stencil_folder)
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
    if (nquad == 0) nquad = 1;

    const int M = Basis<D,Real>::dim_Pi(n);
    const int stencil_min = default_stencil_min_degree();

    if (D == 1 || n < stencil_min)
    {
      build_pruned_dense(n, kappa, nquad, J_all);
      return;
    }

    for (int coord = 0; coord < D; ++coord)
    {
      Real* Ji =
        J_all + static_cast<std::size_t>(coord) *
                static_cast<std::size_t>(M) *
                static_cast<std::size_t>(M);

      JMatStencil S;
      std::memset(&S, 0, sizeof(S));
      load_or_discover_coordinate_stencil(
        nquad, kappa, coord, &S, stencil_folder);
      build_from_deltas(
        n, kappa, nquad, coord, S, Ji);
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

  /* Canonical structural signature used for automatic interning.  The
     coordinate and representative degree are intentionally omitted. */
  static std::string stencil_signature(const JMatStencil& S)
  {
    std::ostringstream out;
    out << "JMat:D=" << D
        << ":ndelta0=" << S.ndelta0 << ":keys0=";
    out << std::hex << std::setfill('0');
    for (int k = 0; k < S.ndelta0; ++k)
    {
      if (k) out << ',';
      out << std::setw(16) << S.keys0[k];
    }

    out << ":ndeltam1=" << std::dec << S.ndeltam1 << ":keysm1=";
    out << std::hex;
    for (int k = 0; k < S.ndeltam1; ++k)
    {
      if (k) out << ',';
      out << std::setw(16) << S.keysm1[k];
    }

    out << ":ndeltap1=" << std::dec << S.ndeltap1 << ":keysp1=";
    out << std::hex;
    for (int k = 0; k < S.ndeltap1; ++k)
    {
      if (k) out << ',';
      out << std::setw(16) << S.keysp1[k];
    }
    return out.str();
  }

  static bool stencil_valid(const JMatStencil& S)
  {
    if (S.ndelta0 < 0 || S.ndeltam1 < 0 || S.ndeltap1 < 0)
      return false;
    if (S.ndelta0 > 0 && !S.keys0) return false;
    if (S.ndeltam1 > 0 && !S.keysm1) return false;
    if (S.ndeltap1 > 0 && !S.keysp1) return false;

    for (int k = 1; k < S.ndelta0; ++k)
    {
      if (!(S.keys0[k - 1] < S.keys0[k])) return false;
    }
    for (int k = 1; k < S.ndeltam1; ++k)
    {
      if (!(S.keysm1[k - 1] < S.keysm1[k])) return false;
    }
    for (int k = 1; k < S.ndeltap1; ++k)
    {
      if (!(S.keysp1[k - 1] < S.keysp1[k])) return false;
    }
    return true;
  }

  static void copy_stencil(const JMatStencil& src, JMatStencil* dst)
  {
    if (!dst || !stencil_valid(src))
    {
      std::cerr << "JMat::copy_stencil: invalid input\n";
      std::exit(1);
    }

    dst->clear();
    dst->j_rep = src.j_rep;
    dst->ndelta0 = src.ndelta0;
    dst->ndeltam1 = src.ndeltam1;
    dst->ndeltap1 = src.ndeltap1;

    auto copy_keys = [](const uint64_t* source,
                        int count,
                        uint64_t** destination)
    {
      *destination = nullptr;
      if (count == 0) return;
      *destination = static_cast<uint64_t*>(
        std::malloc(static_cast<std::size_t>(count) * sizeof(uint64_t)));
      if (!*destination)
      {
        std::cerr << "JMat::copy_stencil: allocation failed\n";
        std::exit(1);
      }
      std::memcpy(
        *destination,
        source,
        static_cast<std::size_t>(count) * sizeof(uint64_t));
    };

    copy_keys(src.keys0, src.ndelta0, &dst->keys0);
    copy_keys(src.keysm1, src.ndeltam1, &dst->keysm1);
    copy_keys(src.keysp1, src.ndeltap1, &dst->keysp1);
  }

  static std::filesystem::path stencil_cache_path(
    int coord,
    const std::string& stencil_folder = "stencils")
  {
    if (coord < 0 || coord >= D)
    {
      std::cerr << "JMat::stencil_cache_path: coordinate out of range\n";
      std::exit(1);
    }

    const std::filesystem::path folder =
      stencil_folder.empty() ? std::filesystem::path(".")
                             : std::filesystem::path(stencil_folder);
    std::ostringstream name;
    name << "jmat_D" << D
         << "_coord" << coord
         << "_v" << stencil_cache_version()
         << ".stencil";
    return folder / name.str();
  }

  /* Return false only when the keyed file is absent.  A present but invalid
     file is a hard error so stale or corrupt cache data is never used. */
  static bool load_stencil_file(
    int coord,
    JMatStencil* S_out,
    const std::string& stencil_folder = "stencils")
  {
    if (!S_out)
    {
      std::cerr << "JMat::load_stencil_file: null output\n";
      std::exit(1);
    }

    const std::filesystem::path path =
      stencil_cache_path(coord, stencil_folder);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
      if (ec)
      {
        std::cerr << "JMat::load_stencil_file: exists failed for "
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
    int file_coord = -1;
    int j_rep = 0;
    int ndelta0 = -1;
    int ndeltam1 = -1;
    int ndeltap1 = -1;

    if (!(in >> magic >> version >> kind >> basis_tag
             >> file_D >> file_coord >> j_rep
             >> ndelta0 >> ndeltam1 >> ndeltap1))
    {
      std::cerr << "JMat::load_stencil_file: malformed header in "
                << path << '\n';
      std::exit(1);
    }

    if (magic != "JPOLYD_STENCIL" ||
        version != stencil_cache_version() ||
        kind != "JMAT" ||
        basis_tag != "KAPPA_MINUS_HALF" ||
        file_D != D ||
        file_coord != coord ||
        ndelta0 < 0 ||
        ndeltam1 < 0 ||
        ndeltap1 < 0)
    {
      std::cerr << "JMat::load_stencil_file: incompatible cache file "
                << path << '\n';
      std::exit(1);
    }

    JMatStencil loaded;
    std::memset(&loaded, 0, sizeof(loaded));
    loaded.j_rep = j_rep;
    loaded.ndelta0 = ndelta0;
    loaded.ndeltam1 = ndeltam1;
    loaded.ndeltap1 = ndeltap1;

    auto read_keys = [&](int count,
                         uint64_t** keys,
                         const char* label)
    {
      *keys = nullptr;
      if (count == 0) return;
      *keys = static_cast<uint64_t*>(
        std::malloc(static_cast<std::size_t>(count) * sizeof(uint64_t)));
      if (!*keys)
      {
        loaded.clear();
        std::cerr << "JMat::load_stencil_file: allocation failed\n";
        std::exit(1);
      }
      for (int i = 0; i < count; ++i)
      {
        if (!(in >> (*keys)[i]))
        {
          loaded.clear();
          std::cerr << "JMat::load_stencil_file: truncated "
                    << label << " in " << path << '\n';
          std::exit(1);
        }
      }
    };

    read_keys(loaded.ndelta0, &loaded.keys0, "keys0");
    read_keys(loaded.ndeltam1, &loaded.keysm1, "keysm1");
    read_keys(loaded.ndeltap1, &loaded.keysp1, "keysp1");

    std::string end_token;
    if (!(in >> end_token) ||
        end_token != "END" ||
        !stencil_valid(loaded))
    {
      loaded.clear();
      std::cerr << "JMat::load_stencil_file: invalid stencil in "
                << path << '\n';
      std::exit(1);
    }

    S_out->clear();
    *S_out = loaded;
    std::memset(&loaded, 0, sizeof(loaded));
    return true;
  }

  static void save_stencil_file(
    int coord,
    const JMatStencil& S,
    const std::string& stencil_folder = "stencils")
  {
    if (!stencil_valid(S))
    {
      std::cerr << "JMat::save_stencil_file: invalid stencil\n";
      std::exit(1);
    }

    const std::filesystem::path path =
      stencil_cache_path(coord, stencil_folder);
    const std::filesystem::path folder = path.parent_path();
    std::error_code ec;
    if (!folder.empty())
    {
      std::filesystem::create_directories(folder, ec);
      if (ec)
      {
        std::cerr << "JMat::save_stencil_file: cannot create "
                  << folder << ": " << ec.message() << '\n';
        std::exit(1);
      }
    }

    const std::filesystem::path temporary = path.string() + ".tmp";
    {
      std::ofstream out(temporary, std::ios::trunc);
      if (!out)
      {
        std::cerr << "JMat::save_stencil_file: cannot open "
                  << temporary << '\n';
        std::exit(1);
      }

      out << "JPOLYD_STENCIL " << stencil_cache_version()
          << " JMAT KAPPA_MINUS_HALF "
          << D << ' ' << coord << ' '
          << S.j_rep << ' '
          << S.ndelta0 << ' '
          << S.ndeltam1 << ' '
          << S.ndeltap1 << '\n';
      for (int i = 0; i < S.ndelta0; ++i) out << S.keys0[i] << '\n';
      for (int i = 0; i < S.ndeltam1; ++i) out << S.keysm1[i] << '\n';
      for (int i = 0; i < S.ndeltap1; ++i) out << S.keysp1[i] << '\n';
      out << "END\n";
      out.flush();
      if (!out)
      {
        std::cerr << "JMat::save_stencil_file: write failed for "
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
      std::cerr << "JMat::save_stencil_file: rename failed for "
                << path << ": " << ec.message() << '\n';
      std::exit(1);
    }
  }

  static void discover_coordinate_stencil(
    unsigned int nquad,
    const Real* kappa,
    int coord,
    int n_min,
    int n_max,
    JMatStencil* S_out)
  {
    discover_stencil_stable(
      nquad, kappa, coord, n_min, n_max, S_out);
  }

  /* Main persistent path.  Returns true when loaded and false when newly
     discovered and written. */
  static bool load_or_discover_coordinate_stencil(
    unsigned int nquad,
    const Real* kappa,
    int coord,
    JMatStencil* S_out,
    const std::string& stencil_folder = "stencils",
    int n_min = default_stencil_min_degree(),
    int n_max = default_stencil_max_degree())
  {
    if (!kappa || !S_out)
    {
      std::cerr
        << "JMat::load_or_discover_coordinate_stencil: null input\n";
      std::exit(1);
    }
    if (coord < 0 || coord >= D)
    {
      std::cerr
        << "JMat::load_or_discover_coordinate_stencil: coordinate out of range\n";
      std::exit(1);
    }
    if (nquad == 0) nquad = 1;

    if (load_stencil_file(coord, S_out, stencil_folder))
      return true;

    discover_coordinate_stencil(
      nquad, kappa, coord, n_min, n_max, S_out);
    save_stencil_file(coord, *S_out, stencil_folder);
    return false;
  }

  static bool load_or_discover_stencil(
    unsigned int nquad,
    const Real* kappa,
    int coord,
    JMatStencil* S_out,
    const std::string& stencil_folder = "stencils",
    int n_min = default_stencil_min_degree(),
    int n_max = default_stencil_max_degree())
  {
    return load_or_discover_coordinate_stencil(
      nquad, kappa, coord, S_out, stencil_folder, n_min, n_max);
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
  
      // Build only the requested coordinate. The former implementation
      // assembled all D coordinate matrices here and discarded D-1 of them.
      Real* Jdense =
        (Real*) std::malloc((std::size_t)M * (std::size_t)M * sizeof(Real));
      int* alpha_table =
        (int*) std::malloc((std::size_t)M * (std::size_t)D * sizeof(int));
      if (!Jdense || !alpha_table)
      {
        std::free(Jdense);
        std::free(alpha_table);
        std::cerr << "discover_stencil_stable: alloc failed\n";
        std::exit(1);
      }

      build_pruned_dense_coordinate(
        n_test,
        kappa,
        nquad,
        coord,
        Jdense
      );

      Basis<D,Real>::build_alpha_table(n_test, alpha_table);

      extract_deltas_from_block(
        n_test,
        Jdense,
        alpha_table,
        &S_cur
      );
  
      std::free(Jdense);
      std::free(alpha_table);
  
      if (n_test > n_min && deltas_equal(S_cur, S_prev))
      {
        // stabilized
        S_prev.clear();
        S_out->clear();
        *S_out = S_cur;                // shallow move
        std::memset(&S_cur, 0, sizeof(S_cur));
        std::cout << "JMat: stabilized at n_test = " << n_test << std::endl;
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

  /* Build only the finite-degree CSC structure obtained by truncating a
     degree-independent JMat delta stencil.  No quadrature or parameter values
     are used. */
  static std::size_t build_csc_pattern_from_stencil(
    int n,
    const JMatStencil& S,
    int** colptr_out,
    int** rowind_out)
  {
    if (!colptr_out || !rowind_out || !stencil_valid(S))
    {
      std::cerr << "JMat::build_csc_pattern_from_stencil: invalid input\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;

    if (n < 0)
    {
      std::cerr << "JMat::build_csc_pattern_from_stencil: n < 0\n";
      std::exit(1);
    }

    const int M = Basis<D,Real>::dim_Pi(n);
    int* colptr = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(M + 1) * sizeof(int)));
    int* alpha_table = static_cast<int*>(
      std::malloc(static_cast<std::size_t>(M) *
                  static_cast<std::size_t>(D) * sizeof(int)));
    if (!colptr || !alpha_table)
    {
      std::free(colptr);
      std::free(alpha_table);
      std::cerr << "JMat::build_csc_pattern_from_stencil: allocation failed\n";
      std::exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    colptr[0] = 0;

    std::vector<int> rows;
    rows.reserve(
      static_cast<std::size_t>(M) *
      static_cast<std::size_t>(
        S.ndelta0 + S.ndeltam1 + S.ndeltap1));

    int dvec[8];
    int dst_alpha[8];

    auto append_keys = [&](int jdeg,
                           int jloc,
                           int ideg,
                           const uint64_t* keys,
                           int nkey)
    {
      if (ideg < 0 || ideg > n) return;

      const int* src =
        hom_decode_ptr(jdeg, jloc, alpha_table);
      const int row0 = Basis<D,Real>::dim_Pi(ideg - 1);
      const int rdeg = Basis<D,Real>::dim_Hom(ideg);

      for (int t = 0; t < nkey; ++t)
      {
        unpack_delta8(keys[t], dvec);

        bool ok = true;
        int sum = 0;
        for (int dim = 0; dim < D; ++dim)
        {
          dst_alpha[dim] = src[dim] + dvec[dim];
          if (dst_alpha[dim] < 0)
          {
            ok = false;
            break;
          }
          sum += dst_alpha[dim];
        }
        if (!ok || sum != ideg) continue;

        const int iloc = hom_encode_rank_fast(ideg, dst_alpha);
        if (iloc < 0 || iloc >= rdeg) continue;
        rows.push_back(row0 + iloc);
      }
    };

    for (int jdeg = 0; jdeg <= n; ++jdeg)
    {
      const int col0 = Basis<D,Real>::dim_Pi(jdeg - 1);
      const int cdeg = Basis<D,Real>::dim_Hom(jdeg);

      for (int jloc = 0; jloc < cdeg; ++jloc)
      {
        const int jg = col0 + jloc;
        append_keys(
          jdeg, jloc, jdeg,
          S.keys0, S.ndelta0);
        append_keys(
          jdeg, jloc, jdeg - 1,
          S.keysm1, S.ndeltam1);
        append_keys(
          jdeg, jloc, jdeg + 1,
          S.keysp1, S.ndeltap1);
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
        std::free(colptr);
        std::free(alpha_table);
        std::cerr
          << "JMat::build_csc_pattern_from_stencil: allocation failed\n";
        std::exit(1);
      }
      std::memcpy(
        rowind,
        rows.data(),
        rows.size() * sizeof(int));
    }

    std::free(alpha_table);
    *colptr_out = colptr;
    *rowind_out = rowind;
    return rows.size();
  }

  /* Fill values on a supplied CSC pattern.  This also supports a union
     stencil: every supplied row/column slot is evaluated numerically. */
  static void fill_pruned_csc_values(
    int n,
    const Real* kappa,
    unsigned int nquad,
    int coord,
    const int* colptr,
    const int* rowind,
    Real* values)
  {
    if (!kappa || !colptr || n < 0)
    {
      std::cerr << "JMat::fill_pruned_csc_values: invalid input\n";
      std::exit(1);
    }
    if (coord < 0 || coord >= D)
    {
      std::cerr << "JMat::fill_pruned_csc_values: coordinate out of range\n";
      std::exit(1);
    }
    if (nquad == 0) nquad = 1;

    const int M = Basis<D,Real>::dim_Pi(n);
    if (colptr[0] != 0)
    {
      std::cerr << "JMat::fill_pruned_csc_values: colptr[0] != 0\n";
      std::exit(1);
    }
    for (int j = 0; j < M; ++j)
    {
      if (colptr[j + 1] < colptr[j])
      {
        std::cerr << "JMat::fill_pruned_csc_values: invalid colptr\n";
        std::exit(1);
      }
    }

    const int nnz = colptr[M];
    if (nnz > 0 && (!rowind || !values))
    {
      std::cerr << "JMat::fill_pruned_csc_values: null nnz arrays\n";
      std::exit(1);
    }
    for (int p = 0; p < nnz; ++p)
    {
      if (rowind[p] < 0 || rowind[p] >= M)
      {
        std::cerr << "JMat::fill_pruned_csc_values: row out of range\n";
        std::exit(1);
      }
    }

    const unsigned int npts_u =
      QuadMapped<D,Real>::npoints(nquad);
    const int npts = static_cast<int>(npts_u);

    Real* X = static_cast<Real*>(
      std::malloc(
        static_cast<std::size_t>(npts) *
        static_cast<std::size_t>(D) * sizeof(Real)));
    Real* wq = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(npts) * sizeof(Real)));
    if (!X || !wq)
    {
      std::free(X);
      std::free(wq);
      std::cerr << "JMat::fill_pruned_csc_values: allocation failed\n";
      std::exit(1);
    }

    const int built =
      QuadMapped<D,Real>::build_kappa(nquad, kappa, X, wq);
    if (built != npts)
    {
      std::free(X);
      std::free(wq);
      std::cerr << "JMat::fill_pruned_csc_values: quadrature failed\n";
      std::exit(1);
    }

    Real sw = Real(0);
    for (int p = 0; p < npts; ++p) sw += wq[p];
    if (sw != Real(0))
    {
      const Real inv_sw = Real(1) / sw;
      for (int p = 0; p < npts; ++p) wq[p] *= inv_sw;
    }

    int* alpha_table = static_cast<int*>(
      std::malloc(
        static_cast<std::size_t>(M) *
        static_cast<std::size_t>(D) * sizeof(int)));
    int* tail_deg = static_cast<int*>(
      std::malloc(
        static_cast<std::size_t>(M) *
        static_cast<std::size_t>(D) * sizeof(int)));
    Real* inv_h = static_cast<Real*>(
      std::malloc(static_cast<std::size_t>(M) * sizeof(Real)));
    Real* V = static_cast<Real*>(
      std::malloc(
        static_cast<std::size_t>(npts) *
        static_cast<std::size_t>(M) * sizeof(Real)));

    if (!alpha_table || !tail_deg || !inv_h || !V)
    {
      std::free(X);
      std::free(wq);
      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h);
      std::free(V);
      std::cerr << "JMat::fill_pruned_csc_values: allocation failed\n";
      std::exit(1);
    }

    Basis<D,Real>::build_alpha_table(n, alpha_table);
    Basis<D,Real>::build_tail_deg(n, alpha_table, tail_deg);
    for (int m = 0; m < M; ++m)
    {
      const int* alpha =
        alpha_table +
        static_cast<std::size_t>(m) * static_cast<std::size_t>(D);
      inv_h[m] = Basis<D,Real>::inv_h_alpha(alpha, kappa);
    }

    const int ldV = npts;
    Basis<D,Real>::eval_all(
      X,
      D,
      1,
      npts,
      kappa,
      n,
      alpha_table,
      tail_deg,
      inv_h,
      V,
      ldV,
      nullptr);

    for (int j = 0; j < M; ++j)
    {
      const Real* Vj =
        V + static_cast<std::size_t>(j) *
            static_cast<std::size_t>(ldV);

      for (int pentry = colptr[j];
           pentry < colptr[j + 1];
           ++pentry)
      {
        const int i = rowind[pentry];
        const Real* Vi =
          V + static_cast<std::size_t>(i) *
              static_cast<std::size_t>(ldV);

        Real sum = Real(0);
        for (int p = 0; p < npts; ++p)
        {
          const Real xp =
            X[static_cast<std::size_t>(p) *
              static_cast<std::size_t>(D) +
              static_cast<std::size_t>(coord)];
          sum += Vi[p] * wq[p] * xp * Vj[p];
        }
        values[pentry] = sum;
      }
    }

    std::free(X);
    std::free(wq);
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(inv_h);
    std::free(V);
  }

  static std::size_t build_pruned_csc_from_stencil(
    int n,
    const Real* kappa,
    unsigned int nquad,
    int coord,
    const JMatStencil& S,
    int** colptr_out,
    int** rowind_out,
    Real** values_out)
  {
    if (!kappa || !colptr_out || !rowind_out || !values_out)
    {
      std::cerr << "JMat::build_pruned_csc_from_stencil: null input\n";
      std::exit(1);
    }

    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *values_out = nullptr;

    const std::size_t nnz =
      build_csc_pattern_from_stencil(
        n, S, colptr_out, rowind_out);

    if (nnz > 0)
    {
      *values_out = static_cast<Real*>(
        std::malloc(nnz * sizeof(Real)));
      if (!*values_out)
      {
        std::free(*colptr_out);
        std::free(*rowind_out);
        *colptr_out = nullptr;
        *rowind_out = nullptr;
        std::cerr
          << "JMat::build_pruned_csc_from_stencil: allocation failed\n";
        std::exit(1);
      }
    }

    fill_pruned_csc_values(
      n, kappa, nquad, coord,
      *colptr_out, *rowind_out, *values_out);
    return nnz;
  }

  /* Backward-compatible name for callers that already hold a stencil. */
  static std::size_t build_from_deltas_csc(
    int n,
    const Real* kappa,
    unsigned int nquad,
    int coord,
    const JMatStencil& S,
    int** colptr_out,
    int** rowind_out,
    Real** values_out)
  {
    return build_pruned_csc_from_stencil(
      n, kappa, nquad, coord, S,
      colptr_out, rowind_out, values_out);
  }

  static std::size_t build_pruned_csc(
    int n,
    const Real* kappa,
    unsigned int nquad,
    int coord,
    int** colptr_out,
    int** rowind_out,
    Real** values_out)
  {
    return build_pruned_csc(
      n, kappa, nquad, coord,
      colptr_out, rowind_out, values_out,
      default_stencil_folder());
  }

  static std::size_t build_pruned_csc(
    int n,
    const Real* kappa,
    unsigned int nquad,
    int coord,
    int** colptr_out,
    int** rowind_out,
    Real** values_out,
    const std::string& stencil_folder)
  {
    if (!kappa || !colptr_out || !rowind_out || !values_out)
    {
      std::cerr << "JMat::build_pruned_csc: null input\n";
      std::exit(1);
    }
    *colptr_out = nullptr;
    *rowind_out = nullptr;
    *values_out = nullptr;

    if (n < 0)
    {
      std::cerr << "JMat::build_pruned_csc: require n >= 0\n";
      std::exit(1);
    }
    if (coord < 0 || coord >= D)
    {
      std::cerr << "JMat::build_pruned_csc: coordinate out of range\n";
      std::exit(1);
    }
    if (nquad == 0) nquad = 1;

    const int M = Basis<D,Real>::dim_Pi(n);
    const int stencil_min = default_stencil_min_degree();

    if (D == 1 || n < stencil_min)
    {
      Real* dense = static_cast<Real*>(
        std::malloc(
          static_cast<std::size_t>(M) *
          static_cast<std::size_t>(M) * sizeof(Real)));
      if (!dense)
      {
        std::cerr << "JMat::build_pruned_csc: allocation failed\n";
        std::exit(1);
      }

      build_pruned_dense_coordinate(
        n, kappa, nquad, coord, dense);
      const std::size_t nnz =
        detail::compress_dense_to_csc(
          M, dense, colptr_out, rowind_out, values_out);
      std::free(dense);
      return nnz;
    }

    JMatStencil S;
    std::memset(&S, 0, sizeof(S));
    load_or_discover_coordinate_stencil(
      nquad, kappa, coord, &S, stencil_folder);
    const std::size_t nnz =
      build_pruned_csc_from_stencil(
        n, kappa, nquad, coord, S,
        colptr_out, rowind_out, values_out);
    S.clear();
    return nnz;
  }


};

} // namespace jsimplex



#endif // JMAT_H
