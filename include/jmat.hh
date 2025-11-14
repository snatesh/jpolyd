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
};

} // namespace jsimplex



#endif // JMAT_H
