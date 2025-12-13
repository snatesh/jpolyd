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

template<int D, class Real>
struct KMat
{
  static void build_tprod(int n,
                          unsigned int q,          // 1D points per axis
                          const Real* kappa_src,   // length D+1
                          const Real* kappa_tgt,   // length D+1
                          Real* K)                 // row-major MxM
  {
    if (!kappa_src || !kappa_tgt || !K || n < 0 || q == 0)
    {
      return;
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
      return;
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
      return;
    }

    const int built = QuadMapped<D,Real>::build_kappa(q, kappa_tgt, X, wq);
    if (built != npts)
    {
      std::cerr << "KMat::build_tprod: build_kappa failed\n";
      std::exit(1);
    }

    //// Normalize weights to sum to 1 (matches your quadrature-file convention).
    //Real sw = Real(0);
    //for (int p = 0; p < npts; ++p) { sw += wq[p]; }
    //if (sw != Real(0))
    //{
    //  const Real inv_sw = Real(1) / sw;
    //  for (int p = 0; p < npts; ++p) { wq[p] *= inv_sw; }
    //}

    // Evaluate basis at quadrature points.
    // Store V[p + m*npts] (ldV = npts), like your earlier code.
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
      return;
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
      Real(100) * std::numeric_limits<Real>::epsilon();
    
    #pragma omp parallel for schedule(static)
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
};

} // namespace jsimplex

#endif // JKMAT_H

