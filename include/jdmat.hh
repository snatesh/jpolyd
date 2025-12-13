#ifndef JPOLYD_DMAT_HH
#define JPOLYD_DMAT_HH

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <limits>

#include "jbasis.hh"
#include "jquad_tprod.hh"   // QuadMapped<D,Real>

namespace jsimplex
{

template<int D, class Real>
struct DMat
{
  // Build dense derivative projection matrix:
  //
  //   Dout[i,j] = < phi_rng_i(kappa_rng), d/dx_axis phi_src_j(kappa_src) >_{w(kappa_rng)}
  //
  // using mapped tensor-product quadrature built for kappa_rng.
  //
  // Output:
  //   Dout is row-major MxM, where M = dim_Pi(n).
  //
  // Notes:
  // - No pruning here (we’ll use this to corroborate which kappa_rng choices yield sparsity).
  // - eval_all is called with a valid V pointer (not nullptr) because your implementation
  //   likely expects it.
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

// Natural parameter shift for coordinate derivative:
//   kappa_rng = kappa_src + e_axis + e_D  (where last index is D)
//
// Builds dense Dout via build_tprod(...), then:
//   (1) enforces the exact degree-drop nullspace: rows for total degree n are set to 0
//       (i.e. rows i >= dim_Pi(n-1) are identically zero for a first derivative)
//   (2) applies row-relative pruning ONLY on the active rows:
//       |D_ij| < (100*eps*||row_i||_2) => 0
static void build_tprod_natural_pruned(int n,
                                       unsigned int q,
                                       const Real* kappa_src,  // length D+1
                                       int axis,               // 0..D-1
                                       Real* Dout)             // (M*M) row-major
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
  const Real rel_factor = Real(100) * eps;

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


};

} // namespace jsimplex

#endif

