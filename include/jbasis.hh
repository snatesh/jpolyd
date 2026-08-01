#ifndef JBASIS_H
#define JBASIS_H

#include <cmath>
#include <cassert>
#include <jdetail.hh>
#include <omp.h>
#include <stdexcept>
#include <vector>

/* Evaluators for the Jacobi basis on the D-simplex. */


namespace jsimplex
{

template<int D, class Real>
struct Basis
{
  /* Dimension of Pi_n^D, the space of polynomials of total degree
     at most n in D variables:
       dim Pi_n^D = Choose(n + D, D). */
  static inline int dim_Pi(int n)
  {
    if (n < 0) { return 0; }
    long long numer = 1;
    long long denom = 1;

    for (int k = 1; k <= D; ++k)
    {
      numer *= (long long)(n + k);
      denom *= (long long)k;
    }

    long long val = numer / denom;
    return (int)val;
  }
  
  /* Dimension of homogeneous space of polynomials of total degree 
     exactly n in D variables */
  static inline int dim_Hom(int n)
  {
    return dim_Pi(n) - dim_Pi(n-1);
  }

  /* Dimension of R_n^D, the homogeneous space of total degree
     exactly n in D variables:

       dim R_n^D = Choose(n + D - 1, D - 1). */
  static inline int dim_R(int n)
  {
    if (n < 0)
    {
      return 0;
    }

    long long numer = 1;
    long long denom = 1;
    int kmax = D - 1;

    for (int k = 1; k <= kmax; ++k)
    {
      numer *= (long long)(n + k);
      denom *= (long long)k;
    }

    long long val = numer / denom;
    return (int)val;
  }

  /* Build the multi-index table alpha_table (M x D) for all alpha
     with total degree |alpha| <= n, in graded lexicographic order.

     Here M = dim_Pi(n).

     Layout:
       alpha_table[m*D + j] = alpha_j for row m, component j. */
  static void build_alpha_table(int n, int* alpha_table)
  {
    if (!alpha_table)
    {
      return;
    }

    int alpha[D];
    int m = 0;

    for (int total = 0; total <= n; ++total)
    {
      fill_degree_rec(total, 0, alpha, alpha_table, m);
    }
  }

  /* Given alpha_table (M x D) in the same graded lex order,
     build the tail-degree table tail_deg (M x D), where

       tail_deg[m*D + j] = sum_{r=j+1}^{D-1} alpha_table[m*D + r].

     This is the integer tail degree |alpha^{j+1}| used by the
     Dunkl-Xu parameters. */
  static void build_tail_deg(int n,
                             const int* alpha_table,
                             int* tail_deg)
  {
    if (!alpha_table || !tail_deg)
    {
      return;
    }

    int M = dim_Pi(n);

    for (int m = 0; m < M; ++m)
    {
      const int* arow = alpha_table + m * D;
      int* trow       = tail_deg     + m * D;

      for (int j = 0; j < D; ++j)
      {
        int sum = 0;
        for (int r = j + 1; r < D; ++r)
        {
          sum += arow[r];
        }
        trow[j] = sum;
      }
    }
  }

  /* Compute inv_h_alpha = 1 / h_alpha for a given multi-index alpha
     and parameter vector kappa, using the Dunkl-Xu structure formula.

     alpha has length D
     kappa has length D+1 (kappa_1,...,kappa_{D+1}). */
  static inline Real inv_h_alpha(const int* alpha,
                                 const Real* kappa)
  {
    if (!alpha || !kappa)
    {
      return Real(0.0);
    }
    for (int j = 0; j <= D; ++j)
    {
      const double kappa_j = static_cast<double>(kappa[j]);
      if (!std::isfinite(kappa_j) || !(kappa[j] > Real(-0.5)))
      {
        throw std::runtime_error(
          "Jacobi simplex params kappa must be finite and > -1/2");
      }
    }

    Real log_h2 = Real(0.0);

    for (int j = 0; j < D; ++j)
    {
      int  alpha_j = alpha[j];
      Real kappa_j = kappa[j];

      int abs_alpha_tail_j = 0;
      for (int m = j; m < D; ++m)
      {
        abs_alpha_tail_j += alpha[m];
      }

      int abs_alpha_tail_j1 = 0;
      for (int m = j + 1; m < D; ++m)
      {
        abs_alpha_tail_j1 += alpha[m];
      }

      Real abs_kappa_tail_j = Real(0.0);
      for (int m = j; m <= D; ++m)
      {
        abs_kappa_tail_j += kappa[m];
      }

      Real abs_kappa_tail_j1 = Real(0.0);
      for (int m = j + 1; m <= D; ++m)
      {
        abs_kappa_tail_j1 += kappa[m];
      }

      Real aj = Real(2.0) * static_cast<Real>(abs_alpha_tail_j1)
              + abs_kappa_tail_j1
              + Real(0.5) * static_cast<Real>(D - j - 2);

      int s = abs_alpha_tail_j + abs_alpha_tail_j1;

      Real contrib = Real(0.0);

      contrib += detail::log_pochhammer(
                   kappa_j + Real(0.5),
                   alpha_j
                 );

      Real A2 = abs_kappa_tail_j1
              + Real(0.5) * static_cast<Real>(D - j);
      contrib += detail::log_pochhammer(A2, s);

      Real A4 = abs_kappa_tail_j
              + Real(0.5) * static_cast<Real>(D - j + 1);
      contrib -= detail::log_pochhammer(A4, s);

      if (alpha_j > 1)
      {
        double arg = static_cast<double>(alpha_j + 1);
        contrib -= static_cast<Real>(std::lgamma(arg));
      }

      /*
       * The final norm factor is t5/t6. When alpha_j == 0,
       * t5 == t6 identically, so this factor is exactly one and
       * its logarithmic contribution is exactly zero.
       *
       * For kappa == 0, the last collapsed coordinate has
       * aj = -1/2 and kappa_j = 0. If alpha_j == 0, this
       * representation gives t5 = t6 = 0 and the old code forms
       * log(0) - log(0), even though the underlying ratio is one.
       */
      if (alpha_j > 0)
      {
        Real t5 = Real(2.0)
                  * (aj + kappa_j + static_cast<Real>(alpha_j))
                  + Real(1.0);
        Real t6 = Real(2.0)
                  * (aj + kappa_j + static_cast<Real>(2 * alpha_j))
                  + Real(1.0);

        assert(t5 > Real(0) &&
               "jbasis.hh: inv_h_alpha: t5 must be > 0");
        assert(t6 > Real(0) &&
               "jbasis.hh: inv_h_alpha: t6 must be > 0");

        contrib += static_cast<Real>(
          std::log(static_cast<double>(t5))
          - std::log(static_cast<double>(t6))
        );
      }

      log_h2 += contrib;
    }

    Real inv_h = static_cast<Real>(
      std::exp(-0.5 * static_cast<double>(log_h2))
    );

    return inv_h;
  }

  /* Evaluate all basis functions of total degree <= n at all points.
    
     Inputs:
       X:      point coordinates, layout X[p*ld_point + j*ld_dim],
               p = 0..npts-1, j = 0..D-1
       ld_point, ld_dim: leading dimensions for X
       npts:   number of points
       kappa:  parameters (kappa[0..D]) of length D+1
       n:      maximum total degree
       alpha_table: M x D, graded lex order, row-major
       tail_deg:    M x D, same layout, |alpha^{j+1}|
       inv_h:       length M, 1 / h_alpha
    
     Output:
       V:    values V[p + m*ld_V] = basis_m(x_p)
       ld_V: leading dimension for V in point index */
  static void eval_all(const Real* X,
                       int ld_point,
                       int ld_dim,
                       int npts,
                       const Real* kappa,
                       int n,
                       const int* alpha_table,
                       const int* tail_deg,
                       const Real* inv_h,
                       Real* V,
                       int ld_V,
                       Real* dV = nullptr)
  {
    if (!X || !kappa || !alpha_table || !tail_deg || !inv_h || !V)
    {
      return;
    }
  
    int M = dim_Pi(n);
  
    // Ktail[j] = sum_{r=j+1}^D kappa[r]
    Real Ktail[D];
    for (int j = 0; j < D; ++j)
    {
      Real sum = Real(0.0);
      for (int r = j + 1; r <= D; ++r)
      {
        sum += kappa[r];
      }
      Ktail[j] = sum;
    }
  
    // Thresholds for detecting proximity to the singular faces.
    const Real eps_face     = Real(1e-14);
    const Real eps_tol_over = Real(1e-14);
  
    //#pragma omp parallel for schedule(static)
    for (int p = 0; p < npts; ++p)
    {
      Real xloc[D];
      Real one_minus[D];
      Real t[D];
      bool near_face_j[D];
      bool near_face_point = false;
  
      Real s = Real(0.0);
      for (int j = 0; j < D; ++j)
      {
        Real xpj = X[p * ld_point + j * ld_dim];
        xloc[j] = xpj;
  
        // true 1 - |x^{j-1}|
        Real omr = Real(1.0) - s;
  
        // detect proximity to the face: 1 - s ≈ 0
        if (omr <= eps_face && omr >= -eps_tol_over)
        {
          near_face_j[j] = true;
          near_face_point = true;
        }
        else
        {
          near_face_j[j] = false;
        }
  
        // safety clamp
        Real om = omr;
        if (om <= Real(0.0))
        {
          // clamp only to avoid division by zero when we are not in limit regime
          om = Real(1.0e-14);
        }
  
        one_minus[j] = om;
        t[j]         = (Real(2.0) * xpj / om) - Real(1.0);
        s           += xpj;
      }
  
      // --- main loop over basis functions m ---
      for (int m = 0; m < M; ++m)
      {
        const int* arow = alpha_table + m * D;
        const int* trow = tail_deg    + m * D;
  
        Real val = inv_h[m];
  
        Real F[D];          // per-level factor
        Real omega_pow[D];  // omega_j^{n_j}
        Real Pnj[D];        // P_{n_j}(t_j) (for analytic grad path)
  
        for (int j = 0; j < D; ++j)
        {
          int n_j    = arow[j];
          int tail_j = trow[j];
  
          Real a_j = Real(2.0) * static_cast<Real>(tail_j)
                   + Ktail[j]
                   + Real(0.5) * static_cast<Real>(D - j - 2);
  
          Real b_j = kappa[j] - Real(0.5);
  
          Real F_j  = Real(1.0);
          Real opow = Real(1.0);
          Real P    = Real(1.0);
  
          if (n_j > 0)
          {
            if (!near_face_j[j])
            {
              // regular evaluation
              opow = std::pow(one_minus[j], static_cast<Real>(n_j));
  
              P = detail::BasisClassic1D<Real>::eval_n(
                    n_j,
                    a_j,
                    b_j,
                    t[j]
                  );
  
              F_j = opow * P;
            }
            else
            {
              // limit formula: (1-s)^n P_n(...) -> C*(2 x_j)^n
              Real nR       = static_cast<Real>(n_j);
              Real two_n_ab = Real(2.0) * nR + a_j + b_j;
  
              double lg_num  = std::lgamma(
                                 static_cast<double>(two_n_ab + Real(1.0))
                               );
              double lg_den1 = std::lgamma(
                                 static_cast<double>(nR + Real(1.0))
                               );
              double lg_den2 = std::lgamma(
                                 static_cast<double>(nR + a_j + b_j + Real(1.0))
                               );
              Real C = static_cast<Real>(
                         std::exp(lg_num - lg_den1 - lg_den2)
                       ) / std::pow(Real(2.0), nR);
  
              Real poly = std::pow(Real(2.0) * xloc[j], nR);
              F_j = C * poly;
  
              // not used in FD path, but keep sane
              opow = Real(1.0);
              P    = Real(1.0);
            }
          }
  
          F[j]         = F_j;
          omega_pow[j] = opow;
          Pnj[j]       = P;
  
          val *= F_j;
        } // j
  
        V[p + m * ld_V] = val;
  
        // --- analytic gradient path ---
        if (!dV || near_face_point)
        {
          continue;  // we will handle FD patch later for near_face_point
        }
  
        // product rule: build prefix/suffix for F[j]
        Real pre[D];
        Real suf[D];
  
        pre[0] = Real(1.0);
        for (int j = 1; j < D; ++j)
        {
          pre[j] = pre[j - 1] * F[j - 1];
        }
  
        suf[D - 1] = Real(1.0);
        for (int j = D - 2; j >= 0; --j)
        {
          suf[j] = suf[j + 1] * F[j + 1];
        }
  
        Real grad[D];
        for (int ell = 0; ell < D; ++ell)
        {
          grad[ell] = Real(0.0);
        }
  
        // d/dx_ell P_m = inv_h[m] * sum_j [ dF_j/dx_ell * prod_{r != j} F_r ]
        for (int j = 0; j < D; ++j)
        {
          int n_j = arow[j];
          if (n_j <= 0)
          {
            continue;
          }
  
          int tail_j = trow[j];
  
          Real omega = one_minus[j];
          Real xj    = xloc[j];
  
          Real a_j = Real(2.0) * static_cast<Real>(tail_j)
                   + Ktail[j]
                   + Real(0.5) * static_cast<Real>(D - j - 2);
  
          Real b_j = kappa[j] - Real(0.5);
  
          Real P = Pnj[j];
  
          Real dPdt = Real(0.0);
          if (n_j > 0)
          {
            Real Pn1 = detail::BasisClassic1D<Real>::eval_n(
                         n_j - 1,
                         a_j + Real(1.0),
                         b_j + Real(1.0),
                         t[j]
                       );
            Real factor = Real(0.5) *
                          (static_cast<Real>(n_j) + a_j + b_j + Real(1.0));
            dPdt = factor * Pn1;
          }
  
          Real omega_p       = omega_pow[j];
          Real prod_except_j = pre[j] * suf[j];
  
          for (int ell = 0; ell < D; ++ell)
          {
            // d omega_j / d x_ell
            Real domega = Real(0.0);
            if (ell < j)
            {
              domega = Real(-1.0);
            }
  
            Real d_omega_p = Real(0.0);
            if (domega != Real(0.0))
            {
              d_omega_p = static_cast<Real>(n_j) *
                          (omega_p / omega) * domega;
            }
  
            // dt_j / d x_ell
            Real dt_dx = Real(0.0);
            if (ell == j)
            {
              dt_dx = Real(2.0) / omega;
            }
            else if (ell < j)
            {
              dt_dx = Real(2.0) * xj / (omega * omega);
            }
  
            Real dJdx = dPdt * dt_dx;
            Real dFdx = d_omega_p * P + omega_p * dJdx;
  
            grad[ell] += inv_h[m] * dFdx * prod_except_j;
          }
        } // j
  
        Real* g_pm = dV + ((p + m * ld_V) * D);
        for (int ell = 0; ell < D; ++ell)
        {
          g_pm[ell] = grad[ell];
        }
      } // m
  
      // --- finite-difference gradient patch for near-face points ---
      if (dV && near_face_point)
      {
        const Real h = Real(1e-6);
        const Real two_h = Real(2.0) * h;
        const Real inv_2h = Real(1.0) / (Real(2.0) * h);
  
        Real sumx = Real(0.0);
        for (int j = 0; j < D; ++j)
        {
          sumx += xloc[j];
        }
  
        // allocate temporary arrays for unpatched evaluations
        Real* Vp1 = (Real*)std::malloc(sizeof(Real) * (size_t)M);
        Real* Vp2 = (Real*)std::malloc(sizeof(Real) * (size_t)M);
        Real* Vm1 = (Real*)std::malloc(sizeof(Real) * (size_t)M);
        Real* Vm2 = (Real*)std::malloc(sizeof(Real) * (size_t)M);
  
        if (!Vp1 || !Vp2 || !Vm1 || !Vm2)
        {
          if (Vp1) std::free(Vp1);
          if (Vp2) std::free(Vp2);
          if (Vm1) std::free(Vm1);
          if (Vm2) std::free(Vm2);
          continue; // out-of-memory; leave gradients as zero / garbage
        }
  
        // f0[m] = V[p + m*ld_V] already computed with patched formula
        for (int ell = 0; ell < D; ++ell)
        {
          // choose one-sided direction that stays inside simplex
          bool forward_ok  = true;
          bool backward_ok = true;
  
          // forward: x_ell + 2h, sumx + 2h
          if (xloc[ell] + two_h > Real(1.0) + eps_tol_over)
          {
            forward_ok = false;
          }
          if (sumx + two_h > Real(1.0) + eps_tol_over)
          {
            forward_ok = false;
          }
  
          // backward: x_ell - 2h, sumx - 2h
          if (xloc[ell] - two_h < Real(0.0) - eps_tol_over)
          {
            backward_ok = false;
          }
          // sumx - 2h is always <= 1, so no check needed for the inequality.
  
          bool use_forward = forward_ok;
          bool use_backward = (!forward_ok && backward_ok);
  
          if (!use_forward && !use_backward)
          {
            // cannot move in this direction safely; set derivative to 0
            for (int m = 0; m < M; ++m)
            {
              Real* g_pm = dV + ((p + m * ld_V) * D);
              g_pm[ell] = Real(0.0);
            }
            continue;
          }
  
          if (use_forward)
          {
            Real x1[D];
            Real x2[D];
            for (int j = 0; j < D; ++j)
            {
              x1[j] = xloc[j];
              x2[j] = xloc[j];
            }
            x1[ell] += h;
            x2[ell] += two_h;
  
            eval_point_value_unpatched(x1,
                                       kappa,
                                       n,
                                       alpha_table,
                                       tail_deg,
                                       inv_h,
                                       Ktail,
                                       Vp1);
  
            eval_point_value_unpatched(x2,
                                       kappa,
                                       n,
                                       alpha_table,
                                       tail_deg,
                                       inv_h,
                                       Ktail,
                                       Vp2);
  
            for (int m = 0; m < M; ++m)
            {
              Real f0 = V[p + m * ld_V];
              Real f1 = Vp1[m];
              Real f2 = Vp2[m];
  
              Real deriv = (-Real(3.0) * f0 + Real(4.0) * f1 - f2) * inv_2h;
              Real* g_pm = dV + ((p + m * ld_V) * D);
              g_pm[ell] = deriv;
            }
          }
          else if (use_backward)
          {
            Real x1[D];
            Real x2[D];
            for (int j = 0; j < D; ++j)
            {
              x1[j] = xloc[j];
              x2[j] = xloc[j];
            }
            x1[ell] -= h;
            x2[ell] -= two_h;
  
            eval_point_value_unpatched(x1,
                                       kappa,
                                       n,
                                       alpha_table,
                                       tail_deg,
                                       inv_h,
                                       Ktail,
                                       Vm1);
  
            eval_point_value_unpatched(x2,
                                       kappa,
                                       n,
                                       alpha_table,
                                       tail_deg,
                                       inv_h,
                                       Ktail,
                                       Vm2);
  
            for (int m = 0; m < M; ++m)
            {
              Real f0 = V[p + m * ld_V];
              Real f1 = Vm1[m];
              Real f2 = Vm2[m];
  
              Real deriv = (Real(3.0) * f0 - Real(4.0) * f1 + f2) * inv_2h;
              Real* g_pm = dV + ((p + m * ld_V) * D);
              g_pm[ell] = deriv;
            }
          }
        } // ell
  
        std::free(Vp1);
        std::free(Vp2);
        std::free(Vm1);
        std::free(Vm2);
      } // if (dV && near_face_point)
  
    } // p
  }

  static void build_structures(int n,
                               const Real* kappa,
                               std::vector<int>& alpha,
                               std::vector<int>& tail,
                               std::vector<Real>& invh)
  {
    if (!kappa)
    {
      throw std::invalid_argument("Basis::build_structures: null kappa");
    }
  
    const int M = dim_Pi(n);
  
    alpha.assign((std::size_t)M * D, 0);
    tail.assign((std::size_t)M * D, 0);
    invh.assign((std::size_t)M, (Real)0);
  
    build_alpha_table(n, alpha.data());
    build_tail_deg(n, alpha.data(), tail.data());
  
    for (int m = 0; m < M; ++m)
    {
      invh[(std::size_t)m] =
        inv_h_alpha(alpha.data() + (std::size_t)m * D, kappa);
    }
  }
  
  static void derivative_kappa_shift(const Real* kappa,
                                     int axis,
                                     Real* kappa_out)
  {
    if (!kappa || !kappa_out)
    {
      throw std::invalid_argument("Basis::derivative_kappa_shift: null pointer");
    }
    assert(0 <= axis && axis < D);
  
    for (int i = 0; i <= D; ++i)
    {
      kappa_out[i] = kappa[i];
    }
  
    // Storage convention: kappa[D] is the barycentric vertex-0 parameter.
    kappa_out[axis] += (Real)1;
    kappa_out[D] += (Real)1;
  }

  static inline void build_structures(const double* kappa,
                                      int n,
                                      int* alpha_table,
                                      int* tail_deg,
                                      double* inv_h)
  {
    int M = dim_Pi(n);
  
    build_alpha_table(n, alpha_table);
    build_tail_deg(n, alpha_table, tail_deg);
  
    for (int m = 0; m < M; ++m)
    {
      const int* alpha = alpha_table + m * D;
      inv_h[m] = inv_h_alpha(alpha, kappa);
    }
  }


private:
  // Recursive helper used only by build_alpha_table.
  static void fill_degree_rec(int total,
                              int coord,
                              int* alpha,
                              int* alpha_table,
                              int& m)
  {
    if (coord == D - 1)
    {
      alpha[coord] = total;
      int offset = m * D;
      for (int j = 0; j < D; ++j)
      {
        alpha_table[offset + j] = alpha[j];
      }
      ++m;
      return;
    }

    for (int a = total; a >= 0; --a)
    {
      alpha[coord] = a;
      fill_degree_rec(total - a, coord + 1, alpha, alpha_table, m);
    }
  }

  // Single-point evaluator using the original product formula
  // (no singular-face patch). Safe as long as 1 - |x^{j-1}| is not tiny.
  static void eval_point_value_unpatched(const Real* xloc,
                                         const Real* kappa,
                                         int n,
                                         const int* alpha_table,
                                         const int* tail_deg,
                                         const Real* inv_h,
                                         const Real* Ktail,
                                         Real* V_out)
  {
    int M = dim_Pi(n);

    Real one_minus[D];
    Real t[D];

    Real s = Real(0.0);
    for (int j = 0; j < D; ++j)
    {
      Real xpj = xloc[j];

      Real om = Real(1.0) - s;
      if (om <= Real(0.0))
      {
        om = Real(1.0e-14);
      }

      one_minus[j] = om;
      t[j]         = (Real(2.0) * xpj / om) - Real(1.0);
      s           += xpj;
    }

    for (int m = 0; m < M; ++m)
    {
      const int* arow = alpha_table + m * D;
      const int* trow = tail_deg    + m * D;

      Real val = inv_h[m];

      for (int j = 0; j < D; ++j)
      {
        int n_j    = arow[j];
        int tail_j = trow[j];

        Real a_j = Real(2.0) * static_cast<Real>(tail_j)
                 + Ktail[j]
                 + Real(0.5) * static_cast<Real>(D - j - 2);

        Real b_j = kappa[j] - Real(0.5);

        if (n_j > 0)
        {
          Real opow = std::pow(one_minus[j], static_cast<Real>(n_j));

          Real P = detail::BasisClassic1D<Real>::eval_n(
                     n_j,
                     a_j,
                     b_j,
                     t[j]
                   );

          val *= (opow * P);
        }
      }

      V_out[m] = val;
    }
  }


}; // Basis


template<int D, class Real>
struct BasisEvalView
{
  static_assert(D >= 1, "BasisEvalView requires D >= 1");

  int n = -1;
  int M = 0;
  int nq = 0;

  const Real* kappa = nullptr;

  // Point-major quadrature coordinates:
  //   X[q * x_point_stride + a * x_coord_stride]
  const Real* X = nullptr;
  const Real* W = nullptr;
  int x_point_stride = D;
  int x_coord_stride = 1;

  // alpha/tail are row-major M x D; invh length M.
  const int* alpha = nullptr;
  const int* tail = nullptr;
  const Real* invh = nullptr;

  // Column-major nq x M basis values:
  //   V[q + ldV * m]
  const Real* V = nullptr;
  int ldV = 0;

  BasisEvalView() = default;

  BasisEvalView(int n_,
                int nq_,
                const Real* kappa_,
                const Real* X_,
                const Real* W_,
                const int* alpha_,
                const int* tail_,
                const Real* invh_,
                const Real* V_,
                int ldV_,
                int x_point_stride_ = D,
                int x_coord_stride_ = 1)
    : n(n_),
      M(Basis<D,Real>::dim_Pi(n_)),
      nq(nq_),
      kappa(kappa_),
      X(X_),
      W(W_),
      x_point_stride(x_point_stride_),
      x_coord_stride(x_coord_stride_),
      alpha(alpha_),
      tail(tail_),
      invh(invh_),
      V(V_),
      ldV(ldV_)
  {}

  bool valid() const
  {
    return n >= 0 &&
           M > 0 &&
           nq > 0 &&
           kappa != nullptr &&
           X != nullptr &&
           W != nullptr &&
           alpha != nullptr &&
           tail != nullptr &&
           invh != nullptr &&
           V != nullptr &&
           ldV >= nq;
  }

  const Real* point_ptr(int q) const
  {
    return X + (std::size_t)q * x_point_stride;
  }

  Real x(int q, int a) const
  {
    return X[(std::size_t)q * x_point_stride +
             (std::size_t)a * x_coord_stride];
  }

  Real weight(int q) const
  {
    return W[q];
  }

  Real value(int q, int m) const
  {
    return V[(std::size_t)q + (std::size_t)ldV * m];
  }

  const int* alpha_ptr(int m) const
  {
    return alpha + (std::size_t)m * D;
  }

  const int* tail_ptr(int m) const
  {
    return tail + (std::size_t)m * D;
  }
}; // BasisEvalView


} // namespace jsimplex

#endif // JBASIS_H
