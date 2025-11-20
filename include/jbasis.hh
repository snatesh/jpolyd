#ifndef JBASIS_H
#define JBASIS_H

#include <cmath>
#include <jdetail.hh>
#include <omp.h>

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

      Real t5 = Real(2.0)
                * (aj + kappa_j + static_cast<Real>(alpha_j))
                + Real(1.0);
      Real t6 = Real(2.0)
                * (aj + kappa_j + static_cast<Real>(2 * alpha_j))
                + Real(1.0);

      contrib += static_cast<Real>(
        std::log(static_cast<double>(t5))
        - std::log(static_cast<double>(t6))
      );

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
                       int ld_V)
  {
    if (!X || !kappa || !alpha_table || !tail_deg || !inv_h || !V)
    {
      return;
    }

    int M = dim_Pi(n);

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
  
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < npts; ++p)
    {
      Real prefix_sum[D];
      Real one_minus[D];
      Real t[D];

      Real s = Real(0.0);
      for (int j = 0; j < D; ++j)
      {
        prefix_sum[j] = s;

        Real xpj = X[p * ld_point + j * ld_dim];
        Real om = Real(1.0) - s;
        if (om <= Real(0.0))
        {
          om = Real(1.0e-30);
        }

        one_minus[j] = om;
        t[j]         = (Real(2.0) * xpj / om) - Real(1.0);
        s           += xpj;
      }

      for (int m = 0; m < M; ++m)
      {
        Real val = inv_h[m];

        const int* arow = alpha_table + m * D;
        const int* trow = tail_deg    + m * D;

        for (int j = 0; j < D; ++j)
        {
          int n_j = arow[j];

          if (n_j > 0)
          {
            val *= std::pow(one_minus[j], static_cast<Real>(n_j));
          }

          int tail_j = trow[j];

          Real a_j = Real(2.0) * static_cast<Real>(tail_j)
                   + Ktail[j]
                   + Real(0.5) * static_cast<Real>(D - j - 2);

          Real b_j = kappa[j] - Real(0.5);

          Real Pnj = Real(1.0);
          if (n_j > 0)
          {
            Pnj = detail::BasisClassic1D<Real>::eval_n(
                    n_j,
                    a_j,
                    b_j,
                    t[j]
                  );
          }

          val *= Pnj;
        }

        V[p + m * ld_V] = val;
      }
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
}; // Basis

} // namespace jsimplex

#endif // JBASIS_H
