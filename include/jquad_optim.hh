#ifndef JQUAD_OPTIM_H
#define JQUAD_OPTIM_H

#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>

#include <jdetail.hh>
#include <jmat.hh>
#include <jbasis.hh>
#include <timer.hh>
#include <nlopt.h>


namespace jsimplex
{

/* ================= Options for quadrature optimization (D ≥ 1) ================ */

template<int D, class Real>
struct QuadOptimOptions
{
  int  max_nlopt_eval;
  int  max_gn_iter;
  Real fd_step;
  Real gn_step;
  Real tol;
  Real tol_up;
  bool verbose;

  QuadOptimOptions()
  : max_nlopt_eval(2000),
    max_gn_iter(200),
    fd_step(static_cast<Real>(1e-6)),
    gn_step(static_cast<Real>(1)),
    tol(static_cast<Real>(1e-15)),
    tol_up(static_cast<Real>(1e3)),
    verbose(false)
  {}
};

/* ================= Internal data structure for D ≥ 2 ========================== */

template<int D, class Real>
struct QuadProblemData
{
  int  node_deg;
  int  m_basis;
  int  N;        // dim Pi_node_deg^D
  int  M;        // dim Pi_m_basis^D
  int  nvar;     // (D+1)*N

  Real kappa[D+1];

  // Basis metadata
  int*  alpha_table;  // [M x D]
  int*  tail_deg;     // [M x D]
  Real* inv_h;        // [M]

  // Workspaces in Real precision
  Real* V;            // [N x M], column-major in p: V[p + m*N]
  Real* dV;           // [N x M x D], layout ((p + m*N)*D + ell)
  Real* F;            // [M]
  Real* e1;           // [M], target moments (usually [1,0,0,...])

  Real* z_real;       // [nvar]
  Real* grad_real;    // [nvar]
  Real* X_real;       // [N*D]
  Real* w_real;       // [N]

  // Gauss-Newton scratch
  Real* J;            // [M x nvar], column-major: J[col*M + row]
  Real* S;            // singular values for gelsd, size min(M,nvar)
  Real* B;            // RHS for gelsd, at least max(M,nvar)

  // NLopt-facing double buffer
  double* z_double;   // [nvar]

  Real  f_real;       // scalar objective value in Real

  bool verbose;
  int obj_eval_count;

  // Timing / diagnostics
  timer obj_timer;
  timer constr_timer;

  double time_obj;      // accumulated time in objective_cb
  double time_constr;   // accumulated time in ineq_mconstraint_cb
  double time_nlopt;    // wall time spent inside nlopt_optimize
  double time_gn;       // wall time spent inside gauss_newton

  int    count_obj;     // number of objective evaluations
  int    count_constr;  // number of constraint evaluations


  QuadProblemData()
  : node_deg(0),
    m_basis(0),
    N(0),
    M(0),
    nvar(0),
    alpha_table(nullptr),
    tail_deg(nullptr),
    inv_h(nullptr),
    V(nullptr),
    dV(nullptr),
    F(nullptr),
    e1(nullptr),
    z_real(nullptr),
    grad_real(nullptr),
    X_real(nullptr),
    w_real(nullptr),
    J(nullptr),
    S(nullptr),
    B(nullptr),
    z_double(nullptr),
    f_real(static_cast<Real>(0)),
    verbose(false),
    obj_eval_count(0),
    time_obj(0.0),
    time_constr(0.0),
    time_nlopt(0.0),
    time_gn(0.0),
    count_obj(0),
    count_constr(0)
  {}

  void init(int node_deg_in,
            int m_basis_in,
            const Real* kappa_in,
            bool want_dV,
            bool want_J,
            bool verbose_in)
  {
    verbose   = verbose_in;
    node_deg  = node_deg_in;
    m_basis   = m_basis_in;

    for (int j = 0; j < D+1; ++j)
    {
      kappa[j] = kappa_in[j];
    }

    N    = Basis<D,Real>::dim_Pi(node_deg);
    M    = Basis<D,Real>::dim_Pi(m_basis);
    nvar = (D + 1) * N;

    // Basis metadata
    alpha_table = static_cast<int*>(
      std::calloc(static_cast<std::size_t>(M * D), sizeof(int))
    );
    tail_deg = static_cast<int*>(
      std::calloc(static_cast<std::size_t>(M * D), sizeof(int))
    );
    inv_h = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(M), sizeof(Real))
    );

    Basis<D,Real>::build_structures(
      kappa,
      m_basis,
      alpha_table,
      tail_deg,
      inv_h
    );

    // Work arrays
    V = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(N * M), sizeof(Real))
    );
    if (want_dV)
    {
      dV = static_cast<Real*>(
        std::calloc(static_cast<std::size_t>(N * M * D), sizeof(Real))
      );
    }
    else
    {
      dV = nullptr;
    }

    F = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(M), sizeof(Real))
    );
    e1 = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(M), sizeof(Real))
    );

    z_real = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(nvar), sizeof(Real))
    );
    grad_real = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(nvar), sizeof(Real))
    );
    X_real = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(N * D), sizeof(Real))
    );
    w_real = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(N), sizeof(Real))
    );

    if (want_J)
    {
      J = static_cast<Real*>(
        std::calloc(static_cast<std::size_t>(M * nvar), sizeof(Real))
      );
      int mn = (M < nvar ? M : nvar);
      S = static_cast<Real*>(
        std::calloc(static_cast<std::size_t>(mn), sizeof(Real))
      );
      int bdim = (M > nvar ? M : nvar);
      B = static_cast<Real*>(
        std::calloc(static_cast<std::size_t>(bdim), sizeof(Real))
      );
    }
    else
    {
      J = nullptr;
      S = nullptr;
      B = nullptr;
    }

    z_double = static_cast<double*>(
      std::calloc(static_cast<std::size_t>(nvar), sizeof(double))
    );

    // e1 = [1, 0, 0, ...]
    if (e1)
    {
      e1[0] = static_cast<Real>(1);
    }

    f_real = static_cast<Real>(0);
  }

  void release()
  {
    std::free(alpha_table);
    std::free(tail_deg);
    std::free(inv_h);
    std::free(V);
    std::free(dV);
    std::free(F);
    std::free(e1);
    std::free(z_real);
    std::free(grad_real);
    std::free(X_real);
    std::free(w_real);
    std::free(J);
    std::free(S);
    std::free(B);
    std::free(z_double);

    alpha_table = nullptr;
    tail_deg    = nullptr;
    inv_h       = nullptr;
    V           = nullptr;
    dV          = nullptr;
    F           = nullptr;
    e1          = nullptr;
    z_real      = nullptr;
    grad_real   = nullptr;
    X_real      = nullptr;
    w_real      = nullptr;
    J           = nullptr;
    S           = nullptr;
    B           = nullptr;
    z_double    = nullptr;
    f_real      = static_cast<Real>(0);
  }
};

/* Helper: unpack z_real into X_real and w_real.
   z layout: [ x_{0,0},...,x_{0,D-1}, x_{1,0},..., x_{N-1,D-1}, w_0,...,w_{N-1} ] */
template<int D, class Real>
static inline void unpack_z(const Real* z,
                            int N,
                            Real* X,
                            Real* w)
{
  for (int p = 0; p < N; ++p)
  {
    for (int j = 0; j < D; ++j)
    {
      X[p*D + j] = z[p*D + j];
    }
  }

  for (int p = 0; p < N; ++p)
  {
    w[p] = z[N*D + p];
  }
}

/* Evaluate basis, residual F, and optionally gradient of f at z_real.
   f(z) = 0.5 * ||F||^2, F = V^T w - e1. */
template<int D, class Real>
static void eval_f_and_grad(QuadProblemData<D,Real>& data,
                            const Real* z_real,
                            bool want_grad)
{
  const int N    = data.N;
  const int M    = data.M;
  const int nvar = data.nvar;

  // unpack z into X_real (N x D) and w_real (N)
  unpack_z<D,Real>(z_real, N, data.X_real, data.w_real);

  // basis + (optionally) ∂P/∂x
  Basis<D,Real>::eval_all(
    data.X_real,
    D, 1,         // ld_point, ld_dim
    N,
    data.kappa,
    data.m_basis,
    data.alpha_table,
    data.tail_deg,
    data.inv_h,
    data.V,
    N,
    want_grad ? data.dV : nullptr
  );

  Real* F = data.F;

  // Compute F[m] = sum_p V[p + m*N] * w[p] - e1[m]
  // and f = 0.5 * sum_m F[m]^2 in one parallel pass.
  Real fval = static_cast<Real>(0);

  //#pragma omp parallel for reduction(+:fval) schedule(static)
  for (int m = 0; m < M; ++m)
  {
    Real sum = static_cast<Real>(0);
    for (int p = 0; p < N; ++p)
    {
      sum += data.V[p + m*N] * data.w_real[p];
    }
    Real Fm = sum - data.e1[m];
    F[m] = Fm;
    fval += Fm * Fm;
  }

  data.f_real = static_cast<Real>(0.5) * fval;

  if (!want_grad)
  {
    return;
  }

  Real* g = data.grad_real;

  // Initialize gradient to zero
  for (int i = 0; i < nvar; ++i)
  {
    g[i] = static_cast<Real>(0);
  }

  // Node part:
  // ∂f/∂x_{p,ell} = w_p * sum_m F_m * (∂P_m/∂x_ell)(x_p)
  // Each (p,ell) writes to a unique g[p*D + ell] ⇒ parallel over p is safe.
  //#pragma omp parallel for schedule(static)
  for (int p = 0; p < N; ++p)
  {
    for (int ell = 0; ell < D; ++ell)
    {
      Real sum = static_cast<Real>(0);
      const Real wp = data.w_real[p];
      for (int m = 0; m < M; ++m)
      {
        Real dP = data.dV[((p + m*N) * D) + ell];
        sum += data.F[m] * (wp * dP);
      }
      g[p*D + ell] = sum;
    }
  }

  // Weight part:
  // ∂f/∂w_p = sum_m F_m * P_m(x_p) = (row p of V) · F
  // Again, each p writes a unique index g[N*D + p] ⇒ parallel over p is safe.
  //#pragma omp parallel for schedule(static)
  for (int p = 0; p < N; ++p)
  {
    Real sum = static_cast<Real>(0);
    for (int m = 0; m < M; ++m)
    {
      sum += data.F[m] * data.V[p + m*N];
    }
    g[N*D + p] = sum;
  }
}
//template<int D, class Real>
//static void eval_f_and_grad(QuadProblemData<D,Real>& data,
//                            const Real* z_real,
//                            bool want_grad)
//{
//  const int N    = data.N;
//  const int M    = data.M;
//  const int nvar = data.nvar;
//
//  unpack_z<D,Real>(z_real, N, data.X_real, data.w_real);
//
//  Basis<D,Real>::eval_all(
//    data.X_real,
//    D, 1,         // ld_point, ld_dim
//    N,
//    data.kappa,
//    data.m_basis,
//    data.alpha_table,
//    data.tail_deg,
//    data.inv_h,
//    data.V,
//    N,
//    want_grad ? data.dV : nullptr
//  );
//
//  Real* F = data.F;
//  for (int m = 0; m < M; ++m)
//  {
//    Real sum = static_cast<Real>(0);
//    for (int p = 0; p < N; ++p)
//    {
//      sum += data.V[p + m*N] * data.w_real[p];
//    }
//    F[m] = sum - data.e1[m];
//  }
//
//  Real fval = static_cast<Real>(0);
//  for (int m = 0; m < M; ++m)
//  {
//    fval += F[m] * F[m];
//  }
//  data.f_real = static_cast<Real>(0.5) * fval;
//
//  if (!want_grad)
//  {
//    return;
//  }
//
//  Real* g = data.grad_real;
//  for (int i = 0; i < nvar; ++i)
//  {
//    g[i] = static_cast<Real>(0);
//  }
//
//  // Node part: ∂f/∂x_{p,ell} = w_p * sum_m F_m * (∂P_m/∂x_ell)(x_p)
//  for (int p = 0; p < N; ++p)
//  {
//    for (int ell = 0; ell < D; ++ell)
//    {
//      Real sum = static_cast<Real>(0);
//      for (int m = 0; m < M; ++m)
//      {
//        Real dP = data.dV[((p + m*N) * D) + ell];
//        sum += data.F[m] * (data.w_real[p] * dP);
//      }
//      g[p*D + ell] = sum;
//    }
//  }
//
//  // Weight part: ∂f/∂w_p = sum_m F_m * P_m(x_p) = (V[p,:] · F)
//  for (int p = 0; p < N; ++p)
//  {
//    Real sum = static_cast<Real>(0);
//    for (int m = 0; m < M; ++m)
//    {
//      sum += data.F[m] * data.V[p + m*N];
//    }
//    g[N*D + p] = sum;
//  }
//}

/* NLopt objective callback: double interface, Real internal */
template<int D, class Real>
static double objective_cb(unsigned n,
                           const double* x,
                           double* grad,
                           void* f_data)
{
  QuadProblemData<D,Real>* data =
    reinterpret_cast<QuadProblemData<D,Real>*>(f_data);

  assert(static_cast<int>(n) == data->nvar);
  // start timing
  data->obj_timer.tic();

  for (unsigned i = 0; i < n; ++i)
  {
    data->z_real[i] = static_cast<Real>(x[i]);
  }

  bool want_grad = (grad != nullptr);
  eval_f_and_grad<D,Real>(*data, data->z_real, want_grad);
  if (!std::isfinite(data->f_real)) {
    std::cerr << "[objective_cb] f_real is NaN/Inf at eval "
              << data->obj_eval_count << "\n";
  
    for (int i = 0; i < data->nvar; ++i) {
      if (!std::isfinite(data->z_real[i])) {
        std::cerr << "  z[" << i << "] = " << data->z_real[i] << " (bad)\n";
      }
    }
  
    for (int m = 0; m < data->M; ++m) {
      if (!std::isfinite(data->F[m])) {
        std::cerr << "  F[" << m << "] = " << data->F[m] << " (bad)\n";
        break;
      }
    }
    if (data->obj_eval_count == 1)
    {
      // ---- Dump z1 ----
      {
        std::ofstream out("dump_z1.txt");
        out << std::setprecision(17) << std::scientific;

        for (int i = 0; i < data->nvar; ++i) {
          out << data->z_real[i] << "\n";
        }
        out.close();
      }
      std::cerr << "Wrote dump_z1.txt\n";

      // ---- Dump V ----
      {
        std::ofstream out("dump_V.txt");
        out << std::setprecision(17) << std::scientific;

        int N = data->N;
        int M = data->M;
        Real* V = data->V;    // layout V[p + m*N]

        for (int m = 0; m < M; ++m) {
          for (int p = 0; p < N; ++p) {
            out << V[p + m*N] << " ";
          }
          out << "\n";
        }
        out.close();
      }
      std::cerr << "Wrote dump_V.txt\n";

      // ---- Dump dV ----
      if (want_grad && data->dV != nullptr) {
        std::ofstream out("dump_dV.txt");
        out << std::setprecision(17) << std::scientific;

        int N = data->N;
        int M = data->M;
        int Ddim = D;
        Real* dV = data->dV;  // layout ((p + m*N)*D + ell)

        for (int m = 0; m < M; ++m) {
          for (int p = 0; p < N; ++p) {
            for (int ell = 0; ell < Ddim; ++ell) {
              Real dv = dV[(p + m*N)*Ddim + ell];
              out << dv << " ";
            }
            out << "\n";
          }
        }
        out.close();
        std::cerr << "Wrote dump_dV.txt\n";
      }
    } 
    // optional: abort to capture a reproducible state
    // std::abort();
  }

  data->obj_eval_count += 1;
  if (data->verbose && (data->obj_eval_count % 100 == 0))
  {
    std::cout << "[objective] eval " << data->obj_eval_count
              << ", f(z) = " << static_cast<double>(data->f_real)
              << std::endl;
  }


  if (grad)
  {
    for (unsigned i = 0; i < n; ++i)
    {
      grad[i] = static_cast<double>(data->grad_real[i]);
    }
  }
  // stop timing and accumulate
  double dt = data->obj_timer.toc();
  data->time_obj += dt;
  data->count_obj += 1;


  return static_cast<double>(data->f_real);
}

/* NLopt inequality m-constraint:
   Constraints g_i(z) <= 0:
   1) Node non-negativity: g_{p,j} = -x_{p,j}      (N*D)
   2) Simplex: g_p^{simp} = sum_j x_{p,j} - 1      (N)
   3) Weight non-negativity: g_p^{w} = -w_p        (N)
   Total m = N*(D+2).
*/
template<int D, class Real>
static void ineq_mconstraint_cb([[maybe_unused]] unsigned m,
                                double* result,
                                unsigned n,
                                const double* x,
                                double* grad,
                                void* f_data)
{
  QuadProblemData<D,Real>* data =
    reinterpret_cast<QuadProblemData<D,Real>*>(f_data);
  data->constr_timer.tic();
  const int N    = data->N;
  [[maybe_unused]]
  const int nvar = data->nvar;

  assert(static_cast<int>(n) == nvar);
  assert(static_cast<int>(m) == N * (D + 2));

  // layout:
  //  idx = 0 .. N*D-1           : node nonnegativity
  //  idx = N*D .. N*(D+1)-1     : simplex sum-1
  //  idx = N*(D+1) .. N*(D+2)-1 : weight nonnegativity

  const int offset_node    = 0;
  const int offset_simplex = N * D;
  const int offset_weight  = N * (D + 1);

  // 1) Node non-negativity: g = -x_{p,j}
  for (int p = 0; p < N; ++p)
  {
    for (int j = 0; j < D; ++j)
    {
      int idx     = offset_node + p*D + j;
      int var_idx = p*D + j;

      result[idx] = -x[var_idx];

      if (grad)
      {
        // zero this row just once
        double* grow = grad + idx*n;
        std::memset(grow, 0, sizeof(double) * n);
        grow[var_idx] = -1.0;
      }
    }
  }

  // 2) Simplex: sum_j x_{p,j} - 1 <= 0
  for (int p = 0; p < N; ++p)
  {
    int idx = offset_simplex + p;

    double sum = 0.0;
    for (int j = 0; j < D; ++j)
    {
      int var_idx = p*D + j;
      sum += x[var_idx];
    }
    result[idx] = sum - 1.0;

    if (grad)
    {
      double* grow = grad + idx*n;
      std::memset(grow, 0, sizeof(double) * n);
      for (int j = 0; j < D; ++j)
      {
        int var_idx = p*D + j;
        grow[var_idx] = 1.0;
      }
    }
  }

  // 3) Weight non-negativity: g = -w_p
  for (int p = 0; p < N; ++p)
  {
    int idx     = offset_weight + p;
    int var_idx = D * N + p;

    result[idx] = -x[var_idx];

    if (grad)
    {
      double* grow = grad + idx*n;
      std::memset(grow, 0, sizeof(double) * n);
      grow[var_idx] = -1.0;
    }
  }
  double dt = data->constr_timer.toc();
  data->time_constr += dt;
  data->count_constr += 1;
}

// NLOPT equality constraint sum(w)=1
// tends to prevent overflow in F
template<int D, class Real>
static void eq_constraint_cb([[maybe_unused]] unsigned m,
                             double* result,
                             unsigned n,
                             const double* x,
                             double* grad,
                             void* f_data)
{
  QuadProblemData<D,Real>* data =
    reinterpret_cast<QuadProblemData<D,Real>*>(f_data);
  const int N    = data->N;
  const int nvar = data->nvar;
  (void)nvar;

  assert(m == 1);
  assert(static_cast<int>(n) == nvar);

  // z = [X_flat, w], w starts at index N*D
  const int offset_w = N * D;

  // h(z) = sum_p w_p - 1
  double sum = 0.0;
  for (int p = 0; p < N; ++p)
    sum += x[offset_w + p];

  result[0] = sum - 1.0;

  if (grad)
  {
    std::memset(grad, 0, sizeof(double)*n);
    for (int p = 0; p < N; ++p)
      grad[0*n + (offset_w + p)] = 1.0;  // row 0, col offset_w+p
  }
}

template<class Real>
static inline void project_simplex_leq1(Real* x, int D)
{
  // 1) Clip to x >= 0
  for (int j = 0; j < D; ++j)
    if (x[j] < Real(0)) x[j] = Real(0);

  // 2) Check if already sum(x) <= 1
  Real s = Real(0);
  for (int j = 0; j < D; ++j) s += x[j];
  if (s <= Real(1)) return;

  // 3) Project onto simplex sum(x)=1, x>=0
  //    Find threshold theta via sorting and cumulative sum

  // copy and sort descending
  std::vector<Real> u(D);
  for (int j = 0; j < D; ++j) u[j] = x[j];
  std::sort(u.begin(), u.end(), std::greater<Real>());

  std::vector<Real> cssv(D);
  cssv[0] = u[0];
  for (int i = 1; i < D; ++i)
    cssv[i] = cssv[i-1] + u[i];

  int rho = -1;
  for (int i = 0; i < D; ++i)
  {
    Real lhs = u[i] * Real(i+1);
    Real rhs = cssv[i] - Real(1);

    if (lhs > rhs) rho = i;
  }

  if (rho < 0) rho = 0;

  Real theta = (cssv[rho] - Real(1)) / Real(rho + 1);

  for (int j = 0; j < D; ++j)
  {
    Real v = x[j] - theta;
    x[j] = (v > Real(0) ? v : Real(0));
  }
}

template<int D, class Real>
static inline void project_z(Real* z, int N)
{
  // z = [X_flat, w]

  Real* X = z;              // length N*D
  Real* w = z + N * D;      // length N

  // Project each node
  for (int p = 0; p < N; ++p)
  {
    Real* xp = X + p * D;
    project_simplex_leq1<Real>(xp, D);
  }

  // Clip weights to non-negative
  for (int p = 0; p < N; ++p)
    if (w[p] < Real(0)) w[p] = Real(0);
}

/* Gauss-Newton refinement on Real z (D ≥ 2).
   Uses analytic Jacobian J (M x nvar) and solves J delta ≈ F in LS sense,
   then updates z <- z - alpha * delta.
*/
template<int D, class Real>
static void gauss_newton(QuadProblemData<D,Real>& data,
                         Real* z,
                         const QuadOptimOptions<D,Real>& opts)
{
  const int N    = data.N;
  const int M    = data.M;
  const int nvar = data.nvar;

  Real* J = data.J;
  Real* B = data.B;
  Real* S = data.S;

  Real pk  = static_cast<Real>(0);
  int iter = 0;

  while (true)
  {
    eval_f_and_grad<D,Real>(data, z, true);

    pk = static_cast<Real>(0);
    for (int m = 0; m < M; ++m)
    {
      pk += data.F[m] * data.F[m];
    }
    pk = std::sqrt(pk);

    if (data.verbose)
    {
      std::cout << "GN iter " << iter << "  ||F|| = " << pk << std::endl;
    }

    if (pk < opts.tol || pk > opts.tol_up || iter >= opts.max_gn_iter)
    {
      break;
    }

    // Build Jacobian J (column-major, M x nvar)
    // Node columns: J[:, p*D+ell] = w_p * dV[p,:,ell]
    //#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p)
    {
      for (int ell = 0; ell < D; ++ell)
      {
        int col = p*D + ell;
        Real wp = data.w_real[p];
        for (int m = 0; m < M; ++m)
        {
          Real dP = data.dV[((p + m*N) * D) + ell];
          J[col*M + m] = wp * dP;
        }
      }
    }
    
    // Weight columns: J[:, N*D+p] = V[p,:]
    //#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p)
    {
      int col = N*D + p;
      for (int m = 0; m < M; ++m)
      {
        J[col*M + m] = data.V[p + m*N];
      }
    }

    // Solve J delta = F in LS sense using LapackGelsd<Real>
    //#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m)
    {
      B[m] = data.F[m];
    }

    lapack_int mA   = static_cast<lapack_int>(M);
    lapack_int nA   = static_cast<lapack_int>(nvar);
    lapack_int nrhs = 1;
    lapack_int lda  = mA;
    lapack_int ldb  = (mA > nA ? mA : nA);
    lapack_int rank = 0;
    Real rcond      = static_cast<Real>(-1);

    lapack_int info = detail::LapackGelsd<Real>::run(
      mA, nA, nrhs,
      J, lda,
      B, ldb,
      S, rcond, &rank
    );

    if (info != 0)
    {
      if (data.verbose)
      {
        std::cerr << "ERROR: LapackGelsd returned info = "
                  << info << std::endl;
      }
      break;
    }

    // After gelsd, B[0..nvar-1] contains the LS solution delta.
    Real alpha = opts.gn_step;
    for (int i = 0; i < nvar; ++i)
    {
      Real delta = B[i];
      z[i] = z[i] - alpha * delta;
    }
    project_z<D,Real>(z,N);

    if (data.verbose && (iter % 100 == 0))
    {
      std::cout << "GN iter " << iter
                << "  ||F|| = " << pk
                << std::endl;
    }


    ++iter;
  }
  if (data.verbose)
  {
    // Compute ||F|| after final iteration
    Real Fnorm = Real(0);
    for (int m = 0; m < data.M; ++m)
      Fnorm += data.F[m] * data.F[m];
    Fnorm = std::sqrt(Fnorm);
  
    // Summaries
    Real wsum = Real(0);
    Real wmin = data.w_real[0];
    Real wmax = data.w_real[0];
    for (int p = 0; p < data.N; ++p)
    {
      Real wp = data.w_real[p];
      wsum += wp;
      if (wp < wmin) wmin = wp;
      if (wp > wmax) wmax = wp;
    }
  
    std::cout << "\n[GN] diagnostics:\n";
    std::cout << "  ||Ihat - e1||_2 = " << static_cast<double>(Fnorm) << "\n";
    std::cout << "  sum(weights)     = " << static_cast<double>(wsum) << "\n";
    std::cout << "  min(weights)     = " << static_cast<double>(wmin) << "\n";
    std::cout << "  max(weights)     = " << static_cast<double>(wmax) << "\n";
    std::cout << std::endl;
  }
}

/* ===================== Core optimizer for D ≥ 2 ============================== */

template<int D, class Real>
struct QuadOptimizer
{
  static int optimize(int node_deg,
                      int m_basis,
                      const Real* kappa,
                      Real* z_io,
                      Real* V_opt,
                      const QuadOptimOptions<D,Real>& opts)
  {
    static_assert(D >= 2, "This primary template is intended for D >= 2.");

    if (!z_io || !kappa)
    {
      return -1;
    }

    QuadProblemData<D,Real> data;
    data.init(node_deg, m_basis, kappa, true, true, opts.verbose);
    if (opts.verbose)
    {
      std::cout << "\n=========================================\n";
      std::cout << "Quadrature Optimization Problem\n";
      std::cout << "-----------------------------------------\n";
      std::cout << "  D          = " << D << "\n";
      std::cout << "  node_deg   = " << node_deg << "\n";
      std::cout << "  N          = " << data.N << "   (dim Pi_" << node_deg << "^" << D << ")\n";
      std::cout << "  m_basis    = " << m_basis << "\n";
      std::cout << "  M          = " << data.M << "   (dim Pi_" << m_basis << "^" << D << ")\n";
      std::cout << "  kappa      = [ ";
      for (int j = 0; j < D+1; ++j)
        std::cout << kappa[j] << " ";
      std::cout << "]\n";
      std::cout << "-----------------------------------------\n\n";
    }


    const int N    = data.N;
    const int nvar = data.nvar;

    // Initialize z_real from user-provided z_io
    for (int i = 0; i < nvar; ++i)
    {
      data.z_real[i]   = z_io[i];
      data.z_double[i] = static_cast<double>(z_io[i]);
    }

    // NLopt phase
    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP,
                                 static_cast<unsigned>(nvar));
    if (!opt)
    {
      data.release();
      return -2;
    }

    nlopt_set_min_objective(opt, objective_cb<D,Real>, &data);

    int m_con = N * (D + 2);
    double* tol = static_cast<double*>(
      std::calloc(static_cast<std::size_t>(m_con), sizeof(double))
    );
    if (!tol)
    {
      nlopt_destroy(opt);
      data.release();
      return -3;
    }
    for (int i = 0; i < m_con; ++i)
    {
      tol[i] = 1e-14;
    }

    nlopt_add_inequality_mconstraint(
      opt,
      static_cast<unsigned>(m_con),
      ineq_mconstraint_cb<D,Real>,
      &data,
      tol
    );

    unsigned meq = 1;
    double eq_tol[1] = { 1e-14 };
    
    nlopt_add_equality_mconstraint(
      opt,
      meq,
      eq_constraint_cb<D,Real>,
      &data,
      eq_tol
    );

    for (int i = 0; i < nvar; ++i) {
      if (!std::isfinite(data.z_real[i])) {
        std::cerr << "NaN/Inf in z0 at i=" << i << "\n";
      }
    }

    nlopt_set_xtol_rel(opt, 1e-9);
    nlopt_set_ftol_rel(opt, static_cast<double>(opts.tol));
    nlopt_set_maxeval(opt, opts.max_nlopt_eval);
    std::vector<double> dx(nvar);  // nvar = (D+1)*N

    timer nl_timer;
    nl_timer.tic();
    double minf = 0.0;
    nlopt_result res = nlopt_optimize(opt, data.z_double, &minf);
    data.time_nlopt = nl_timer.toc();
    if (opts.verbose)
    {
      std::cout << "NLopt finished with result code " << res
                << ", f = " << minf << std::endl;
    }

    nlopt_destroy(opt);
    std::free(tol);

    for (int i = 0; i < nvar; ++i)
    {
      data.z_real[i] = static_cast<Real>(data.z_double[i]);
    }

    // Gauss-Newton refinement
    timer gn_timer;
    gn_timer.tic();
    gauss_newton<D,Real>(data, data.z_real, opts);
    data.time_gn = gn_timer.toc();
    // Copy result to z_io
    for (int i = 0; i < nvar; ++i)
    {
      z_io[i] = data.z_real[i];
    }

    // Optionally compute V_opt at final nodes
    if (V_opt)
    {
      unpack_z<D,Real>(data.z_real, N, data.X_real, data.w_real);

      Basis<D,Real>::eval_all(
        data.X_real,
        D, 1,
        N,
        data.kappa,
        data.m_basis,
        data.alpha_table,
        data.tail_deg,
        data.inv_h,
        V_opt,
        N,
        nullptr
      );
    }
    if (opts.verbose)
    {
      std::cout << "Final objective f(z_opt) = "
                << static_cast<double>(data.f_real) << "\n";
    
      if (V_opt)
      {
        Real condV = detail::cond_number<Real>(data.N, data.M, V_opt);
        std::cout << "Condition number of interp op on abscissa: "
                  << static_cast<double>(condV) << "\n";
      }
    
      std::cout << "\n[Timing diagnostics]\n";
      std::cout << "  NLopt time           : " << data.time_nlopt << " s\n";
      std::cout << "  Gauss-Newton time    : " << data.time_gn    << " s\n";

      if (data.count_obj > 0)
      {
        std::cout << "  Objective calls      : " << data.count_obj << "\n";
        std::cout << "  Objective time (tot) : " << data.time_obj << " s\n";
        std::cout << "  Objective time (avg) : "
                  << (data.time_obj / data.count_obj) << " s\n";
      }

      if (data.count_constr > 0)
      {
        std::cout << "  Constraint calls     : " << data.count_constr << "\n";
        std::cout << "  Constraint time (tot): " << data.time_constr << " s\n";
        std::cout << "  Constraint time (avg): "
                  << (data.time_constr / data.count_constr) << " s\n";
      }
      std::cout << std::endl;
    }

    data.release();
    return (res < 0 ? -4 : 0);
  }
};

/* ===================== Specialization for D = 1 ============================== */
/* For D=1, we do not run NLopt/GN. We just build the Gauss–Jacobi rule on [0,1].
   node_deg is interpreted as the number of points n.
   m_basis is the max polynomial degree for V_opt (dim Pi_m^1 = m+1).
   kappa[0],kappa[1] are Jacobi parameters.
   z_io layout: [ t_0,...,t_{n-1}, w_0,...,w_{n-1} ].
   If V_opt != nullptr, we also evaluate the 1D basis at these nodes.
*/

template<class Real>
struct QuadOptimizer<1, Real>
{
  static int optimize(int node_deg,
                      int m_basis,
                      const Real* kappa,
                      Real* z_io,
                      Real* V_opt,
                      const QuadOptimOptions<1,Real>& opts)
  {
    if (opts.verbose)
    {
      int N = node_deg;
      int eff_m_basis = 2*node_deg - 1;
      int eff_M       = eff_m_basis + 1;
    
      std::cout << "\n=========================================\n";
      std::cout << "Gauss–Jacobi Quadrature (D = 1)\n";
      std::cout << "-----------------------------------------\n";
      std::cout << "  n_points   = " << N << "\n";
      std::cout << "  exactness  = deg ≤ " << eff_m_basis << "\n";
      std::cout << "  M          = " << eff_M << "\n";
      std::cout << "  kappa      = [ " << kappa[0] << " , " << kappa[1] << " ]\n";
      std::cout << "-----------------------------------------\n\n";
    }

    (void)opts;  // no NLopt/GN here

    if (!kappa || !z_io || node_deg <= 0)
    {
      return -1;
    }

    unsigned int n = static_cast<unsigned int>(node_deg);

    Real* t = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(n), sizeof(Real))
    );
    Real* w = static_cast<Real*>(
      std::calloc(static_cast<std::size_t>(n), sizeof(Real))
    );

    if (!t || !w)
    {
      std::free(t);
      std::free(w);
      return -2;
    }

    // Build Gauss–Jacobi rule on [0,1]
    detail::gauss_jacobi_unit<Real>(n, kappa, t, w);

    // Pack into z_io
    for (unsigned int i = 0; i < n; ++i)
    {
      z_io[i]       = t[i];  // nodes
      z_io[n + i]   = w[i];  // weights
    }

    // Optionally build V_opt: basis P_m(t_i), m=0..m_basis (dim = m_basis+1)
    if (V_opt && m_basis >= 0)
    {
      int N = static_cast<int>(n);
      int M = Basis<1,Real>::dim_Pi(m_basis);  // = m_basis+1

      // Build temporary 1D basis metadata
      int*  alpha_table = static_cast<int*>(
        std::calloc(static_cast<std::size_t>(M), sizeof(int))
      );
      int*  tail_deg = static_cast<int*>(
        std::calloc(static_cast<std::size_t>(M), sizeof(int))
      );
      Real* inv_h = static_cast<Real*>(
        std::calloc(static_cast<std::size_t>(M), sizeof(Real))
      );

      if (!alpha_table || !tail_deg || !inv_h)
      {
        std::free(alpha_table);
        std::free(tail_deg);
        std::free(inv_h);
        std::free(t);
        std::free(w);
        return -3;
      }

      Basis<1,Real>::build_structures(
        kappa,
        m_basis,
        alpha_table,
        tail_deg,
        inv_h
      );

      // Evaluate basis at Gauss nodes
      // X: t (N x 1), ld_point = 1, ld_dim = 1, D=1
      Basis<1,Real>::eval_all(
        t,
        1, 1,
        N,
        kappa,
        m_basis,
        alpha_table,
        tail_deg,
        inv_h,
        V_opt,
        N,
        nullptr
      );

      std::free(alpha_table);
      std::free(tail_deg);
      std::free(inv_h);
    }

    std::free(t);
    std::free(w);

    return 0;
  }
};

} // namespace jsimplex

#endif // JQUAD_OPTIM_H
