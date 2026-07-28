#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <jmat.hh>
#include <jbasis.hh>
#include <jmult_c.h>
#include <jmult.hh>

using namespace jsimplex;

struct JMultOpaque
{
  void* impl = nullptr;

  int (*apply)(void* impl,
               const double* q,
               const double* c,
               double* y_out) = nullptr;

  int (*workspace_create)(void* impl,
                          void** workspace_impl_out) = nullptr;

  int (*apply_workspace)(void* impl,
                         void* workspace_impl,
                         const double* q,
                         const double* c,
                         double* y_out) = nullptr;

  void (*workspace_destroy)(void* workspace_impl) = nullptr;

  int (*test_concurrency)(void* impl,
                          const double* q,
                          const double* c,
                          int ntrials,
                          int nthreads,
                          double rtol,
                          double atol,
                          double* max_abs_error_out,
                          double* max_rel_error_out,
                          int* threads_used_out) = nullptr;

  void (*destroy)(void* impl) = nullptr;
};

struct JMultWorkspaceOpaque
{
  void* impl = nullptr;

  // Workspaces created by the C API are tied to the exact plan instance that
  // created them. This catches accidental cross-plan use even when dimensions
  // happen to match.
  const void* owner_impl = nullptr;

  void (*destroy)(void* workspace_impl) = nullptr;
};

template<int D>
struct JMultHandle
{
  // Owned copies
  std::vector<int> alpha_p; // Mp*D row-major

  // Owned CSC storage for J_poly and J_fun (allocated with malloc)
  int* colptr_poly[D];
  int* rowind_poly[D];
  double* x_poly[D];
  int N_poly = 0;
  int nnz_poly[D];

  int* colptr_fun[D];
  int* rowind_fun[D];
  double* x_fun[D];
  int N_fun = 0;
  int nnz_fun[D];

  // Views passed into MultByQClenshaw
  CSCView<double> J_poly_view[D];
  CSCView<double> J_fun_view[D];

  MultByQClenshaw<D,double> mbq;

  JMultHandle()
  {
    for (int i = 0; i < D; ++i)
    {
      colptr_poly[i] = nullptr; rowind_poly[i] = nullptr; x_poly[i] = nullptr; nnz_poly[i] = 0;
      colptr_fun[i]  = nullptr; rowind_fun[i]  = nullptr; x_fun[i]  = nullptr; nnz_fun[i]  = 0;
    }
  }

  ~JMultHandle()
  {
    // free CSC arrays
    for (int i = 0; i < D; ++i)
    {
      if (colptr_poly[i]) std::free(colptr_poly[i]);
      if (rowind_poly[i]) std::free(rowind_poly[i]);
      if (x_poly[i])      std::free(x_poly[i]);

      if (colptr_fun[i]) std::free(colptr_fun[i]);
      if (rowind_fun[i]) std::free(rowind_fun[i]);
      if (x_fun[i])      std::free(x_fun[i]);
    }
  }
};

template<int D>
struct JMultWorkspaceHandle
{
  MultByQClenshawWorkspace<D,double> workspace;
};

template<int D>
static inline int build_coord_csc(const double* kappa,
                                  int n,
                                  unsigned int nquad,
                                  int coord,
                                  int** colptr_out,
                                  int** rowind_out,
                                  double** x_out,
                                  int* N_out,
                                  int* nnz_out)
{
  const int N = Basis<D,double>::dim_Pi(n);

  int* colptr = nullptr;
  int* rowind = nullptr;
  double* x   = nullptr;

  const std::size_t nnz_s =
    JMat<D,double>::build_pruned_csc(n, kappa, nquad, coord, &colptr, &rowind, &x);

  if (nnz_s > (std::size_t)std::numeric_limits<int>::max())
  {
    if (colptr) std::free(colptr);
    if (rowind) std::free(rowind);
    if (x)      std::free(x);
    return 4;
  }

  *colptr_out = colptr;
  *rowind_out = rowind;
  *x_out      = x;
  *N_out      = N;
  *nnz_out    = (int)nnz_s;
  return 0;
}

template<int D>
static inline int jmult_create_D(const double* kappa,
                                 int p,
                                 int K,
                                 const int* alpha_p,
                                 int Mp,
                                 int assume_symmetric,
                                 jmult_handle_t* handle_out)
{
  if (!kappa || !alpha_p || !handle_out) return 1;
  if (p < 0 || K < 0) return 2;
  if (Mp != Basis<D,double>::dim_Pi(p)) return 3;

  auto* H = new JMultHandle<D>();

  // copy alpha_p into owned storage
  H->alpha_p.assign(alpha_p, alpha_p + (std::size_t)Mp * D);

  // Build J_poly for degree p, J_fun for degree K
  const unsigned int nquad_poly = (unsigned int)(p + 1);
  const unsigned int nquad_fun  = (unsigned int)(K + 1);

  int ret = 0;

  for (int coord = 0; coord < D; ++coord)
  {
    ret = build_coord_csc<D>(kappa, p, nquad_poly, coord,
                             &H->colptr_poly[coord],
                             &H->rowind_poly[coord],
                             &H->x_poly[coord],
                             &H->N_poly,
                             &H->nnz_poly[coord]);
    if (ret != 0) { delete H; return 10 + ret; }

    ret = build_coord_csc<D>(kappa, K, nquad_fun, coord,
                             &H->colptr_fun[coord],
                             &H->rowind_fun[coord],
                             &H->x_fun[coord],
                             &H->N_fun,
                             &H->nnz_fun[coord]);
    if (ret != 0) { delete H; return 20 + ret; }

    H->J_poly_view[coord] = CSCView<double>{
      H->N_poly, H->N_poly,
      H->colptr_poly[coord],
      H->rowind_poly[coord],
      H->x_poly[coord]
    };
    H->J_fun_view[coord] = CSCView<double>{
      H->N_fun, H->N_fun,
      H->colptr_fun[coord],
      H->rowind_fun[coord],
      H->x_fun[coord]
    };
  }

  // init mbq
  H->mbq.init(p, K,
              H->alpha_p.data(), Mp,
              H->J_poly_view,
              H->J_fun_view,
              assume_symmetric != 0);

  *handle_out = (jmult_handle_t)H;
  return 0;
}

template<int D>
static inline int jmult_apply_D(jmult_handle_t handle,
                                const double* q,
                                const double* c,
                                double* y_out)
{
  if (!handle || !q || !c || !y_out) return 1;
  auto* H = (JMultHandle<D>*)handle;
  H->mbq.apply(q, c, y_out);
  return 0;
}

template<int D>
static inline int jmult_workspace_create_D(jmult_handle_t handle,
                                           void** workspace_impl_out)
{
  if (!handle || !workspace_impl_out) return 1;
  *workspace_impl_out = nullptr;

  auto* H = (JMultHandle<D>*)handle;
  JMultWorkspaceHandle<D>* W = nullptr;

  try
  {
    W = new JMultWorkspaceHandle<D>();
    W->workspace.init(H->mbq.p, H->mbq.MK);
  }
  catch (...)
  {
    delete W;
    return 4;
  }

  *workspace_impl_out = (void*)W;
  return 0;
}

template<int D>
static inline int jmult_apply_workspace_D(jmult_handle_t handle,
                                          void* workspace_impl,
                                          const double* q,
                                          const double* c,
                                          double* y_out)
{
  if (!handle || !workspace_impl || !q || !c || !y_out) return 1;

  auto* H = (JMultHandle<D>*)handle;
  auto* W = (JMultWorkspaceHandle<D>*)workspace_impl;

  if (!W->workspace.compatible(H->mbq.p, H->mbq.MK)) return 2;

  H->mbq.apply(q, c, y_out, W->workspace);
  return 0;
}

template<int D>
static inline void jmult_workspace_destroy_D(void* workspace_impl)
{
  auto* W = (JMultWorkspaceHandle<D>*)workspace_impl;
  delete W;
}

static inline double jmult_case_value(double base,
                                      int trial,
                                      int index,
                                      int salt)
{
  // Integer-only pattern generation keeps serial and parallel test inputs
  // identical without relying on a random-number generator or shared state.
  const int pattern = (((trial + 1) * (index + 3 + salt)) % 19) - 9;
  const double scale = 1.0e-3 * (double)pattern;
  return base + scale * (1.0 + std::abs(base));
}

template<int D>
static inline int jmult_test_concurrency_D(jmult_handle_t handle,
                                           const double* q,
                                           const double* c,
                                           int ntrials,
                                           int nthreads,
                                           double rtol,
                                           double atol,
                                           double* max_abs_error_out,
                                           double* max_rel_error_out,
                                           int* threads_used_out)
{
  if (max_abs_error_out) *max_abs_error_out = 0.0;
  if (max_rel_error_out) *max_rel_error_out = 0.0;
  if (threads_used_out) *threads_used_out = 0;

  if (!handle || !q || !c) return 1;
  if (ntrials <= 0 || nthreads < 0 || rtol < 0.0 || atol < 0.0) return 3;

#ifndef _OPENMP
  (void)nthreads;
  return 7;
#else
  auto* H = (JMultHandle<D>*)handle;
  const int Mp = H->mbq.Mp;
  const int MK = H->mbq.MK;

  int requested_threads = nthreads;
  if (requested_threads <= 0) requested_threads = omp_get_max_threads();
  requested_threads = std::max(1, requested_threads);
  requested_threads = std::min(requested_threads, ntrials);

  try
  {
    std::vector<double> q_cases((std::size_t)ntrials * (std::size_t)Mp);
    std::vector<double> c_cases((std::size_t)ntrials * (std::size_t)MK);
    std::vector<double> serial_results((std::size_t)ntrials * (std::size_t)MK);
    std::vector<double> abs_errors((std::size_t)ntrials, 0.0);
    std::vector<double> rel_errors((std::size_t)ntrials, 0.0);
    std::vector<unsigned char> failed((std::size_t)ntrials, 0);

    MultByQClenshawWorkspace<D,double> serial_workspace;
    serial_workspace.init(H->mbq.p, H->mbq.MK);

    for (int trial = 0; trial < ntrials; ++trial)
    {
      double* qt = q_cases.data() + (std::size_t)trial * (std::size_t)Mp;
      double* ct = c_cases.data() + (std::size_t)trial * (std::size_t)MK;
      double* yt = serial_results.data() + (std::size_t)trial * (std::size_t)MK;

      for (int i = 0; i < Mp; ++i)
        qt[i] = jmult_case_value(q[i], trial, i, 0);

      for (int i = 0; i < MK; ++i)
        ct[i] = jmult_case_value(c[i], trial, i, 7);

      H->mbq.apply(qt, ct, yt, serial_workspace);
    }

    std::vector<MultByQClenshawWorkspace<D,double>> workspaces(
      (std::size_t)requested_threads);
    for (int t = 0; t < requested_threads; ++t)
      workspaces[(std::size_t)t].init(H->mbq.p, H->mbq.MK);

    int actual_threads = 1;

#pragma omp parallel num_threads(requested_threads) shared(actual_threads)
    {
      const int tid = omp_get_thread_num();
      std::vector<double> y((std::size_t)MK);

#pragma omp single
      actual_threads = omp_get_num_threads();

#pragma omp for schedule(static)
      for (int trial = 0; trial < ntrials; ++trial)
      {
        const double* qt = q_cases.data() + (std::size_t)trial * (std::size_t)Mp;
        const double* ct = c_cases.data() + (std::size_t)trial * (std::size_t)MK;
        const double* yr = serial_results.data() + (std::size_t)trial * (std::size_t)MK;

        H->mbq.apply(qt, ct, y.data(), workspaces[(std::size_t)tid]);

        double abs_error = 0.0;
        double ref_norm = 0.0;
        for (int i = 0; i < MK; ++i)
        {
          abs_error = std::max(abs_error, std::abs(y[(std::size_t)i] - yr[i]));
          ref_norm = std::max(ref_norm, std::abs(yr[i]));
        }

        const double rel_error =
          abs_error / std::max(1.0e-300, ref_norm);

        abs_errors[(std::size_t)trial] = abs_error;
        rel_errors[(std::size_t)trial] = rel_error;
        failed[(std::size_t)trial] =
          (abs_error > atol + rtol * ref_norm) ? 1 : 0;
      }
    }

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool any_failed = false;

    for (int trial = 0; trial < ntrials; ++trial)
    {
      max_abs_error = std::max(max_abs_error, abs_errors[(std::size_t)trial]);
      max_rel_error = std::max(max_rel_error, rel_errors[(std::size_t)trial]);
      any_failed = any_failed || (failed[(std::size_t)trial] != 0);
    }

    if (max_abs_error_out) *max_abs_error_out = max_abs_error;
    if (max_rel_error_out) *max_rel_error_out = max_rel_error;
    if (threads_used_out) *threads_used_out = actual_threads;

    return any_failed ? 6 : 0;
  }
  catch (...)
  {
    return 4;
  }
#endif
}

template<int D>
static inline void jmult_destroy_D(jmult_handle_t handle)
{
  auto* H = (JMultHandle<D>*)handle;
  delete H;
}

template<int D>
static int jmult_apply_opaque(void* impl,
                              const double* q,
                              const double* c,
                              double* y_out)
{
  return jmult_apply_D<D>((jmult_handle_t)impl, q, c, y_out);
}

template<int D>
static int jmult_workspace_create_opaque(void* impl,
                                         void** workspace_impl_out)
{
  return jmult_workspace_create_D<D>(
    (jmult_handle_t)impl,
    workspace_impl_out);
}

template<int D>
static int jmult_apply_workspace_opaque(void* impl,
                                        void* workspace_impl,
                                        const double* q,
                                        const double* c,
                                        double* y_out)
{
  return jmult_apply_workspace_D<D>(
    (jmult_handle_t)impl,
    workspace_impl,
    q,
    c,
    y_out);
}

template<int D>
static void jmult_workspace_destroy_opaque(void* workspace_impl)
{
  jmult_workspace_destroy_D<D>(workspace_impl);
}

template<int D>
static int jmult_test_concurrency_opaque(void* impl,
                                         const double* q,
                                         const double* c,
                                         int ntrials,
                                         int nthreads,
                                         double rtol,
                                         double atol,
                                         double* max_abs_error_out,
                                         double* max_rel_error_out,
                                         int* threads_used_out)
{
  return jmult_test_concurrency_D<D>(
    (jmult_handle_t)impl,
    q,
    c,
    ntrials,
    nthreads,
    rtol,
    atol,
    max_abs_error_out,
    max_rel_error_out,
    threads_used_out);
}

template<int D>
static void jmult_destroy_opaque(void* impl)
{
  jmult_destroy_D<D>((jmult_handle_t)impl);
}


/* Public C API */

extern "C"
{

int jmult_clenshaw_create(const double* kappa,
                          int D,
                          int p,
                          int K,
                          const int* alpha_p,
                          int Mp,
                          int assume_symmetric,
                          jmult_handle_t* handle_out)
{
  if (!handle_out) return 1;
  *handle_out = nullptr;

  if (!kappa || !alpha_p) return 1;
  if (D < 1 || D > 5) return 5;

  // Allocate wrapper first (so apply/destroy always know what to do).
  JMultOpaque* W = new JMultOpaque();

  int ret = 0;
  jmult_handle_t impl = nullptr;

  switch (D)
  {
    case 1:
      ret = jmult_create_D<1>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0)
      {
        W->impl = (void*)impl;
        W->apply = &jmult_apply_opaque<1>;
        W->workspace_create = &jmult_workspace_create_opaque<1>;
        W->apply_workspace = &jmult_apply_workspace_opaque<1>;
        W->workspace_destroy = &jmult_workspace_destroy_opaque<1>;
        W->test_concurrency = &jmult_test_concurrency_opaque<1>;
        W->destroy = &jmult_destroy_opaque<1>;
      }
      break;
    case 2:
      ret = jmult_create_D<2>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0)
      {
        W->impl = (void*)impl;
        W->apply = &jmult_apply_opaque<2>;
        W->workspace_create = &jmult_workspace_create_opaque<2>;
        W->apply_workspace = &jmult_apply_workspace_opaque<2>;
        W->workspace_destroy = &jmult_workspace_destroy_opaque<2>;
        W->test_concurrency = &jmult_test_concurrency_opaque<2>;
        W->destroy = &jmult_destroy_opaque<2>;
      }
      break;
    case 3:
      ret = jmult_create_D<3>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0)
      {
        W->impl = (void*)impl;
        W->apply = &jmult_apply_opaque<3>;
        W->workspace_create = &jmult_workspace_create_opaque<3>;
        W->apply_workspace = &jmult_apply_workspace_opaque<3>;
        W->workspace_destroy = &jmult_workspace_destroy_opaque<3>;
        W->test_concurrency = &jmult_test_concurrency_opaque<3>;
        W->destroy = &jmult_destroy_opaque<3>;
      }
      break;
    case 4:
      ret = jmult_create_D<4>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0)
      {
        W->impl = (void*)impl;
        W->apply = &jmult_apply_opaque<4>;
        W->workspace_create = &jmult_workspace_create_opaque<4>;
        W->apply_workspace = &jmult_apply_workspace_opaque<4>;
        W->workspace_destroy = &jmult_workspace_destroy_opaque<4>;
        W->test_concurrency = &jmult_test_concurrency_opaque<4>;
        W->destroy = &jmult_destroy_opaque<4>;
      }
      break;
    case 5:
      ret = jmult_create_D<5>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0)
      {
        W->impl = (void*)impl;
        W->apply = &jmult_apply_opaque<5>;
        W->workspace_create = &jmult_workspace_create_opaque<5>;
        W->apply_workspace = &jmult_apply_workspace_opaque<5>;
        W->workspace_destroy = &jmult_workspace_destroy_opaque<5>;
        W->test_concurrency = &jmult_test_concurrency_opaque<5>;
        W->destroy = &jmult_destroy_opaque<5>;
      }
      break;
    default:
      ret = 5;
      break;
  }

  if (ret != 0)
  {
    delete W;
    return ret;
  }

  *handle_out = (jmult_handle_t)W;
  return 0;
}

int jmult_clenshaw_apply(jmult_handle_t handle,
                         const double* q,
                         const double* c,
                         double* y_out)
{
  if (!handle || !q || !c || !y_out) return 1;

  JMultOpaque* W = (JMultOpaque*)handle;
  if (!W->impl || !W->apply) return 2;

  return W->apply(W->impl, q, c, y_out);
}

int jmult_clenshaw_workspace_create(jmult_handle_t plan,
                                    jmult_workspace_t* workspace_out)
{
  if (!workspace_out) return 1;
  *workspace_out = nullptr;

  if (!plan) return 1;

  JMultOpaque* P = (JMultOpaque*)plan;
  if (!P->impl || !P->workspace_create || !P->workspace_destroy) return 2;

  void* workspace_impl = nullptr;
  const int ret = P->workspace_create(P->impl, &workspace_impl);
  if (ret != 0) return ret;
  if (!workspace_impl) return 4;

  JMultWorkspaceOpaque* W = nullptr;
  try
  {
    W = new JMultWorkspaceOpaque();
  }
  catch (...)
  {
    P->workspace_destroy(workspace_impl);
    return 4;
  }

  W->impl = workspace_impl;
  W->owner_impl = P->impl;
  W->destroy = P->workspace_destroy;

  *workspace_out = (jmult_workspace_t)W;
  return 0;
}

int jmult_clenshaw_apply_workspace(jmult_handle_t plan,
                                   jmult_workspace_t workspace,
                                   const double* q,
                                   const double* c,
                                   double* y_out)
{
  if (!plan || !workspace || !q || !c || !y_out) return 1;

  JMultOpaque* P = (JMultOpaque*)plan;
  JMultWorkspaceOpaque* W = (JMultWorkspaceOpaque*)workspace;

  if (!P->impl || !P->apply_workspace || !W->impl) return 2;
  if (W->owner_impl != P->impl) return 2;

  return P->apply_workspace(P->impl, W->impl, q, c, y_out);
}

void jmult_clenshaw_workspace_destroy(jmult_workspace_t workspace)
{
  if (!workspace) return;

  JMultWorkspaceOpaque* W = (JMultWorkspaceOpaque*)workspace;
  if (W->destroy && W->impl)
  {
    W->destroy(W->impl);
    W->impl = nullptr;
  }
  delete W;
}

int jmult_clenshaw_test_concurrency(jmult_handle_t plan,
                                    const double* q,
                                    const double* c,
                                    int ntrials,
                                    int nthreads,
                                    double rtol,
                                    double atol,
                                    double* max_abs_error_out,
                                    double* max_rel_error_out,
                                    int* threads_used_out)
{
  if (!plan || !q || !c) return 1;

  JMultOpaque* P = (JMultOpaque*)plan;
  if (!P->impl || !P->test_concurrency) return 2;

  return P->test_concurrency(
    P->impl,
    q,
    c,
    ntrials,
    nthreads,
    rtol,
    atol,
    max_abs_error_out,
    max_rel_error_out,
    threads_used_out);
}

void jmult_clenshaw_destroy(jmult_handle_t handle)
{
  if (!handle) return;

  JMultOpaque* W = (JMultOpaque*)handle;
  if (W->destroy && W->impl)
  {
    W->destroy(W->impl);
    W->impl = nullptr;
  }
  delete W;
}

} // extern "C"
