#include <cstdlib>
#include <cstring>
#include <limits>

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

  void (*destroy)(void* impl) = nullptr;
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
      if (ret == 0) { W->impl = (void*)impl; W->apply = &jmult_apply_opaque<1>; W->destroy = &jmult_destroy_opaque<1>; }
      break;
    case 2:
      ret = jmult_create_D<2>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0) { W->impl = (void*)impl; W->apply = &jmult_apply_opaque<2>; W->destroy = &jmult_destroy_opaque<2>; }
      break;
    case 3:
      ret = jmult_create_D<3>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0) { W->impl = (void*)impl; W->apply = &jmult_apply_opaque<3>; W->destroy = &jmult_destroy_opaque<3>; }
      break;
    case 4:
      ret = jmult_create_D<4>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0) { W->impl = (void*)impl; W->apply = &jmult_apply_opaque<4>; W->destroy = &jmult_destroy_opaque<4>; }
      break;
    case 5:
      ret = jmult_create_D<5>(kappa, p, K, alpha_p, Mp, assume_symmetric, &impl);
      if (ret == 0) { W->impl = (void*)impl; W->apply = &jmult_apply_opaque<5>; W->destroy = &jmult_destroy_opaque<5>; }
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
