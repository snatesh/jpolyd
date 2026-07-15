#include <jlsmr_c.h>
#include <jlsmr.hh>

extern "C" {

int lsmr_dense_solve_colmajor(
  int m,
  int n,
  const double* A_colmajor,
  const double* b,
  double* x,
  double damp,
  double atol,
  double btol,
  double conlim,
  int itnlim,
  int nout,
  int localsize,
  int ctest,
  int* istop_out,
  int* itn_out,
  int* stat_out,
  double* normr_out,
  double* normA_out,
  double* condA_out,
  double* normb_out,
  double* normx_out,
  double* normAr_out)
{
  if (m <= 0 || n <= 0)
  {
    return -1;
  }

  if (A_colmajor == nullptr || b == nullptr || x == nullptr ||
      istop_out == nullptr || itn_out == nullptr || stat_out == nullptr ||
      normr_out == nullptr || normA_out == nullptr || condA_out == nullptr ||
      normb_out == nullptr || normx_out == nullptr || normAr_out == nullptr)
  {
    return -2;
  }

  jsimplex::detail::LsmrOptions<double> opt;
  opt.damp = damp;
  opt.atol = atol;
  opt.btol = btol;
  opt.conlim = conlim;
  opt.itnlim = itnlim;
  opt.nout = nout;
  opt.localsize = localsize;
  opt.ctest = ctest;

  jsimplex::detail::LsmrInfo<double> info;

  const int ret = jsimplex::lsmr_dense_solve_colmajor<double>(
    m,
    n,
    A_colmajor,
    b,
    x,
    opt,
    &info);

  *istop_out = info.istop;
  *itn_out = info.itn;
  *stat_out = info.stat;
  *normr_out = info.normr;
  *normA_out = info.normA;
  *condA_out = info.condA;
  *normb_out = info.normb;
  *normx_out = info.normx;
  *normAr_out = info.normAr;

  return ret;
}

} // extern "C"

//#include <algorithm>
//#include <cmath>
//#include <cstddef>
//#include <exception>
//#include <limits>
//#include <vector>
//
//extern "C" {
//
//void lsmr_c_set_options(
//    const double* atol,
//    const double* btol,
//    const double* conlim,
//    const int* itnlim,
//    const int* nout,
//    const int* localsize,
//    const int* ctest);
//
//void lsmr_c_step(
//    const int* m,
//    const int* n,
//    int* action,
//    double* u,
//    double* v,
//    const double* b,
//    const double* damp,
//    double* x,
//    int* istop,
//    int* itn,
//    int* stat,
//    double* normr,
//    double* normA,
//    double* condA,
//    double* normb,
//    double* normx,
//    double* normAr);
//
//} // extern "C"
//
//namespace {
//
//// Column-major dense matrix storage:
////
////   A(i,j) = A[i + m*j],  0 <= i < m, 0 <= j < n.
////
//// This matches Fortran/BLAS/NumPy F_CONTIGUOUS layout.
//
//void apply_A_colmajor(
//    const int m,
//    const int n,
//    const double* A,
//    const double* x,
//    double* y) {
//  // y = y + A*x
//  for (int j = 0; j < n; ++j) {
//    const double xj = x[j];
//    const double* Aj = A + static_cast<std::ptrdiff_t>(m) * j;
//
//    for (int i = 0; i < m; ++i) {
//      y[i] += Aj[i] * xj;
//    }
//  }
//}
//
//void apply_AT_colmajor(
//    const int m,
//    const int n,
//    const double* A,
//    const double* y,
//    double* x) {
//  // x = x + A^T*y
//  for (int j = 0; j < n; ++j) {
//    const double* Aj = A + static_cast<std::ptrdiff_t>(m) * j;
//    double sum = 0.0;
//
//    for (int i = 0; i < m; ++i) {
//      sum += Aj[i] * y[i];
//    }
//
//    x[j] += sum;
//  }
//}
//
//void cleanup_lsmr_state(
//    const int m,
//    const int n,
//    std::vector<double>& u,
//    std::vector<double>& v,
//    const double* b,
//    const double damp,
//    double* x,
//    int& istop,
//    int& itn,
//    int& stat,
//    double& normr,
//    double& normA,
//    double& condA,
//    double& normb,
//    double& normx,
//    double& normAr) {
//  int action = 10;
//
//  lsmr_c_step(
//      &m,
//      &n,
//      &action,
//      u.data(),
//      v.data(),
//      b,
//      &damp,
//      x,
//      &istop,
//      &itn,
//      &stat,
//      &normr,
//      &normA,
//      &condA,
//      &normb,
//      &normx,
//      &normAr);
//}
//
//} // namespace
//
//extern "C" {
//
//// Solve min_x ||A*x - b||_2 using the Fortran LSMR reverse-communication core.
////
//// Inputs:
////   m, n        Matrix dimensions, A is m-by-n.
////   A_colmajor  Dense column-major matrix, A(i,j)=A_colmajor[i + m*j].
////   b           Right-hand side length m.
////   damp        LSMR damping parameter. Use 0 for ordinary least squares.
////   atol, btol  LSMR stopping tolerances.
////   conlim      LSMR condition estimate stopping threshold.
////   itnlim      Maximum iterations.
////   nout        Fortran output unit. Use 0 to suppress output if supported by SOL code,
////               or 6 for stdout.
////   localsize   Local reorthogonalization storage count. Use 0 initially.
////   ctest       SOL convergence-test setting. Use 3 for detailed tests/diagnostics.
////
//// Outputs:
////   x           Solution length n. This routine overwrites it, starting from zero.
////   istop,itn,stat,normr,normA,condA,normb,normx,normAr
////               LSMR diagnostics copied from inform.
////
//// Return code:
////    0  success from wrapper perspective; inspect istop/stat for LSMR status.
////   -1  invalid dimensions.
////   -2  null required pointer.
////   -3  allocation failure or C++ exception.
////   -4  unexpected reverse-communication action.
////
//// Notes:
////   This uses the singleton Fortran shim state. It is suitable for sequential ctypes
////   calls, but not thread-safe/concurrent solves. For concurrency, replace the shim
////   with a handle-based API that allocates one keep/options/inform state per solver.
//int lsmr_dense_solve_colmajor(
//    int m,
//    int n,
//    const double* A_colmajor,
//    const double* b,
//    double* x,
//    double damp,
//    double atol,
//    double btol,
//    double conlim,
//    int itnlim,
//    int nout,
//    int localsize,
//    int ctest,
//    int* istop_out,
//    int* itn_out,
//    int* stat_out,
//    double* normr_out,
//    double* normA_out,
//    double* condA_out,
//    double* normb_out,
//    double* normx_out,
//    double* normAr_out) {
//  if (m <= 0 || n <= 0) {
//    return -1;
//  }
//
//  if (A_colmajor == nullptr || b == nullptr || x == nullptr ||
//      istop_out == nullptr || itn_out == nullptr || stat_out == nullptr ||
//      normr_out == nullptr || normA_out == nullptr || condA_out == nullptr ||
//      normb_out == nullptr || normx_out == nullptr || normAr_out == nullptr) {
//    return -2;
//  }
//
//  try {
//    std::vector<double> u(static_cast<std::size_t>(m), 0.0);
//    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
//
//    std::fill(x, x + n, 0.0);
//
//    lsmr_c_set_options(
//        &atol,
//        &btol,
//        &conlim,
//        &itnlim,
//        &nout,
//        &localsize,
//        &ctest);
//
//    int action = 0;
//
//    int istop = 0;
//    int itn = 0;
//    int stat = 0;
//
//    double normr = 0.0;
//    double normA = 0.0;
//    double condA = 0.0;
//    double normb = 0.0;
//    double normx = 0.0;
//    double normAr = 0.0;
//
//    while (true) {
//      lsmr_c_step(
//          &m,
//          &n,
//          &action,
//          u.data(),
//          v.data(),
//          b,
//          &damp,
//          x,
//          &istop,
//          &itn,
//          &stat,
//          &normr,
//          &normA,
//          &condA,
//          &normb,
//          &normx,
//          &normAr);
//
//      if (action == 0) {
//        break;
//      }
//
//      if (action == 1) {
//        // v = v + A^T*u
//        apply_AT_colmajor(m, n, A_colmajor, u.data(), v.data());
//      } else if (action == 2) {
//        // u = u + A*v
//        apply_A_colmajor(m, n, A_colmajor, v.data(), u.data());
//      } else {
//        cleanup_lsmr_state(
//            m,
//            n,
//            u,
//            v,
//            b,
//            damp,
//            x,
//            istop,
//            itn,
//            stat,
//            normr,
//            normA,
//            condA,
//            normb,
//            normx,
//            normAr);
//        return -4;
//      }
//    }
//
//    cleanup_lsmr_state(
//        m,
//        n,
//        u,
//        v,
//        b,
//        damp,
//        x,
//        istop,
//        itn,
//        stat,
//        normr,
//        normA,
//        condA,
//        normb,
//        normx,
//        normAr);
//
//    *istop_out = istop;
//    *itn_out = itn;
//    *stat_out = stat;
//    *normr_out = normr;
//    *normA_out = normA;
//    *condA_out = condA;
//    *normb_out = normb;
//    *normx_out = normx;
//    *normAr_out = normAr;
//
//    return 0;
//  } catch (...) {
//    return -3;
//  }
//}
//
//} // extern "C"
