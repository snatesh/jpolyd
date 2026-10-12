#ifndef JNEUMANN_ROOT_HH
#define JNEUMANN_ROOT_HH

#include <jdetail.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace jsimplex {

// Algebraic root solve for an HPS Neumann operator with a verified,
// one-dimensional constant-trace right kernel. The matrix and vectors
// are expressed in the existing root-face coefficient ordering.
//
// R lambda = h, w^T lambda = eta.
//
// The left null vector ell is computed at factorization time, and the
// compatible system is solved by factoring [R ell; w^T 0].
// The factorization is source-independent and may be reused for many h, eta.
template<int D, class Real>
struct NeumannRootPlan
{
  static_assert(D >= 1, "NeumannRootPlan requires D >= 1");

  int nb = 0;
  std::vector<Real> left_null;
  std::vector<Real> gauge_row;
  std::vector<Real> bordered_lu;
  std::vector<lapack_int> piv;

  Real sigma_max = Real(0);
  Real sigma_min = Real(0);
  Real sigma_next = Real(0);
  Real constant_residual_rel = Real(0);
};

namespace neumann_root_detail {

template<class Real>
inline Real dot(const std::vector<Real>& x, const std::vector<Real>& y)
{
  if (x.size() != y.size())
    throw std::invalid_argument("Neumann root: dot dimension mismatch");

  Real sum = Real(0);
  for (std::size_t i = 0; i < x.size(); ++i)
    sum += x[i] * y[i];
  return sum;
}

template<class Real>
inline Real norm2(const std::vector<Real>& x)
{
  return std::sqrt(dot(x, x));
}

} // namespace neumann_root_detail

// R: nb x nb column-major root operator.
// v: coefficient vector representing the constant boundary trace.
// w: prescribed gauge row (usually normalized so w^T v = 1).
//
// This function is deliberately only for the rank-(nb-1) case, e.g. c=0
// with an ordinary elliptic operator. For nonsingular pure-flux problems
// with reaction, keep using the usual root LU factorization.
template<int D, class Real>
inline NeumannRootPlan<D,Real> factor_neumann_root(
  const std::vector<Real>& R,
  const std::vector<Real>& v,
  const std::vector<Real>& w)
{
  static_assert(D >= 1, "factor_neumann_root requires D >= 1");

  const int nb = static_cast<int>(v.size());
  if (nb < 2 || R.size() != (std::size_t)nb * nb ||
      w.size() != (std::size_t)nb)
    throw std::invalid_argument("Neumann root: invalid matrix/vector sizes");

  const Real vn = neumann_root_detail::norm2(v);
  const Real wn = neumann_root_detail::norm2(w);
  const Real wv = neumann_root_detail::dot(w, v);
  const Real eps = std::numeric_limits<Real>::epsilon();
  if (!(vn > Real(0)) || !(wn > Real(0)) ||
      !(std::abs(wv) > Real(100) * eps * vn * wn))
    throw std::invalid_argument("Neumann root: gauge does not fix constants");

  std::vector<Real> Rcopy = R;
  std::vector<Real> singular((std::size_t)nb);
  std::vector<Real> U((std::size_t)nb * nb);
  std::vector<Real> VT((std::size_t)nb * nb);

  // LapackGesdd already exists in jdetail.hh. 'A' requests full U and
  // VT; the last column of U is the numerical left null vector.
  const lapack_int info_svd = detail::LapackGesdd<Real>::run(
    'A', (lapack_int)nb, (lapack_int)nb,
    Rcopy.data(), (lapack_int)nb,
    singular.data(), U.data(), (lapack_int)nb,
    VT.data(), (lapack_int)nb);
  if (info_svd != 0)
    throw std::runtime_error("Neumann root: SVD failed");

  NeumannRootPlan<D,Real> plan;
  plan.nb = nb;
  plan.sigma_max = singular.front();
  plan.sigma_next = singular[(std::size_t)nb - 2];
  plan.sigma_min = singular.back();

  if (!(plan.sigma_max > Real(0)))
    throw std::runtime_error("Neumann root: zero matrix / multiple null modes");

  const Real rank_rel_tol = Real(1000) * eps * Real(nb);
  const Real rank_abs_tol = rank_rel_tol * plan.sigma_max;
  if (plan.sigma_min > rank_abs_tol ||
      plan.sigma_next <= Real(10) * rank_abs_tol)
    throw std::runtime_error(
      "Neumann root: expected exactly one numerical null mode");

  std::vector<Real> Rv((std::size_t)nb, Real(0));
  for (int j = 0; j < nb; ++j)
    for (int i = 0; i < nb; ++i)
      Rv[(std::size_t)i] +=
        R[(std::size_t)i + (std::size_t)nb * j] * v[(std::size_t)j];

  plan.constant_residual_rel =
    neumann_root_detail::norm2(Rv) / (plan.sigma_max * vn);
  if (plan.constant_residual_rel > Real(10) * rank_rel_tol)
    throw std::runtime_error("Neumann root: kernel is not the constant trace");

  plan.left_null.resize((std::size_t)nb);
  for (int i = 0; i < nb; ++i)
    plan.left_null[(std::size_t)i] =
      U[(std::size_t)i + (std::size_t)nb * (nb - 1)];
  plan.gauge_row = w;

  // Build [R ell; w^T 0] in column-major storage.
  const int N = nb + 1;
  plan.bordered_lu.assign((std::size_t)N * N, Real(0));
  for (int j = 0; j < nb; ++j)
  {
    for (int i = 0; i < nb; ++i)
      plan.bordered_lu[(std::size_t)i + (std::size_t)N * j] =
        R[(std::size_t)i + (std::size_t)nb * j];
    plan.bordered_lu[(std::size_t)nb + (std::size_t)N * j] =
      w[(std::size_t)j];
  }
  for (int i = 0; i < nb; ++i)
    plan.bordered_lu[(std::size_t)i + (std::size_t)N * nb] =
      plan.left_null[(std::size_t)i];

  plan.piv.resize((std::size_t)N);
  const lapack_int info_lu = detail::LapackGetrf<Real>::run(
    (lapack_int)N, (lapack_int)N,
    plan.bordered_lu.data(), (lapack_int)N, plan.piv.data());
  if (info_lu != 0)
    throw std::runtime_error("Neumann root: bordered LU failed");

  return plan;
}

// Check compatibility before solving. The extra bordered variable must not
// silently absorb inconsistent source and boundary data.
// A negative compatibility_rtol requests a precision-dependent default.
template<int D, class Real>
inline void solve_neumann_root(
  const NeumannRootPlan<D,Real>& plan,
  const std::vector<Real>& h,
  Real gauge_rhs,
  std::vector<Real>& lambda,
  Real compatibility_rtol = Real(-1))
{
  const int nb = plan.nb;
  if (nb < 2 || h.size() != (std::size_t)nb ||
      !std::isfinite(gauge_rhs))
    throw std::invalid_argument("Neumann root: invalid solve inputs");

  if (compatibility_rtol < Real(0))
    compatibility_rtol = std::max(
      Real(1e-9), Real(100) * std::numeric_limits<Real>::epsilon() * Real(nb));
  if (!(compatibility_rtol > Real(0)) ||
      !std::isfinite(compatibility_rtol))
    throw std::invalid_argument("Neumann root: invalid compatibility tolerance");

  const Real defect = std::abs(neumann_root_detail::dot(plan.left_null, h));
  const Real scale = std::max(Real(1), neumann_root_detail::norm2(h));
  if (defect > compatibility_rtol * scale)
  {
    std::ostringstream msg;
    msg << "Neumann root: incompatible data, left-null defect="
        << defect / scale << " exceeds tolerance=" << compatibility_rtol;
    throw std::runtime_error(msg.str());
  }

  const int N = nb + 1;
  std::vector<Real> rhs((std::size_t)N);
  std::copy(h.begin(), h.end(), rhs.begin());
  rhs[(std::size_t)nb] = gauge_rhs;

  const lapack_int info = detail::LapackGetrs<Real>::run(
    'N', (lapack_int)N, 1,
    plan.bordered_lu.data(), (lapack_int)N,
    plan.piv.data(), rhs.data(), (lapack_int)N);
  if (info != 0)
    throw std::runtime_error("Neumann root: bordered solve failed");

  lambda.assign(rhs.begin(), rhs.begin() + nb);
}

} // namespace jsimplex

#endif // JNEUMANN_ROOT_HH
