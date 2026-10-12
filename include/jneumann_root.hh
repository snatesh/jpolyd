#ifndef JNEUMANN_ROOT_HH
#define JNEUMANN_ROOT_HH

#include <jdetail.hh>
#include <jnode.hh>
#include <jprecomp.hh>

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

// Classify the completed pure-flux root operator by its numerical rank.
// Returns true only for a one-dimensional constant-trace nullspace.
// A full-rank root (e.g. c > 0) requires no gauge. Other singularities
// are outside the present constant-gauge implementation.
template<int D, class Real>
inline bool neumann_root_has_constant_kernel(
  const std::vector<Real>& R,
  const std::vector<Real>& v)
{
  const int nb = static_cast<int>(v.size());
  if (nb < 2 || R.size() != (std::size_t)nb * nb)
    throw std::invalid_argument("Neumann root: invalid rank-check dimensions");

  const Real vn = neumann_root_detail::norm2(v);
  if (!(vn > Real(0)))
    throw std::invalid_argument("Neumann root: zero constant trace");

  std::vector<Real> A = R;
  std::vector<Real> s((std::size_t)nb);
  std::vector<Real> U((std::size_t)nb * nb);
  std::vector<Real> VT((std::size_t)nb * nb);

  const lapack_int info = detail::LapackGesdd<Real>::run(
    'A', (lapack_int)nb, (lapack_int)nb,
    A.data(), (lapack_int)nb, s.data(),
    U.data(), (lapack_int)nb,
    VT.data(), (lapack_int)nb);
  if (info != 0)
    throw std::runtime_error("Neumann root: rank-check SVD failed");

  if (!(s.front() > Real(0)))
    throw std::runtime_error("Neumann root: zero root matrix");

  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real rank_rel_tol = Real(1000) * eps * Real(nb);
  const Real rank_abs_tol = rank_rel_tol * s.front();
  if (s.back() > rank_abs_tol)
    return false; // Nonsingular: use ordinary LU.

  if (s[(std::size_t)nb - 2] <= Real(10) * rank_abs_tol)
    throw std::runtime_error("Neumann root: multiple numerical null modes");

  std::vector<Real> Rv((std::size_t)nb, Real(0));
  for (int j = 0; j < nb; ++j)
    for (int i = 0; i < nb; ++i)
      Rv[(std::size_t)i] +=
        R[(std::size_t)i + (std::size_t)nb * j] * v[(std::size_t)j];

  const Real relative_defect =
    neumann_root_detail::norm2(Rv) / (s.front() * vn);
  if (relative_defect > Real(10) * rank_rel_tol)
    throw std::runtime_error(
      "Neumann root: singular null mode is not the constant trace");

  return true;
}

// The continuous c=0 divergence-form operator annihilates constants,
// but the discretely assembled HPS DtN matrix may have R*v != 0 due to
// leaf/merge roundoff. Enforce the known right kernel without changing
// R on the physically chosen zero-mean trace subspace ker(w^T).
//
// R_corrected = R - (R*v) * w^T / (w^T*v).
//
// This correction must only be used when the PDE is known to have
// constants in its kernel. It must NOT be applied to a problem with
// nonzero reaction or another genuinely invertible Neumann operator.
template<int D, class Real>
inline std::vector<Real> neumann_restore_constant_kernel(
  const std::vector<Real>& R,
  const std::vector<Real>& v,
  const std::vector<Real>& w,
  Real defect_rtol = Real(1e-7))
{
  static_assert(D >= 1, "Neumann root requires D >= 1");

  const int nb = static_cast<int>(v.size());
  if (nb < 2 || R.size() != (std::size_t)nb * nb ||
      w.size() != (std::size_t)nb || !(defect_rtol > Real(0)))
    throw std::invalid_argument(
      "Neumann root: invalid constant-kernel correction inputs");

  const Real vn = neumann_root_detail::norm2(v);
  const Real wv = neumann_root_detail::dot(w, v);
  if (!(vn > Real(0)) || !(std::abs(wv) > Real(0)))
    throw std::invalid_argument("Neumann root: invalid constant/gauge vectors");

  std::vector<Real> Rv((std::size_t)nb, Real(0));
  Real norm_R_sq = Real(0);
  for (int j = 0; j < nb; ++j)
    for (int i = 0; i < nb; ++i)
    {
      const Real a = R[(std::size_t)i + (std::size_t)nb * j];
      Rv[(std::size_t)i] += a * v[(std::size_t)j];
      norm_R_sq += a * a;
    }

  const Real norm_R = std::sqrt(norm_R_sq);
  if (!(norm_R > Real(0)))
    throw std::runtime_error("Neumann root: zero root operator");

  const Real defect = neumann_root_detail::norm2(Rv) / (norm_R * vn);
  if (!std::isfinite(defect) || defect > defect_rtol)
  {
    std::ostringstream msg;
    msg << "Neumann root: constant-mode defect=" << defect
        << " exceeds tolerance=" << defect_rtol;
    throw std::runtime_error(msg.str());
  }

  std::vector<Real> corrected = R;
  for (int j = 0; j < nb; ++j)
  {
    const Real scale = w[(std::size_t)j] / wv;
    for (int i = 0; i < nb; ++i)
      corrected[(std::size_t)i + (std::size_t)nb * j] -=
        Rv[(std::size_t)i] * scale;
  }

  return corrected;
}


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

// Physical (unweighted) boundary mean and the represented constant trace.
// Place in the anonymous namespace of jhps_c.cpp, before EllipticSolverImpl.
//
// Output: v is the coefficient vector for u|boundary = 1;
//         w^T lambda is the physical surface mean of the root trace.
//         w^T v = 1 (normalized explicitly).
template<int D, class Real>
inline void build_root_neumann_boundary_mean(
  const jsimplex::Mesh<D,Real>& mesh,
  const jsimplex::Node<D,Real>& root,
  const jsimplex::RefSimplexPrecomp<D,Real>& pre,
  std::vector<Real>& v,
  std::vector<Real>& w)
{
  const int kf = pre.kf;
  const int nf = root.nfaces();
  const int nb = root.nb();
  if (nf <= 0 || kf <= 0 || root.kf != kf || nb != nf * kf)
    throw std::invalid_argument("Neumann gauge: invalid root dimensions");

  v.assign((std::size_t)nb, Real(0));
  w.assign((std::size_t)nb, Real(0));

  // The first canonical face Jacobi polynomial is a nonzero constant.
  // For D=1 the face basis consists of the constant 1.
  const Real phi0 = pre.V_face.front();
  if (!std::isfinite(phi0) || phi0 == Real(0))
    throw std::runtime_error("Neumann gauge: invalid constant face basis");


  // Integrate the *original* canonical face basis against unweighted
  // surface measure, independently of the Jacobi weight kappa_face.
  std::vector<Real> face_moments((std::size_t)kf, Real(0));
  if constexpr (D == 1)
  {
    face_moments[0] = Real(1);
  }
  else
  {
    const int q = std::max(pre.n + 2, 2);
    const int nq = static_cast<int>(
      jsimplex::QuadMapped<D-1,Real>::npoints((unsigned int)q));
    std::vector<Real> points((std::size_t)nq * (D-1));
    std::vector<Real> weights((std::size_t)nq);
    if (jsimplex::QuadMapped<D-1,Real>::build(
          (unsigned int)q, points.data(), weights.data()) != nq)
      throw std::runtime_error("Neumann gauge: unweighted face quadrature failed");

    std::vector<int> alpha_table, tail_deg;
    std::vector<Real> invh;
    jsimplex::Basis<D-1,Real>::build_structures(
      pre.n, pre.kappa_face.data(), alpha_table, tail_deg, invh);

    std::vector<Real> V((std::size_t)nq * kf);
    jsimplex::Basis<D-1,Real>::eval_all(
      points.data(), D-1, 1, nq,
      pre.kappa_face.data(), pre.n,
      alpha_table.data(), tail_deg.data(), invh.data(),
      V.data(), nq, nullptr);

    for (int j = 0; j < kf; ++j)
      for (int p = 0; p < nq; ++p)
        face_moments[(std::size_t)j] +=
          weights[(std::size_t)p] * V[(std::size_t)p + (std::size_t)nq * j];
  }

  Real boundary_measure = Real(0);
  for (int f = 0; f < nf; ++f)
  {
    const auto& key = root.boundary_faces[(std::size_t)f];
    const auto& inc = mesh.incidences(key);
    if (inc.size() != 1)
      throw std::invalid_argument("Neumann gauge: root face is not exterior");

    // face_geom.scale is the physical Jacobian relative to the
    // canonical (D-1)-simplex (for D=1, its value is 1).
    const Real scale = mesh.element(inc[0].element_id)
      .face_geom[(std::size_t)inc[0].local_face_id].scale;
    if (!(scale > Real(0)))
      throw std::runtime_error("Neumann gauge: invalid face measure scale");

    // Root traces store physical face moments.
    v[(std::size_t)f * kf] = scale / phi0;
    
    // Convert those moments to the physical boundary integral.
    for (int j = 0; j < kf; ++j)
      w[(std::size_t)f * kf + j] =
        face_moments[(std::size_t)j];

    // Unweighted integral of 1 over a canonical (D-1)-simplex.
    Real ref_face_measure = Real(1);
    for (int j = 2; j <= D-1; ++j)
      ref_face_measure /= Real(j);
    boundary_measure += scale * ref_face_measure;
  }

  if (!(boundary_measure > Real(0)))
    throw std::runtime_error("Neumann gauge: zero boundary measure");

  for (Real& value : w)
    value /= boundary_measure;

  // Enforce w^T v=1 up to a final floating-point normalization.
  Real wv = Real(0);
  for (int i = 0; i < nb; ++i)
    wv += w[(std::size_t)i] * v[(std::size_t)i];
  if (!(wv > Real(0)) || !std::isfinite(wv))
    throw std::runtime_error("Neumann gauge: invalid constant normalization");
  for (Real& value : w)
    value /= wv;
}

} // namespace jsimplex

#endif // JNEUMANN_ROOT_HH
