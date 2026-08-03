#ifndef JELLIPTIC_HH
#define JELLIPTIC_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <jbasis.hh>
#include <jdetail.hh>
#include <jgeom.hh>
#include <jmat.hh>
#include <jmult.hh>
#include <jprecomp.hh>
#ifdef TIMING
#include <timer.hh>
#endif

namespace jsimplex {

/*
  Nondivergence-form affine-element operator

    L u = sum_{r,s=0}^{D-1} a_rs(x) d_{x_r x_s} u
        + sum_{r=0}^{D-1} b_r(x) d_{x_r} u
        + c(x) u.

  The coefficient fields are supplied as modal coefficients of their affine
  pullbacks in the common residual basis kappa_res.  Physical component
  indexing is retained:

    A[(r*D + s)*Mp2 + alpha]
    b[r*Mp1 + alpha]
    c[alpha].

  A degree value of -1 disables the corresponding differential order.
*/
struct EllipticDegreeSpec
{
  int p2 = -1;
  int p1 = -1;
  int p0 = -1;
};


enum class EllipticResidualPolicy
{
  TrialDegree = 0,            // R = n
  SecondDerivativeDegree = 1 // R = n - 2 (legacy/Poisson range)
};

enum class EllipticMultiplicationAssembler
{
  Quadrature = 0,
  ClenshawColumns = 1
};


#ifdef TIMING
/* Accumulated matrix-free action timings. */
struct EllipticActionTimings
{
  long long forward_calls = 0;
  long long transpose_calls = 0;
  long long forward_principal_multiplier_calls = 0;
  long long forward_first_multiplier_calls = 0;
  long long forward_zero_multiplier_calls = 0;
  long long transpose_principal_multiplier_calls = 0;
  long long transpose_first_multiplier_calls = 0;
  long long transpose_zero_multiplier_calls = 0;

  double forward_total_seconds = 0.0;
  double forward_partial_dag_seconds = 0.0;
  double forward_principal_geometry_seconds = 0.0;
  double forward_principal_clenshaw_seconds = 0.0;
  double forward_first_geometry_seconds = 0.0;
  double forward_first_clenshaw_seconds = 0.0;
  double forward_zero_clenshaw_seconds = 0.0;
  double forward_output_scale_seconds = 0.0;

  double transpose_total_seconds = 0.0;
  double transpose_residual_scale_seconds = 0.0;
  double transpose_principal_clenshaw_seconds = 0.0;
  double transpose_principal_geometry_seconds = 0.0;
  double transpose_first_clenshaw_seconds = 0.0;
  double transpose_first_geometry_seconds = 0.0;
  double transpose_zero_clenshaw_seconds = 0.0;
  double transpose_partial_dag_seconds = 0.0;

  void reset() { *this = EllipticActionTimings{}; }
  void add(const EllipticActionTimings& o)
  {
    forward_calls += o.forward_calls;
    transpose_calls += o.transpose_calls;
    forward_principal_multiplier_calls += o.forward_principal_multiplier_calls;
    forward_first_multiplier_calls += o.forward_first_multiplier_calls;
    forward_zero_multiplier_calls += o.forward_zero_multiplier_calls;
    transpose_principal_multiplier_calls += o.transpose_principal_multiplier_calls;
    transpose_first_multiplier_calls += o.transpose_first_multiplier_calls;
    transpose_zero_multiplier_calls += o.transpose_zero_multiplier_calls;
    forward_total_seconds += o.forward_total_seconds;
    forward_partial_dag_seconds += o.forward_partial_dag_seconds;
    forward_principal_geometry_seconds += o.forward_principal_geometry_seconds;
    forward_principal_clenshaw_seconds += o.forward_principal_clenshaw_seconds;
    forward_first_geometry_seconds += o.forward_first_geometry_seconds;
    forward_first_clenshaw_seconds += o.forward_first_clenshaw_seconds;
    forward_zero_clenshaw_seconds += o.forward_zero_clenshaw_seconds;
    forward_output_scale_seconds += o.forward_output_scale_seconds;
    transpose_total_seconds += o.transpose_total_seconds;
    transpose_residual_scale_seconds += o.transpose_residual_scale_seconds;
    transpose_principal_clenshaw_seconds += o.transpose_principal_clenshaw_seconds;
    transpose_principal_geometry_seconds += o.transpose_principal_geometry_seconds;
    transpose_first_clenshaw_seconds += o.transpose_first_clenshaw_seconds;
    transpose_first_geometry_seconds += o.transpose_first_geometry_seconds;
    transpose_zero_clenshaw_seconds += o.transpose_zero_clenshaw_seconds;
    transpose_partial_dag_seconds += o.transpose_partial_dag_seconds;
  }
};
#endif

template<int D, class Real>
struct EllipticElementCoefficientsView
{
  const Real* A = nullptr; // D*D*dim(Pi_p2), physical (r,s) components
  const Real* b = nullptr; // D*dim(Pi_p1), physical r components
  const Real* c = nullptr; // dim(Pi_p0)
};

template<int D, class Real>
class EllipticPlan;

template<int D, class Real>
class EllipticActionWorkspace;

template<int D, class Real>
class EllipticDenseWorkspace;

// Backward compatibility for existing dense assembly callers. New
// matrix-free code should name EllipticActionWorkspace explicitly.
template<int D, class Real>
using EllipticWorkspace = EllipticDenseWorkspace<D,Real>;

template<int D, class Real>
void jdsimplex_assemble_elliptic_L_int(
  const EllipticPlan<D,Real>& plan,
  EllipticDenseWorkspace<D,Real>& work,
  const Real* BinvT,
  Real detBabs,
  const Real* Lij_ref,
  const Real* Li_ref,
  const Real* L0_ref,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  Real* L_int_out);

template<int D, class Real>
void jdsimplex_assemble_elliptic_L_int_dag(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticDenseWorkspace<D,Real>& work,
  Real* L_int_out);

/*
  Matrix-free affine-element actions.

  Forward:
    y = L_int x,  x has length M and y has length mR.

  Transpose:
    x = L_int^T y, y has length mR and x has length M.

  Outputs are overwritten. The caller owns one EllipticActionWorkspace per
  concurrent application. This type contains no dense assembly matrices.
*/
template<int D, class Real>
void jdsimplex_apply_elliptic(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticActionWorkspace<D,Real>& work,
  const Real* x,
  Real* y);

template<int D, class Real>
void jdsimplex_apply_elliptic_transpose(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticActionWorkspace<D,Real>& work,
  const Real* y,
  Real* x);

namespace elliptic_detail {

template<int D, class Real>
class OwnedClenshawPlan
{
public:
  int p = 0;
  int K = 0;
  int Mp = 0;
  int MK = 0;
  bool scalar_only = false;

  std::vector<int> alpha_p;
  std::array<int*, D> colptr_poly{};
  std::array<int*, D> rowind_poly{};
  std::array<Real*, D> x_poly{};
  std::array<int*, D> colptr_fun{};
  std::array<int*, D> rowind_fun{};
  std::array<Real*, D> x_fun{};
  std::array<CSCView<Real>, D> J_poly_view{};
  std::array<CSCView<Real>, D> J_fun_view{};

  MultByQClenshaw<D,Real> mult;

  OwnedClenshawPlan(const Real* kappa_res,
                    int p_in,
                    int K_in,
                    bool assume_symmetric)
    : p(p_in), K(K_in)
  {
    if (!kappa_res)
      throw std::invalid_argument("OwnedClenshawPlan: null kappa_res");
    if (p < 0 || K < p)
      throw std::invalid_argument("OwnedClenshawPlan: invalid p or K");

    Mp = Basis<D,Real>::dim_Pi(p);
    MK = Basis<D,Real>::dim_Pi(K);
    scalar_only = (p == 0);

    if (scalar_only)
      return;

    std::vector<int> tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(
      p, kappa_res, alpha_p, tail, invh);

    if ((int)alpha_p.size() != Mp * D)
      throw std::runtime_error("OwnedClenshawPlan: bad alpha table size");

    for (int coord = 0; coord < D; ++coord)
    {
      build_coord_csc(
        kappa_res, p, (unsigned int)(p + 1), coord,
        colptr_poly[(std::size_t)coord],
        rowind_poly[(std::size_t)coord],
        x_poly[(std::size_t)coord]);

      build_coord_csc(
        kappa_res, K, (unsigned int)(K + 1), coord,
        colptr_fun[(std::size_t)coord],
        rowind_fun[(std::size_t)coord],
        x_fun[(std::size_t)coord]);

      J_poly_view[(std::size_t)coord] = CSCView<Real>{
        Mp, Mp,
        colptr_poly[(std::size_t)coord],
        rowind_poly[(std::size_t)coord],
        x_poly[(std::size_t)coord]
      };

      J_fun_view[(std::size_t)coord] = CSCView<Real>{
        MK, MK,
        colptr_fun[(std::size_t)coord],
        rowind_fun[(std::size_t)coord],
        x_fun[(std::size_t)coord]
      };
    }

    mult.init(
      p,
      K,
      alpha_p.data(),
      Mp,
      J_poly_view.data(),
      J_fun_view.data(),
      assume_symmetric);
  }

  OwnedClenshawPlan(const OwnedClenshawPlan&) = delete;
  OwnedClenshawPlan& operator=(const OwnedClenshawPlan&) = delete;

  ~OwnedClenshawPlan()
  {
    for (int coord = 0; coord < D; ++coord)
    {
      std::free(colptr_poly[(std::size_t)coord]);
      std::free(rowind_poly[(std::size_t)coord]);
      std::free(x_poly[(std::size_t)coord]);
      std::free(colptr_fun[(std::size_t)coord]);
      std::free(rowind_fun[(std::size_t)coord]);
      std::free(x_fun[(std::size_t)coord]);
    }
  }

private:
  static void build_coord_csc(const Real* kappa,
                              int degree,
                              unsigned int nquad,
                              int coord,
                              int*& colptr,
                              int*& rowind,
                              Real*& values)
  {
    const std::size_t nnz = JMat<D,Real>::build_pruned_csc(
      degree, kappa, nquad, coord, &colptr, &rowind, &values);

    if (nnz > (std::size_t)std::numeric_limits<int>::max())
      throw std::overflow_error("OwnedClenshawPlan: CSC nnz overflow");

    if (!colptr || (nnz > 0 && (!rowind || !values)))
      throw std::runtime_error("OwnedClenshawPlan: JMat CSC build failed");
  }
};

} // namespace elliptic_detail

template<int D, class Real>
class EllipticPlan
{
public:
  static_assert(D >= 1, "EllipticPlan requires D>=1");

  int n = 0;
  int M = 0;
  int m2 = 0; // active second-derivative image dim(Pi_{n-2})
  int m1 = 0; // active first-derivative image dim(Pi_{n-1})
  int residual_degree = 0;
  int mR = 0;
  int q_mult = 0;
  EllipticDegreeSpec degrees{};
  EllipticResidualPolicy residual_policy =
    EllipticResidualPolicy::TrialDegree;
  EllipticMultiplicationAssembler multiplication_assembler =
    EllipticMultiplicationAssembler::Quadrature;
  std::array<Real, D + 1> kappa_res{};
  Real phi0_res = Real(0);

  explicit EllipticPlan(
    const RefSimplexPrecomp<D,Real>& pre,
    const EllipticDegreeSpec& degree_spec,
    bool assume_symmetric = true,
    EllipticResidualPolicy residual_policy_in =
      EllipticResidualPolicy::TrialDegree,
    EllipticMultiplicationAssembler multiplication_assembler_in =
      EllipticMultiplicationAssembler::Quadrature,
    bool build_clenshaw_plans = true)
    : EllipticPlan(
        pre.n,
        pre.kappa_res.data(),
        degree_spec,
        assume_symmetric,
        residual_policy_in,
        multiplication_assembler_in,
        build_clenshaw_plans)
  {}

  EllipticPlan(
    int n_in,
    const Real* kappa_res_in,
    const EllipticDegreeSpec& degree_spec,
    bool assume_symmetric = true,
    EllipticResidualPolicy residual_policy_in =
      EllipticResidualPolicy::TrialDegree,
    EllipticMultiplicationAssembler multiplication_assembler_in =
      EllipticMultiplicationAssembler::Quadrature,
    bool build_clenshaw_plans = true)
    : n(n_in),
      degrees(degree_spec),
      residual_policy(residual_policy_in),
      multiplication_assembler(multiplication_assembler_in)
  {
    if (!kappa_res_in)
      throw std::invalid_argument("EllipticPlan: null kappa_res");
    if (n < 2)
      throw std::invalid_argument("EllipticPlan: require n>=2");
    if (degrees.p2 < -1 || degrees.p1 < -1 || degrees.p0 < -1)
      throw std::invalid_argument("EllipticPlan: degrees must be >= -1");
    if (degrees.p2 < 0 && degrees.p1 < 0 && degrees.p0 < 0)
      throw std::invalid_argument("EllipticPlan: all differential orders disabled");

    for (int i = 0; i <= D; ++i)
      kappa_res[(std::size_t)i] = kappa_res_in[i];

    M = Basis<D,Real>::dim_Pi(n);
    m2 = Basis<D,Real>::dim_Pi(n - 2);
    m1 = Basis<D,Real>::dim_Pi(n - 1);
    residual_degree =
      residual_policy == EllipticResidualPolicy::TrialDegree
      ? n
      : n - 2;
    mR = Basis<D,Real>::dim_Pi(residual_degree);

    phi0_res = evaluate_constant_basis_value();

    if (build_clenshaw_plans ||
        multiplication_assembler ==
          EllipticMultiplicationAssembler::ClenshawColumns)
    {
      plan_id_[0] = register_order_plan(2, degrees.p2, assume_symmetric);
      plan_id_[1] = register_order_plan(1, degrees.p1, assume_symmetric);
      plan_id_[2] = register_order_plan(0, degrees.p0, assume_symmetric);
    }

    if (multiplication_assembler ==
        EllipticMultiplicationAssembler::Quadrature)
    {
      int maximum_triple_degree = 0;
      int maximum_basis_degree = residual_degree;
      for (int order = 0; order <= 2; ++order)
      {
        const int p = coefficient_degree(order);
        if (p < 0)
          continue;
        const int N = derivative_degree(order);
        maximum_triple_degree = std::max(
          maximum_triple_degree,
          residual_degree + N + p);
        maximum_basis_degree = std::max(
          maximum_basis_degree,
          std::max(N, p));
      }

      // A q-point Gauss-Jacobi rule is exact through degree 2q-1.
      // This is the direct analogue of q_vol=n+1 (q_pad=1): choose the
      // smallest q with 2q-1 >= the triple-product degree bound.
      q_mult = maximum_triple_degree / 2 + 1;
      quadrature_plan_ = std::make_unique<
        MultByQQuadraturePlan<D,Real>>(
          kappa_res.data(),
          maximum_basis_degree,
          q_mult);
    }
  }

  int num_plans() const
  {
    return (int)plans_.size();
  }

  int plan_id_for_order(int order) const
  {
    if (order == 2) return plan_id_[0];
    if (order == 1) return plan_id_[1];
    if (order == 0) return plan_id_[2];
    throw std::invalid_argument("EllipticPlan: order must be 0, 1, or 2");
  }

  int coefficient_degree(int order) const
  {
    if (order == 2) return degrees.p2;
    if (order == 1) return degrees.p1;
    if (order == 0) return degrees.p0;
    throw std::invalid_argument("EllipticPlan: order must be 0, 1, or 2");
  }

  int coefficient_size(int order) const
  {
    const int p = coefficient_degree(order);
    return (p < 0) ? 0 : Basis<D,Real>::dim_Pi(p);
  }

  int derivative_degree(int order) const
  {
    if (order == 2) return n - 2;
    if (order == 1) return n - 1;
    if (order == 0) return n;
    throw std::invalid_argument("EllipticPlan: order must be 0, 1, or 2");
  }

  // Internal plan-entry access used by EllipticWorkspace and the assembly
  // kernel. Entries are immutable after construction.
  const elliptic_detail::OwnedClenshawPlan<D,Real>& entry(int id) const
  {
    return *plans_.at((std::size_t)id);
  }


  const MultByQQuadraturePlan<D,Real>& quadrature_plan() const
  {
    if (!quadrature_plan_)
      throw std::logic_error(
        "EllipticPlan: quadrature multiplication plan is unavailable");
    return *quadrature_plan_;
  }

private:
  std::vector<std::unique_ptr<elliptic_detail::OwnedClenshawPlan<D,Real>>> plans_;
  std::unique_ptr<MultByQQuadraturePlan<D,Real>> quadrature_plan_;
  std::array<int, 3> plan_id_{{-1, -1, -1}};

  friend class EllipticActionWorkspace<D,Real>;
  friend class EllipticDenseWorkspace<D,Real>;

  int register_order_plan(int order,
                          int p,
                          bool assume_symmetric)
  {
    if (p < 0)
      return -1;

    const int N = derivative_degree(order);
    const int K = N + p;

    for (int id = 0; id < (int)plans_.size(); ++id)
    {
      const auto& P = *plans_[(std::size_t)id];
      if (P.p == p && P.K == K)
        return id;
    }

    plans_.push_back(
      std::make_unique<elliptic_detail::OwnedClenshawPlan<D,Real>>(
        kappa_res.data(), p, K, assume_symmetric));
    return (int)plans_.size() - 1;
  }

  Real evaluate_constant_basis_value() const
  {
    std::vector<int> alpha;
    std::vector<int> tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(
      0, kappa_res.data(), alpha, tail, invh);

    std::array<Real, D> x{};
    for (int i = 0; i < D; ++i)
      x[(std::size_t)i] = Real(1) / Real(D + 1);

    Real value = Real(0);
    Basis<D,Real>::eval_all(
      x.data(), D, 1, 1,
      kappa_res.data(), 0,
      alpha.data(), tail.data(), invh.data(),
      &value, 1, nullptr);
    return value;
  }
};

template<int D, class Real>
class EllipticActionWorkspace
{
public:
  static constexpr bool has_dense_assembly_storage = false;

  explicit EllipticActionWorkspace(
    const EllipticPlan<D,Real>& plan)
    : owner_(&plan)
  {
    mult_workspaces_.resize(
      (std::size_t)plan.num_plans());

    int max_MK = 0;
    for (int id = 0; id < plan.num_plans(); ++id)
    {
      const auto& P = plan.entry(id);
      max_MK = std::max(max_MK, P.MK);
      if (!P.scalar_only)
      {
        mult_workspaces_[(std::size_t)id].init(
          P.p,
          P.MK);
      }
    }

    cK_.resize((std::size_t)max_MK);
    yK_.resize((std::size_t)max_MK);

    const std::size_t dag_partial_value_count =
      (std::size_t)plan.M
      + (std::size_t)D * (std::size_t)plan.m1
      + (std::size_t)(D * (D + 1) / 2)
        * (std::size_t)plan.m2;

    dag_partial_column_.resize(
      dag_partial_value_count);
    dag_partial_adjoint_.resize(
      dag_partial_value_count);

    const int max_partial_size =
      std::max(
        plan.M,
        std::max(plan.m1, plan.m2));
    physical_derivative_.resize(
      (std::size_t)max_partial_size);
    scaled_residual_.resize(
      (std::size_t)plan.mR);
  }

  bool compatible(
    const EllipticPlan<D,Real>& plan) const
  {
    return owner_ == &plan;
  }

#ifdef TIMING
  const EllipticActionTimings& timings() const { return timings_; }
  void reset_timings() { timings_.reset(); }
#endif

protected:
  const EllipticPlan<D,Real>* owner_ = nullptr;

  // Clenshaw action state shared by forward, transpose, and dense diagnostic
  // assembly. These vectors are all O(dim Pi_K), not O(M^2).
  std::vector<MultByQClenshawWorkspace<D,Real>>
    mult_workspaces_;
  std::vector<Real> cK_;
  std::vector<Real> yK_;

  // Matrix-free derivative/promotion DAG state.
  typename RefSimplexPrecomp<D,Real>::PartialWorkspace
    dag_partial_work_;
  std::vector<Real> dag_partial_column_;
  std::vector<Real> dag_partial_adjoint_;
  std::vector<Real> physical_derivative_;
  std::vector<Real> scaled_residual_;
#ifdef TIMING
  EllipticActionTimings timings_{};
#endif

  friend void jdsimplex_apply_elliptic<D,Real>(
    const RefSimplexPrecomp<D,Real>&,
    const DSimplexGeom<D,Real>&,
    const EllipticPlan<D,Real>&,
    const EllipticElementCoefficientsView<D,Real>&,
    EllipticActionWorkspace<D,Real>&,
    const Real*,
    Real*);

  friend void jdsimplex_apply_elliptic_transpose<D,Real>(
    const RefSimplexPrecomp<D,Real>&,
    const DSimplexGeom<D,Real>&,
    const EllipticPlan<D,Real>&,
    const EllipticElementCoefficientsView<D,Real>&,
    EllipticActionWorkspace<D,Real>&,
    const Real*,
    Real*);

  friend class EllipticDenseWorkspace<D,Real>;
};


template<int D, class Real>
class EllipticDenseWorkspace
  : public EllipticActionWorkspace<D,Real>
{
public:
  static constexpr bool has_dense_assembly_storage = true;

  explicit EllipticDenseWorkspace(
    const EllipticPlan<D,Real>& plan)
    : EllipticActionWorkspace<D,Real>(plan)
  {
    int max_MN = 0;
    for (int order = 0; order <= 2; ++order)
    {
      if (plan.coefficient_degree(order) >= 0)
      {
        max_MN = std::max(
          max_MN,
          Basis<D,Real>::dim_Pi(
            plan.derivative_degree(order)));
      }
    }

    Mq_.resize(
      (std::size_t)plan.mR
      * (std::size_t)max_MN);
    Dphys_.resize(
      (std::size_t)plan.M
      * (std::size_t)plan.M);

    const std::size_t dag_partial_value_count =
      (std::size_t)plan.M
      + (std::size_t)D * (std::size_t)plan.m1
      + (std::size_t)(D * (D + 1) / 2)
        * (std::size_t)plan.m2;

    dag_source_.resize((std::size_t)plan.M);
    dag_partial_matrix_.resize(
      dag_partial_value_count
      * (std::size_t)plan.M);
  }

private:
  // Dense-only coefficient multiplication and physical derivative matrices.
  std::vector<Real> Mq_;
  std::vector<Real> Dphys_;
  MultByQQuadratureWorkspace<Real> quadrature_work_;

  // Dense-only identity-column materialization of all unique DAG partials.
  std::vector<Real> dag_source_;
  std::vector<Real> dag_partial_matrix_;

  friend void jdsimplex_assemble_elliptic_L_int<D,Real>(
    const EllipticPlan<D,Real>&,
    EllipticDenseWorkspace<D,Real>&,
    const Real*,
    Real,
    const Real*,
    const Real*,
    const Real*,
    const EllipticElementCoefficientsView<D,Real>&,
    Real*);

  friend void jdsimplex_assemble_elliptic_L_int_dag<D,Real>(
    const RefSimplexPrecomp<D,Real>&,
    const DSimplexGeom<D,Real>&,
    const EllipticPlan<D,Real>&,
    const EllipticElementCoefficientsView<D,Real>&,
    EllipticDenseWorkspace<D,Real>&,
    Real*);
};

/*
  Raw-array assembly entry point.

  Storage:
    BinvT       D x D column-major, equal to B^{-T}
    Lij_ref     M x M x D x D, Fortran order
    Li_ref      M x M x D, Fortran order
    L0_ref      M x M, column-major
    L_int_out   mR x M, column-major

  The physical derivative contractions are

    H_rs = sum_{i,j} (B^{-1})_{i r}(B^{-1})_{j s} Lij_ref(i,j),
    G_r  = sum_i     (B^{-1})_{i r} Li_ref(i).

  Since BinvT = B^{-T}, (B^{-1})_{i r} = BinvT(r,i).
*/
template<int D, class Real>
void jdsimplex_assemble_elliptic_L_int(
  const EllipticPlan<D,Real>& plan,
  EllipticDenseWorkspace<D,Real>& work,
  const Real* BinvT,
  Real detBabs,
  const Real* Lij_ref,
  const Real* Li_ref,
  const Real* L0_ref,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  Real* L_int_out)
{
  if (!BinvT || !Lij_ref || !Li_ref || !L0_ref || !L_int_out)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: null array");
  if (!work.compatible(plan))
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: incompatible workspace");
  if (!(detBabs > Real(0)) || !std::isfinite((double)detBabs))
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: invalid detBabs");
  if (plan.degrees.p2 >= 0 && !coeffs.A)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: null A coefficients");
  if (plan.degrees.p1 >= 0 && !coeffs.b)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: null b coefficients");
  if (plan.degrees.p0 >= 0 && !coeffs.c)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: null c coefficients");

  const int M = plan.M;
  const int mR = plan.mR;
  const int m2 = plan.m2;
  const int m1 = plan.m1;
  std::fill(
    L_int_out,
    L_int_out + (std::size_t)mR * (std::size_t)M,
    Real(0));

  auto build_restricted_mult = [&](int order, const Real* q, int MN) -> const Real*
  {
    Real* Mq = work.Mq_.data();
    std::fill(Mq, Mq + (std::size_t)mR * (std::size_t)MN, Real(0));

    if (plan.multiplication_assembler ==
        EllipticMultiplicationAssembler::Quadrature)
    {
      assemble_restricted_mult_quadrature<D,Real>(
        plan.quadrature_plan(),
        q,
        plan.coefficient_degree(order),
        plan.derivative_degree(order),
        plan.residual_degree,
        work.quadrature_work_,
        Mq);
      return Mq;
    }

    const int id = plan.plan_id_for_order(order);
    if (id < 0)
      throw std::logic_error("build_restricted_mult: missing Clenshaw plan");
    const auto& P = plan.entry(id);

    if (P.scalar_only)
    {
      // q(x) = q_0 phi_0(x), so multiplication is by q_0*phi_0.
      const Real scalar = q[0] * plan.phi0_res;
      const int ndiag = std::min(mR, MN);
      for (int j = 0; j < ndiag; ++j)
        Mq[(std::size_t)j + (std::size_t)mR * (std::size_t)j] = scalar;
      return Mq;
    }

    Real* cK = work.cK_.data();
    Real* yK = work.yK_.data();
    auto& mult_work = work.mult_workspaces_[(std::size_t)id];

    for (int col = 0; col < MN; ++col)
    {
      std::fill(cK, cK + P.MK, Real(0));
      cK[col] = Real(1);
      P.mult.apply(q, cK, yK, mult_work);
      const int active_rows = std::min(mR, P.MK);
      for (int row = 0; row < active_rows; ++row)
        Mq[(std::size_t)row + (std::size_t)mR * (std::size_t)col] = yK[row];
    }
    return Mq;
  };

  // Principal part: sum_{r,s} M_{a_rs} H_rs.
  if (plan.degrees.p2 >= 0)
  {
    const int Mp2 = plan.coefficient_size(2);
    const int MN = m2;

    for (int r = 0; r < D; ++r)
    {
      for (int s = 0; s < D; ++s)
      {
        Real* Hrs = work.Dphys_.data(); // compact MN x M, column-major
        for (int col = 0; col < M; ++col)
        {
          for (int row = 0; row < MN; ++row)
          {
            Real acc = Real(0);
            for (int i = 0; i < D; ++i)
            {
              const Real Cir = BinvT[(std::size_t)r + (std::size_t)D * i];
              for (int j = 0; j < D; ++j)
              {
                const Real Cjs = BinvT[(std::size_t)s + (std::size_t)D * j];
                const std::size_t block =
                  (std::size_t)i + (std::size_t)D * (std::size_t)j;
                const Real* Lij =
                  Lij_ref + (std::size_t)M * (std::size_t)M * block;
                acc += Cir * Cjs * Lij[
                  (std::size_t)row + (std::size_t)M * (std::size_t)col];
              }
            }
            Hrs[(std::size_t)row + (std::size_t)MN * (std::size_t)col] = acc;
          }
        }

        const Real* q = coeffs.A
          + ((std::size_t)r * D + (std::size_t)s) * (std::size_t)Mp2;
        const Real* Mq = build_restricted_mult(2, q, MN);

        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          mR,
          M,
          MN,
          Real(1),
          Mq,
          mR,
          Hrs,
          MN,
          Real(1),
          L_int_out,
          mR);
      }
    }
  }

  // First-order part: sum_r M_{b_r} G_r.
  if (plan.degrees.p1 >= 0)
  {
    const int Mp1 = plan.coefficient_size(1);
    const int MN = m1;

    for (int r = 0; r < D; ++r)
    {
      Real* Gr = work.Dphys_.data(); // compact MN x M, column-major
      for (int col = 0; col < M; ++col)
      {
        for (int row = 0; row < MN; ++row)
        {
          Real acc = Real(0);
          for (int i = 0; i < D; ++i)
          {
            const Real Cir = BinvT[(std::size_t)r + (std::size_t)D * i];
            const Real* Li =
              Li_ref + (std::size_t)M * (std::size_t)M * (std::size_t)i;
            acc += Cir * Li[
              (std::size_t)row + (std::size_t)M * (std::size_t)col];
          }
          Gr[(std::size_t)row + (std::size_t)MN * (std::size_t)col] = acc;
        }
      }

      const Real* q = coeffs.b + (std::size_t)r * (std::size_t)Mp1;
      const Real* Mq = build_restricted_mult(1, q, MN);

      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        mR,
        M,
        MN,
        Real(1),
        Mq,
        mR,
        Gr,
        MN,
        Real(1),
        L_int_out,
        mR);
    }
  }

  // Zero-order part: M_c L0_ref.
  if (plan.degrees.p0 >= 0)
  {
    const int MN = M;
    const Real* Mq = build_restricted_mult(0, coeffs.c, MN);

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      mR,
      M,
      MN,
      Real(1),
      Mq,
      mR,
      L0_ref,
      M,
      Real(1),
      L_int_out,
      mR);
  }

  for (std::size_t k = 0; k < (std::size_t)mR * (std::size_t)M; ++k)
    L_int_out[k] *= detBabs;
}


/*
  DAG-backed affine-element assembly.

  The derivative/promotion DAG is applied to the coefficient-space identity
  once per source column. Its unique partial channels are stored in compact
  column-major blocks:

    order zero:  dim(Pi_n)     x M,
    order one:   D blocks of dim(Pi_{n-1}) x M,
    order two:   D(D+1)/2 unique blocks of dim(Pi_{n-2}) x M.

  Physical derivative contractions and the existing coefficient
  multiplication plans then assemble the same dense L_int consumed by Leaf
  and the current dense LSMR/HPS interface. The padded dense L0_ref, Li_ref,
  and Lij_ref arrays are not used.
*/
template<int D, class Real>
void jdsimplex_assemble_elliptic_L_int_dag(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticDenseWorkspace<D,Real>& work,
  Real* L_int_out)
{
  using Precomp = RefSimplexPrecomp<D,Real>;
  using PartialPlan = typename Precomp::PartialPlan;

  if (!L_int_out)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: null output");
  if (!geom.valid)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: invalid geometry");
  if (!work.compatible(plan))
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: incompatible workspace");
  if (pre.n != plan.n || pre.M != plan.M)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: precompute/plan mismatch");
  if (pre.partials.empty() || pre.partial_value_count == 0)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: missing partial DAG");
  if (!(geom.detBabs > Real(0)) ||
      !std::isfinite((double)geom.detBabs))
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: invalid detBabs");
  if (plan.degrees.p2 >= 0 && !coeffs.A)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: null A coefficients");
  if (plan.degrees.p1 >= 0 && !coeffs.b)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: null b coefficients");
  if (plan.degrees.p0 >= 0 && !coeffs.c)
    throw std::invalid_argument(
      "jdsimplex_assemble_elliptic_L_int_dag: null c coefficients");

  const int M = plan.M;
  const int mR = plan.mR;
  const int m2 = plan.m2;
  const int m1 = plan.m1;

  if (work.dag_source_.size() < (std::size_t)M ||
      work.dag_partial_column_.size() < pre.partial_value_count ||
      work.dag_partial_matrix_.size() <
        pre.partial_value_count * (std::size_t)M)
    throw std::runtime_error(
      "jdsimplex_assemble_elliptic_L_int_dag: undersized workspace");

  std::fill(
    L_int_out,
    L_int_out + (std::size_t)mR * (std::size_t)M,
    Real(0));

  std::array<int, D> alpha_zero{};
  const PartialPlan& zero_plan =
    pre.partial_plan(alpha_zero);

  std::array<const PartialPlan*, D> first_plan{};
  for (int axis = 0; axis < D; ++axis)
  {
    std::array<int, D> alpha{};
    alpha[(std::size_t)axis] = 1;
    first_plan[(std::size_t)axis] =
      &pre.partial_plan(alpha);
  }

  std::array<const PartialPlan*, D * D> second_plan{};
  for (int first_axis = 0; first_axis < D; ++first_axis)
  {
    for (int second_axis = 0; second_axis < D; ++second_axis)
    {
      std::array<int, D> alpha{};
      ++alpha[(std::size_t)first_axis];
      ++alpha[(std::size_t)second_axis];
      second_plan[
        (std::size_t)first_axis +
        (std::size_t)D * (std::size_t)second_axis] =
        &pre.partial_plan(alpha);
    }
  }

  // Materialize the unique compact partial blocks from the DAG.
  for (int column = 0; column < M; ++column)
  {
    std::fill(
      work.dag_source_.begin(),
      work.dag_source_.begin() + M,
      Real(0));
    work.dag_source_[(std::size_t)column] = Real(1);

    pre.apply_partials(
      work.dag_source_.data(),
      work.dag_partial_column_.data(),
      work.dag_partial_work_);

    for (const PartialPlan& partial : pre.partials)
    {
      Real* block =
        work.dag_partial_matrix_.data()
        + partial.value_offset * (std::size_t)M;

      const Real* source =
        work.dag_partial_column_.data()
        + partial.value_offset;

      for (int row = 0; row < partial.size; ++row)
      {
        block[
          (std::size_t)row
          + (std::size_t)partial.size
            * (std::size_t)column] =
          source[(std::size_t)row];
      }
    }
  }

  auto partial_block =
    [&](const PartialPlan& partial) -> const Real*
  {
    return
      work.dag_partial_matrix_.data()
      + partial.value_offset * (std::size_t)M;
  };

  auto build_restricted_mult =
    [&](int order, const Real* q, int MN) -> const Real*
  {
    Real* Mq = work.Mq_.data();
    std::fill(
      Mq,
      Mq + (std::size_t)mR * (std::size_t)MN,
      Real(0));

    if (plan.multiplication_assembler ==
        EllipticMultiplicationAssembler::Quadrature)
    {
      assemble_restricted_mult_quadrature<D,Real>(
        plan.quadrature_plan(),
        q,
        plan.coefficient_degree(order),
        plan.derivative_degree(order),
        plan.residual_degree,
        work.quadrature_work_,
        Mq);
      return Mq;
    }

    const int id = plan.plan_id_for_order(order);
    if (id < 0)
      throw std::logic_error(
        "jdsimplex_assemble_elliptic_L_int_dag: missing Clenshaw plan");
    const auto& P = plan.entry(id);

    if (P.scalar_only)
    {
      const Real scalar = q[0] * plan.phi0_res;
      const int ndiag = std::min(mR, MN);
      for (int j = 0; j < ndiag; ++j)
      {
        Mq[
          (std::size_t)j
          + (std::size_t)mR * (std::size_t)j] =
          scalar;
      }
      return Mq;
    }

    Real* cK = work.cK_.data();
    Real* yK = work.yK_.data();
    auto& mult_work =
      work.mult_workspaces_[(std::size_t)id];

    for (int col = 0; col < MN; ++col)
    {
      std::fill(cK, cK + P.MK, Real(0));
      cK[col] = Real(1);
      P.mult.apply(q, cK, yK, mult_work);

      const int active_rows = std::min(mR, P.MK);
      for (int row = 0; row < active_rows; ++row)
      {
        Mq[
          (std::size_t)row
          + (std::size_t)mR * (std::size_t)col] =
          yK[row];
      }
    }

    return Mq;
  };

  // Principal part: sum_{r,s} M_{a_rs} H_rs.
  if (plan.degrees.p2 >= 0)
  {
    const int Mp2 = plan.coefficient_size(2);
    const int MN = m2;

    for (int r = 0; r < D; ++r)
    {
      for (int s = 0; s < D; ++s)
      {
        Real* Hrs =
          work.Dphys_.data();

        for (int col = 0; col < M; ++col)
        {
          for (int row = 0; row < MN; ++row)
          {
            Real value = Real(0);

            for (int i = 0; i < D; ++i)
            {
              const Real Cir =
                geom.BinvT[
                  (std::size_t)r
                  + (std::size_t)D * (std::size_t)i];

              for (int j = 0; j < D; ++j)
              {
                const Real Cjs =
                  geom.BinvT[
                    (std::size_t)s
                    + (std::size_t)D * (std::size_t)j];

                const PartialPlan& partial =
                  *second_plan[
                    (std::size_t)i
                    + (std::size_t)D * (std::size_t)j];

                const Real* Dij =
                  partial_block(partial);

                value +=
                  Cir * Cjs *
                  Dij[
                    (std::size_t)row
                    + (std::size_t)MN
                      * (std::size_t)col];
              }
            }

            Hrs[
              (std::size_t)row
              + (std::size_t)MN * (std::size_t)col] =
              value;
          }
        }

        const Real* q =
          coeffs.A
          + (
              (std::size_t)r * (std::size_t)D
              + (std::size_t)s
            ) * (std::size_t)Mp2;

        const Real* Mq =
          build_restricted_mult(2, q, MN);

        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          mR,
          M,
          MN,
          Real(1),
          Mq,
          mR,
          Hrs,
          MN,
          Real(1),
          L_int_out,
          mR);
      }
    }
  }

  // First-order part: sum_r M_{b_r} G_r.
  if (plan.degrees.p1 >= 0)
  {
    const int Mp1 = plan.coefficient_size(1);
    const int MN = m1;

    for (int r = 0; r < D; ++r)
    {
      Real* Gr =
        work.Dphys_.data();

      for (int col = 0; col < M; ++col)
      {
        for (int row = 0; row < MN; ++row)
        {
          Real value = Real(0);

          for (int i = 0; i < D; ++i)
          {
            const Real Cir =
              geom.BinvT[
                (std::size_t)r
                + (std::size_t)D * (std::size_t)i];

            const PartialPlan& partial =
              *first_plan[(std::size_t)i];
            const Real* Di =
              partial_block(partial);

            value +=
              Cir *
              Di[
                (std::size_t)row
                + (std::size_t)MN * (std::size_t)col];
          }

          Gr[
            (std::size_t)row
            + (std::size_t)MN * (std::size_t)col] =
            value;
        }
      }

      const Real* q =
        coeffs.b
        + (std::size_t)r * (std::size_t)Mp1;

      const Real* Mq =
        build_restricted_mult(1, q, MN);

      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        mR,
        M,
        MN,
        Real(1),
        Mq,
        mR,
        Gr,
        MN,
        Real(1),
        L_int_out,
        mR);
    }
  }

  // Zero-order part: M_c D^0.
  if (plan.degrees.p0 >= 0)
  {
    const int MN = M;
    const Real* Mq =
      build_restricted_mult(0, coeffs.c, MN);
    const Real* D0 =
      partial_block(zero_plan);

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      mR,
      M,
      MN,
      Real(1),
      Mq,
      mR,
      D0,
      MN,
      Real(1),
      L_int_out,
      mR);
  }

  for (std::size_t entry = 0;
       entry < (std::size_t)mR * (std::size_t)M;
       ++entry)
  {
    L_int_out[entry] *= geom.detBabs;
  }
}


/*
  Matrix-free forward action.

  The derivative/promotion DAG produces all unique reference partial vectors.
  Geometry contractions are performed on vectors, and coefficient
  multiplication is applied directly through the Clenshaw plans. No partial
  matrix, L_int matrix, or identity-column materialization is formed.
*/
template<int D, class Real>
void jdsimplex_apply_elliptic(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticActionWorkspace<D,Real>& work,
  const Real* x,
  Real* y)
{
  using Precomp = RefSimplexPrecomp<D,Real>;
  using PartialPlan = typename Precomp::PartialPlan;

  if (!x || !y)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: null vector");
  if (!geom.valid)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: invalid geometry");
  if (!work.compatible(plan))
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: incompatible workspace");
  if (pre.n != plan.n || pre.M != plan.M)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: precompute/plan mismatch");
  if (pre.partials.empty() || pre.partial_value_count == 0)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: missing partial DAG");
  if (!(geom.detBabs > Real(0)) ||
      !std::isfinite((double)geom.detBabs))
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: invalid detBabs");
  if (plan.degrees.p2 >= 0 && !coeffs.A)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: null A coefficients");
  if (plan.degrees.p1 >= 0 && !coeffs.b)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: null b coefficients");
  if (plan.degrees.p0 >= 0 && !coeffs.c)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic: null c coefficients");

  const int M = plan.M;
  const int mR = plan.mR;
  const int m2 = plan.m2;
  const int m1 = plan.m1;

  if (work.dag_partial_column_.size() < pre.partial_value_count ||
      work.physical_derivative_.size() <
        (std::size_t)std::max(M, std::max(m1, m2)))
    throw std::runtime_error(
      "jdsimplex_apply_elliptic: undersized workspace");

#ifdef TIMING
  timer forward_total_timer; forward_total_timer.tic();
#endif
  std::fill(y, y + mR, Real(0));
#ifdef TIMING
  timer forward_partial_timer; forward_partial_timer.tic();
#endif
  pre.apply_partials(
    x,
    work.dag_partial_column_.data(),
    work.dag_partial_work_);
#ifdef TIMING
  work.timings_.forward_partial_dag_seconds += forward_partial_timer.toc();
#endif

  std::array<int, D> alpha_zero{};
  const PartialPlan& zero_plan =
    pre.partial_plan(alpha_zero);

  std::array<const PartialPlan*, D> first_plan{};
  for (int axis = 0; axis < D; ++axis)
  {
    std::array<int, D> alpha{};
    alpha[(std::size_t)axis] = 1;
    first_plan[(std::size_t)axis] =
      &pre.partial_plan(alpha);
  }

  std::array<const PartialPlan*, D * D> second_plan{};
  for (int first_axis = 0; first_axis < D; ++first_axis)
  {
    for (int second_axis = 0;
         second_axis < D;
         ++second_axis)
    {
      std::array<int, D> alpha{};
      ++alpha[(std::size_t)first_axis];
      ++alpha[(std::size_t)second_axis];
      second_plan[
        (std::size_t)first_axis
        + (std::size_t)D * (std::size_t)second_axis] =
        &pre.partial_plan(alpha);
    }
  }

  auto apply_restricted_multiplier =
    [&](int order,
        const Real* q,
        const Real* input,
        int input_size)
  {
    const int id = plan.plan_id_for_order(order);
    if (id < 0)
      throw std::logic_error(
        "jdsimplex_apply_elliptic: disabled order");

    const auto& P = plan.entry(id);

    if (P.scalar_only)
    {
      const Real scalar = q[0] * plan.phi0_res;
      const int count = std::min(mR, input_size);
      for (int row = 0; row < count; ++row)
        y[row] += scalar * input[row];
      return;
    }

    Real* cK = work.cK_.data();
    Real* yK = work.yK_.data();
    std::fill(cK, cK + P.MK, Real(0));
    std::copy(input, input + input_size, cK);

    auto& mult_work =
      work.mult_workspaces_[(std::size_t)id];
    P.mult.apply(q, cK, yK, mult_work);

    const int active_rows = std::min(mR, P.MK);
    for (int row = 0; row < active_rows; ++row)
      y[row] += yK[row];
  };

  // Principal part.
  if (plan.degrees.p2 >= 0)
  {
    const int Mp2 = plan.coefficient_size(2);

    for (int r = 0; r < D; ++r)
    {
      for (int s = 0; s < D; ++s)
      {
#ifdef TIMING
        timer forward_principal_geometry_timer; forward_principal_geometry_timer.tic();
#endif
        Real* Hrs = work.physical_derivative_.data();
        std::fill(Hrs, Hrs + m2, Real(0));

        for (int i = 0; i < D; ++i)
        {
          const Real Cir =
            geom.BinvT[
              (std::size_t)r
              + (std::size_t)D * (std::size_t)i];

          for (int j = 0; j < D; ++j)
          {
            const Real Cjs =
              geom.BinvT[
                (std::size_t)s
                + (std::size_t)D * (std::size_t)j];

            const PartialPlan& partial =
              *second_plan[
                (std::size_t)i
                + (std::size_t)D * (std::size_t)j];
            const Real* Dij =
              work.dag_partial_column_.data()
              + partial.value_offset;
            const Real scale = Cir * Cjs;

            for (int row = 0; row < m2; ++row)
              Hrs[row] += scale * Dij[row];
          }
        }

#ifdef TIMING
        work.timings_.forward_principal_geometry_seconds += forward_principal_geometry_timer.toc();
#endif
        const Real* q =
          coeffs.A
          + (
              (std::size_t)r * (std::size_t)D
              + (std::size_t)s
            ) * (std::size_t)Mp2;
#ifdef TIMING
        timer forward_principal_clenshaw_timer; forward_principal_clenshaw_timer.tic();
#endif
        apply_restricted_multiplier(2, q, Hrs, m2);
#ifdef TIMING
        work.timings_.forward_principal_clenshaw_seconds += forward_principal_clenshaw_timer.toc();
        ++work.timings_.forward_principal_multiplier_calls;
#endif
      }
    }
  }

  // First-order part.
  if (plan.degrees.p1 >= 0)
  {
    const int Mp1 = plan.coefficient_size(1);

    for (int r = 0; r < D; ++r)
    {
#ifdef TIMING
      timer forward_first_geometry_timer; forward_first_geometry_timer.tic();
#endif
      Real* Gr = work.physical_derivative_.data();
      std::fill(Gr, Gr + m1, Real(0));

      for (int i = 0; i < D; ++i)
      {
        const Real Cir =
          geom.BinvT[
            (std::size_t)r
            + (std::size_t)D * (std::size_t)i];

        const PartialPlan& partial =
          *first_plan[(std::size_t)i];
        const Real* Di =
          work.dag_partial_column_.data()
          + partial.value_offset;

        for (int row = 0; row < m1; ++row)
          Gr[row] += Cir * Di[row];
      }

#ifdef TIMING
      work.timings_.forward_first_geometry_seconds += forward_first_geometry_timer.toc();
#endif
      const Real* q =
        coeffs.b
        + (std::size_t)r * (std::size_t)Mp1;
#ifdef TIMING
      timer forward_first_clenshaw_timer; forward_first_clenshaw_timer.tic();
#endif
      apply_restricted_multiplier(1, q, Gr, m1);
#ifdef TIMING
      work.timings_.forward_first_clenshaw_seconds += forward_first_clenshaw_timer.toc();
      ++work.timings_.forward_first_multiplier_calls;
#endif
    }
  }

  // Zero-order part.
  if (plan.degrees.p0 >= 0)
  {
    const Real* D0 =
      work.dag_partial_column_.data()
      + zero_plan.value_offset;
#ifdef TIMING
    timer forward_zero_clenshaw_timer; forward_zero_clenshaw_timer.tic();
#endif
    apply_restricted_multiplier(0, coeffs.c, D0, M);
#ifdef TIMING
    work.timings_.forward_zero_clenshaw_seconds += forward_zero_clenshaw_timer.toc();
    ++work.timings_.forward_zero_multiplier_calls;
#endif
  }
#ifdef TIMING
  timer forward_scale_timer; forward_scale_timer.tic();
#endif
  for (int row = 0; row < mR; ++row)
    y[row] *= geom.detBabs;
#ifdef TIMING
  work.timings_.forward_output_scale_seconds += forward_scale_timer.toc();
  work.timings_.forward_total_seconds += forward_total_timer.toc();
  ++work.timings_.forward_calls;
#endif
}


/*
  Exact transpose of jdsimplex_apply_elliptic.

  Multiplication by a real modal coefficient field is self-adjoint in the
  common orthonormal residual basis. Therefore the transpose of the
  rectangular restricted multiplier is obtained by embedding the residual
  adjoint into the full Clenshaw space, applying the same multiplier, and
  restricting to the derivative input space.
*/
template<int D, class Real>
void jdsimplex_apply_elliptic_transpose(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticActionWorkspace<D,Real>& work,
  const Real* y,
  Real* x)
{
  using Precomp = RefSimplexPrecomp<D,Real>;
  using PartialPlan = typename Precomp::PartialPlan;

  if (!y || !x)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: null vector");
  if (!geom.valid)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: invalid geometry");
  if (!work.compatible(plan))
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: incompatible workspace");
  if (pre.n != plan.n || pre.M != plan.M)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: precompute/plan mismatch");
  if (pre.partials.empty() || pre.partial_value_count == 0)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: missing partial DAG");
  if (!(geom.detBabs > Real(0)) ||
      !std::isfinite((double)geom.detBabs))
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: invalid detBabs");
  if (plan.degrees.p2 >= 0 && !coeffs.A)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: null A coefficients");
  if (plan.degrees.p1 >= 0 && !coeffs.b)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: null b coefficients");
  if (plan.degrees.p0 >= 0 && !coeffs.c)
    throw std::invalid_argument(
      "jdsimplex_apply_elliptic_transpose: null c coefficients");

  const int M = plan.M;
  const int mR = plan.mR;
  const int m2 = plan.m2;
  const int m1 = plan.m1;

  if (work.dag_partial_adjoint_.size() <
        pre.partial_value_count ||
      work.physical_derivative_.size() <
        (std::size_t)std::max(M, std::max(m1, m2)) ||
      work.scaled_residual_.size() < (std::size_t)mR)
    throw std::runtime_error(
      "jdsimplex_apply_elliptic_transpose: undersized workspace");
#ifdef TIMING
  timer transpose_total_timer; transpose_total_timer.tic();
  timer transpose_residual_scale_timer; transpose_residual_scale_timer.tic();
#endif
  std::fill(
    work.dag_partial_adjoint_.begin(),
    work.dag_partial_adjoint_.begin()
      + pre.partial_value_count,
    Real(0));

  for (int row = 0; row < mR; ++row)
    work.scaled_residual_[(std::size_t)row] =
      geom.detBabs * y[row];
#ifdef TIMING
  work.timings_.transpose_residual_scale_seconds += transpose_residual_scale_timer.toc();
#endif
  std::array<int, D> alpha_zero{};
  const PartialPlan& zero_plan =
    pre.partial_plan(alpha_zero);

  std::array<const PartialPlan*, D> first_plan{};
  for (int axis = 0; axis < D; ++axis)
  {
    std::array<int, D> alpha{};
    alpha[(std::size_t)axis] = 1;
    first_plan[(std::size_t)axis] =
      &pre.partial_plan(alpha);
  }

  std::array<const PartialPlan*, D * D> second_plan{};
  for (int first_axis = 0; first_axis < D; ++first_axis)
  {
    for (int second_axis = 0;
         second_axis < D;
         ++second_axis)
    {
      std::array<int, D> alpha{};
      ++alpha[(std::size_t)first_axis];
      ++alpha[(std::size_t)second_axis];
      second_plan[
        (std::size_t)first_axis
        + (std::size_t)D * (std::size_t)second_axis] =
        &pre.partial_plan(alpha);
    }
  }

  auto apply_restricted_multiplier_transpose =
    [&](int order,
        const Real* q,
        Real* output,
        int output_size)
  {
    const int id = plan.plan_id_for_order(order);
    if (id < 0)
      throw std::logic_error(
        "jdsimplex_apply_elliptic_transpose: disabled order");

    const auto& P = plan.entry(id);

    if (P.scalar_only)
    {
      const Real scalar = q[0] * plan.phi0_res;
      std::fill(output, output + output_size, Real(0));
      const int count = std::min(mR, output_size);
      for (int row = 0; row < count; ++row)
      {
        output[row] =
          scalar *
          work.scaled_residual_[(std::size_t)row];
      }
      return;
    }

    Real* cK = work.cK_.data();
    Real* yK = work.yK_.data();
    std::fill(cK, cK + P.MK, Real(0));
    std::copy(
      work.scaled_residual_.begin(),
      work.scaled_residual_.begin()
        + std::min(mR, P.MK),
      cK);

    auto& mult_work =
      work.mult_workspaces_[(std::size_t)id];
    P.mult.apply(q, cK, yK, mult_work);

    std::copy(yK, yK + output_size, output);
  };

  // Principal-part transpose.
  if (plan.degrees.p2 >= 0)
  {
    const int Mp2 = plan.coefficient_size(2);

    for (int r = 0; r < D; ++r)
    {
      for (int s = 0; s < D; ++s)
      {
        Real* Hrs_adjoint =
          work.physical_derivative_.data();

        const Real* q =
          coeffs.A
          + (
              (std::size_t)r * (std::size_t)D
              + (std::size_t)s
            ) * (std::size_t)Mp2;

#ifdef TIMING
        timer transpose_principal_clenshaw_timer; transpose_principal_clenshaw_timer.tic();
#endif
        apply_restricted_multiplier_transpose(
          2,
          q,
          Hrs_adjoint,
          m2);
#ifdef TIMING
        work.timings_.transpose_principal_clenshaw_seconds += transpose_principal_clenshaw_timer.toc();
        ++work.timings_.transpose_principal_multiplier_calls;
        timer transpose_principal_geometry_timer; transpose_principal_geometry_timer.tic();
#endif
        for (int i = 0; i < D; ++i)
        {
          const Real Cir =
            geom.BinvT[
              (std::size_t)r
              + (std::size_t)D * (std::size_t)i];

          for (int j = 0; j < D; ++j)
          {
            const Real Cjs =
              geom.BinvT[
                (std::size_t)s
                + (std::size_t)D * (std::size_t)j];

            const PartialPlan& partial =
              *second_plan[
                (std::size_t)i
                + (std::size_t)D * (std::size_t)j];
            Real* Dij_adjoint =
              work.dag_partial_adjoint_.data()
              + partial.value_offset;
            const Real scale = Cir * Cjs;

            for (int row = 0; row < m2; ++row)
              Dij_adjoint[row] +=
                scale * Hrs_adjoint[row];
          }
        }
#ifdef TIMING
        work.timings_.transpose_principal_geometry_seconds += transpose_principal_geometry_timer.toc();
#endif
      }
    }
  }

  // First-order transpose.
  if (plan.degrees.p1 >= 0)
  {
    const int Mp1 = plan.coefficient_size(1);

    for (int r = 0; r < D; ++r)
    {
      Real* Gr_adjoint =
        work.physical_derivative_.data();

      const Real* q =
        coeffs.b
        + (std::size_t)r * (std::size_t)Mp1;

#ifdef TIMING
      timer transpose_first_clenshaw_timer; transpose_first_clenshaw_timer.tic();
#endif
      apply_restricted_multiplier_transpose(
        1,
        q,
        Gr_adjoint,
        m1);
#ifdef TIMING
      work.timings_.transpose_first_clenshaw_seconds += transpose_first_clenshaw_timer.toc();
      ++work.timings_.transpose_first_multiplier_calls;
      timer transpose_first_geometry_timer; transpose_first_geometry_timer.tic();
#endif
      for (int i = 0; i < D; ++i)
      {
        const Real Cir =
          geom.BinvT[
            (std::size_t)r
            + (std::size_t)D * (std::size_t)i];

        const PartialPlan& partial =
          *first_plan[(std::size_t)i];
        Real* Di_adjoint =
          work.dag_partial_adjoint_.data()
          + partial.value_offset;

        for (int row = 0; row < m1; ++row)
          Di_adjoint[row] += Cir * Gr_adjoint[row];
      }
#ifdef TIMING
      work.timings_.transpose_first_geometry_seconds += transpose_first_geometry_timer.toc();
#endif
    }
  }

  // Zero-order transpose.
  if (plan.degrees.p0 >= 0)
  {
    Real* D0_adjoint =
      work.physical_derivative_.data();

#ifdef TIMING
    timer transpose_zero_clenshaw_timer; transpose_zero_clenshaw_timer.tic();
#endif
    apply_restricted_multiplier_transpose(
      0,
      coeffs.c,
      D0_adjoint,
      M);
#ifdef TIMING
    work.timings_.transpose_zero_clenshaw_seconds += transpose_zero_clenshaw_timer.toc();
    ++work.timings_.transpose_zero_multiplier_calls;
#endif
    Real* zero_adjoint =
      work.dag_partial_adjoint_.data()
      + zero_plan.value_offset;
    for (int row = 0; row < M; ++row)
      zero_adjoint[row] += D0_adjoint[row];
  }

#ifdef TIMING
  timer transpose_partial_timer; transpose_partial_timer.tic();
#endif
  pre.apply_partials_transpose(
    work.dag_partial_adjoint_.data(),
    x,
    work.dag_partial_work_);
#ifdef TIMING
  work.timings_.transpose_partial_dag_seconds += transpose_partial_timer.toc();
  work.timings_.transpose_total_seconds += transpose_total_timer.toc();
  ++work.timings_.transpose_calls;
#endif
}


/* Convenience overload for direct C++ use from Leaf. */
template<int D, class Real>
void jdsimplex_assemble_elliptic_L_int(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  const EllipticPlan<D,Real>& plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticDenseWorkspace<D,Real>& work,
  Real* L_int_out)
{
  if (!geom.valid)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: invalid geometry");
  if (pre.n != plan.n || pre.M != plan.M)
    throw std::invalid_argument("jdsimplex_assemble_elliptic_L_int: precompute/plan mismatch");

  jdsimplex_assemble_elliptic_L_int<D,Real>(
    plan,
    work,
    geom.BinvT.data(),
    geom.detBabs,
    pre.Lij_ref.data(),
    pre.Li_ref.data(),
    pre.L0_ref.data(),
    coeffs,
    L_int_out);
}

} // namespace jsimplex

#endif // JELLIPTIC_HH
