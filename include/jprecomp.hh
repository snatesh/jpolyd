#ifndef JPRECOMP_HH
#define JPRECOMP_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <jbasis.hh>
#include <jquad_tprod.hh>
#include <jkmat.hh>
#include <jdmat.hh>
#include <jgeom.hh>
#include <jperms.hh>
#include <jdetail.hh>

namespace jsimplex {

template<int D, class Real>
class RefSimplexPrecomp
{
public:
  static_assert(D >= 1, "RefSimplexPrecomp requires D>=1");

  struct CSCBlock
  {
    int rows = 0;
    int cols = 0;
    std::vector<int> colptr;
    std::vector<int> rowind;
    std::vector<Real> values;

    std::size_t nnz() const { return values.size(); }
  };

  // Degree-n CSC structure generated from a degree-independent DMat/KMat
  // delta stencil. Numerical values are filled separately for each shifted
  // factor. Equal signatures are interned automatically within each operator
  // family, so multiple axes/parameters may reference the same structure.
  struct SharedOperatorStructure
  {
    int rows = 0;
    int cols = 0;
    int degree = 0;
    std::string signature;
    std::vector<int> colptr;
    std::vector<int> rowind;

    std::size_t nnz() const { return rowind.size(); }
  };

  // Integer Jacobi-family shift relative to the source kappa.
  struct ShiftState
  {
    std::array<int, D + 1> delta{};

    bool operator==(const ShiftState& other) const
    {
      return delta == other.delta;
    }
  };

  // One numerical sparse factor. The CSC structure is owned by either
  // d_structures or k_structures; only the values vary with the shifted
  // source kappa.
  struct SparseFactor
  {
    int structure_index = -1;
    int degree = 0;
    int coordinate = 0;
    ShiftState source_shift{};
    std::vector<Real> values;
  };

  struct FactorLane
  {
    int source_partial = -1;
    int target_partial = -1;
    int factor_index = -1;
  };

  // Lanes in a group share one finite-degree CSC structure. Values are packed
  // entry-major: values[pos * lanes.size() + lane].
  struct BatchedFactorGroup
  {
    int stage = 0;
    int structure_index = -1;
    std::vector<FactorLane> lanes;
    std::vector<Real> values;
  };

  struct PartialPlan
  {
    std::array<int, D> alpha{};
    int order = 0;
    int degree = 0;
    int size = 0;
    int parent_partial = -1;
    int derivative_factor = -1;
    std::vector<int> promotion_factors;
    std::size_t value_offset = 0;
  };

  // Caller-owned storage keeps the immutable precompute thread-safe.
  struct PartialWorkspace
  {
    std::vector<Real> derivative_values;
    std::vector<Real> promotion_a;
    std::vector<Real> promotion_b;

    void resize(std::size_t count)
    {
      derivative_values.resize(count);
      promotion_a.resize(count);
      promotion_b.resize(count);
    }
  };

  struct DAGCompatibilityReport
  {
    Real L0_relative = Real(0);
    Real Li_relative = Real(0);
    Real Lij_relative = Real(0);

    Real maximum() const
    {
      return std::max(
        L0_relative,
        std::max(Li_relative, Lij_relative));
    }
  };

  int n = 0;
  int q_pad = 2;
  int q_vol = 0;
  int q_face = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int nface = D + 1;
  int nsigma = factorial_int(D);
  int nq_vol = 0;
  int nq_face = 0;

  std::array<Real, D + 1> kappa{};
  std::array<Real, D + 1> kappa_res{};
  std::array<Real, D> kappa_face{};

  // Persistent stencil cache directory used by DMat/KMat. Missing keyed
  // stencils are discovered once and written here; subsequent precomputes
  // load them without running stencil stabilization.
  std::string stencil_folder = "stencils";

  // Interned finite-degree structures. The legacy axis/parameter arrays point
  // to the degree-n structures. The DAG also uses degree n-1 and n-2
  // structures through private (degree,key) maps.
  std::vector<SharedOperatorStructure> d_structures;
  std::vector<SharedOperatorStructure> k_structures;
  std::array<int, D> d_axis_structure{};
  std::array<int, D + 1> k_parameter_structure{};

  // Order-two derivative/promotion DAG. Existing dense Lij_ref/Li_ref/L0_ref
  // remain the downstream HPS interface during compatibility validation.
  std::vector<SparseFactor> dag_d_factors;
  std::vector<SparseFactor> dag_k_factors;
  std::vector<PartialPlan> partials;
  std::vector<BatchedFactorGroup> derivative_batches;
  std::vector<BatchedFactorGroup> promotion_forward_batches;
  std::vector<BatchedFactorGroup> promotion_transpose_batches;
  std::size_t partial_value_count = 0;

  // Volume quadrature/basis in source-kappa convention.
  std::vector<Real> X_vol;     // row-major nq_vol x D (for basis evaluator)
  std::vector<Real> W_vol;     // nq_vol
  std::vector<Real> V_vol;     // col-major nq_vol x M
  std::vector<Real> V_vol_int; // col-major nq_vol x m_int

  // Volume quadrature/basis in PDE-residual kappa+2 convention.
  // The full degree-n basis is stored so the same data can test/build
  // second-, first-, and zero-order residual operators.
  std::vector<Real> X_res;      // row-major nq_vol x D
  std::vector<Real> W_res;      // nq_vol
  std::vector<Real> V_res;      // col-major nq_vol x M

  // Canonical face quadrature/basis. For D=1, Y_face is empty, W_face=[1], V_face=[1].
  std::vector<Real> Y_face;    // row-major nq_face x (D-1) (for basis evaluator)
  std::vector<Real> W_face;    // nq_face, intrinsic canonical face weights
  std::vector<Real> V_face;    // col-major nq_face x kf

  // Embedded reference-face measure scale from canonical (D-1)-simplex to the
  // corresponding face of the reference D-simplex.
  std::vector<Real> face_ref_scale; // nface

  // Dense reference operator blocks, all Fortran/column-major in public layout.
  // Lij_ref:    M  x M x D x D
  // Li_ref:     M  x M x D
  // L0_ref:     M  x M
  // T_ref:      kf x M x nsigma x nface
  // Fgrad_ref:  kf x M x D x nsigma x nface
  // Mface_ref:  kf x kf x nface
  std::vector<Real> Lij_ref;
  std::vector<Real> Li_ref;
  std::vector<Real> L0_ref;
  std::vector<Real> T_ref;
  std::vector<Real> Fgrad_ref;
  std::vector<Real> Mface_ref;

  // Pruned CSC copies of the boundary reference blocks.  Dense storage and
  // all existing dense interfaces remain unchanged.
  //
  // T_ref_csc block order:       sigma + nsigma * face
  // Fgrad_ref_csc block order:   axis + D * (sigma + nsigma * face)
  // Mface_ref_csc block order:   face
  std::vector<CSCBlock> T_ref_csc;
  std::vector<CSCBlock> Fgrad_ref_csc;
  std::vector<CSCBlock> Mface_ref_csc;

  // Relative Frobenius threshold used only for the boundary CSC copies.
  Real boundary_csc_rel_prune = Real(1.0e-12);

  const CSCBlock& T_ref_csc_block(int sigma, int face) const
  {
    if (sigma < 0 || sigma >= nsigma || face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: T_ref_csc block index");
    }
    return T_ref_csc[(std::size_t)sigma + (std::size_t)nsigma * face];
  }

  const CSCBlock& Fgrad_ref_csc_block(int axis, int sigma, int face) const
  {
    if (axis < 0 || axis >= D || sigma < 0 || sigma >= nsigma ||
        face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: Fgrad_ref_csc block index");
    }
    return Fgrad_ref_csc[(std::size_t)axis + (std::size_t)D *
      ((std::size_t)sigma + (std::size_t)nsigma * face)];
  }

  const CSCBlock& Mface_ref_csc_block(int face) const
  {
    if (face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: Mface_ref_csc block index");
    }
    return Mface_ref_csc[(std::size_t)face];
  }

  int partial_index(const std::array<int, D>& alpha) const
  {
    for (std::size_t index = 0; index < partials.size(); ++index)
    {
      if (partials[index].alpha == alpha)
      {
        return static_cast<int>(index);
      }
    }
    return -1;
  }

  const PartialPlan& partial_plan(
    const std::array<int, D>& alpha) const
  {
    const int index = partial_index(alpha);
    if (index < 0)
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: partial multi-index");
    }
    return partials[static_cast<std::size_t>(index)];
  }

  // Apply the stacked order-zero, order-one, and order-two reference partial
  // map. Every channel has already been promoted into kappa_res = kappa + 2.
  // plan.value_offset and plan.size locate each channel in partial_values.
  void apply_partials(
    const Real* x,
    Real* partial_values,
    PartialWorkspace& workspace) const
  {
    if (!x || !partial_values)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null partial apply data");
    }
    if (partials.empty())
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: partial DAG is empty");
    }

    workspace.resize(partial_value_count);
    std::fill(
      workspace.derivative_values.begin(),
      workspace.derivative_values.end(),
      Real(0));
    std::fill(
      workspace.promotion_a.begin(),
      workspace.promotion_a.end(),
      Real(0));
    std::fill(
      workspace.promotion_b.begin(),
      workspace.promotion_b.end(),
      Real(0));

    const PartialPlan& root = partials.front();
    std::copy(
      x,
      x + root.size,
      workspace.derivative_values.begin() + root.value_offset);

    for (const BatchedFactorGroup& group : derivative_batches)
    {
      apply_derivative_group_forward(
        group,
        workspace.derivative_values,
        workspace.derivative_values);
    }

    workspace.promotion_a = workspace.derivative_values;

    for (const BatchedFactorGroup& group : promotion_forward_batches)
    {
      std::vector<Real>& input =
        (group.stage % 2 == 0)
          ? workspace.promotion_a
          : workspace.promotion_b;
      std::vector<Real>& output =
        (group.stage % 2 == 0)
          ? workspace.promotion_b
          : workspace.promotion_a;

      for (const FactorLane& lane : group.lanes)
      {
        const PartialPlan& plan =
          partials[static_cast<std::size_t>(lane.target_partial)];
        std::fill(
          output.begin() + plan.value_offset,
          output.begin() + plan.value_offset + plan.size,
          Real(0));
      }
      apply_promotion_group_forward(group, input, output);
    }

    for (const PartialPlan& plan : partials)
    {
      const std::vector<Real>& source =
        (plan.promotion_factors.size() % 2 == 0)
          ? workspace.promotion_a
          : workspace.promotion_b;
      std::copy(
        source.begin() + plan.value_offset,
        source.begin() + plan.value_offset + plan.size,
        partial_values + plan.value_offset);
    }
  }

  // Exact transpose of apply_partials. The input uses the same flat channel
  // layout. Contributions from all derivative children accumulate at parents.
  void apply_partials_transpose(
    const Real* partial_adjoints,
    Real* x,
    PartialWorkspace& workspace) const
  {
    if (!partial_adjoints || !x)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null partial transpose data");
    }
    if (partials.empty())
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: partial DAG is empty");
    }

    workspace.resize(partial_value_count);
    std::fill(
      workspace.derivative_values.begin(),
      workspace.derivative_values.end(),
      Real(0));
    std::fill(
      workspace.promotion_a.begin(),
      workspace.promotion_a.end(),
      Real(0));
    std::fill(
      workspace.promotion_b.begin(),
      workspace.promotion_b.end(),
      Real(0));
    std::copy(
      partial_adjoints,
      partial_adjoints + partial_value_count,
      workspace.promotion_a.begin());

    for (const BatchedFactorGroup& group : promotion_transpose_batches)
    {
      std::vector<Real>& input =
        (group.stage % 2 == 0)
          ? workspace.promotion_a
          : workspace.promotion_b;
      std::vector<Real>& output =
        (group.stage % 2 == 0)
          ? workspace.promotion_b
          : workspace.promotion_a;

      for (const FactorLane& lane : group.lanes)
      {
        const PartialPlan& plan =
          partials[static_cast<std::size_t>(lane.target_partial)];
        std::fill(
          output.begin() + plan.value_offset,
          output.begin() + plan.value_offset + plan.size,
          Real(0));
      }
      apply_promotion_group_transpose(group, input, output);
    }

    for (const PartialPlan& plan : partials)
    {
      const std::vector<Real>& source =
        (plan.promotion_factors.size() % 2 == 0)
          ? workspace.promotion_a
          : workspace.promotion_b;
      std::copy(
        source.begin() + plan.value_offset,
        source.begin() + plan.value_offset + plan.size,
        workspace.derivative_values.begin() + plan.value_offset);
    }

    for (auto batch = derivative_batches.rbegin();
         batch != derivative_batches.rend();
         ++batch)
    {
      apply_derivative_group_transpose(
        *batch,
        workspace.derivative_values,
        workspace.derivative_values);
    }

    const PartialPlan& root = partials.front();
    std::copy(
      workspace.derivative_values.begin() + root.value_offset,
      workspace.derivative_values.begin() + root.value_offset + root.size,
      x);
  }

  // Materialize the DAG into the existing dense public layouts without
  // modifying the stored Lij_ref/Li_ref/L0_ref arrays.
  void materialize_partials_from_dag(
    std::vector<Real>& Lij_out,
    std::vector<Real>& Li_out,
    std::vector<Real>& L0_out) const
  {
    if (partials.empty())
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: partial DAG is empty");
    }

    Lij_out.assign(
      static_cast<std::size_t>(M) * M * D * D,
      Real(0));
    Li_out.assign(
      static_cast<std::size_t>(M) * M * D,
      Real(0));
    L0_out.assign(
      static_cast<std::size_t>(M) * M,
      Real(0));

    std::array<int, D> zero_alpha{};
    const PartialPlan& zero_plan = partial_plan(zero_alpha);

    std::array<int, D> first_index{};
    std::array<std::array<int, D>, D> first_alpha{};
    for (int axis = 0; axis < D; ++axis)
    {
      first_alpha[static_cast<std::size_t>(axis)].fill(0);
      first_alpha[static_cast<std::size_t>(axis)]
                 [static_cast<std::size_t>(axis)] = 1;
      first_index[static_cast<std::size_t>(axis)] =
        partial_index(first_alpha[static_cast<std::size_t>(axis)]);
    }

    std::array<std::array<int, D>, D> second_index{};
    for (int first_axis = 0; first_axis < D; ++first_axis)
    {
      for (int second_axis = 0; second_axis < D; ++second_axis)
      {
        std::array<int, D> alpha{};
        ++alpha[static_cast<std::size_t>(first_axis)];
        ++alpha[static_cast<std::size_t>(second_axis)];
        second_index[static_cast<std::size_t>(first_axis)]
                    [static_cast<std::size_t>(second_axis)] =
          partial_index(alpha);
      }
    }

    std::vector<Real> x(static_cast<std::size_t>(M), Real(0));
    std::vector<Real> channels(partial_value_count, Real(0));
    PartialWorkspace workspace;

    for (int col = 0; col < M; ++col)
    {
      std::fill(x.begin(), x.end(), Real(0));
      x[static_cast<std::size_t>(col)] = Real(1);
      apply_partials(x.data(), channels.data(), workspace);

      for (int row = 0; row < zero_plan.size; ++row)
      {
        L0_out[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(M) * col] =
          channels[zero_plan.value_offset + static_cast<std::size_t>(row)];
      }

      for (int axis = 0; axis < D; ++axis)
      {
        const int plan_index =
          first_index[static_cast<std::size_t>(axis)];
        if (plan_index < 0)
        {
          throw std::runtime_error(
            "RefSimplexPrecomp: missing first partial plan");
        }
        const PartialPlan& plan =
          partials[static_cast<std::size_t>(plan_index)];
        Real* block =
          Li_out.data() +
          static_cast<std::size_t>(M) * M * axis;
        for (int row = 0; row < plan.size; ++row)
        {
          block[static_cast<std::size_t>(row) +
                static_cast<std::size_t>(M) * col] =
            channels[plan.value_offset + static_cast<std::size_t>(row)];
        }
      }

      for (int first_axis = 0; first_axis < D; ++first_axis)
      {
        for (int second_axis = 0; second_axis < D; ++second_axis)
        {
          const int plan_index =
            second_index[static_cast<std::size_t>(first_axis)]
                        [static_cast<std::size_t>(second_axis)];
          if (plan_index < 0)
          {
            throw std::runtime_error(
              "RefSimplexPrecomp: missing second partial plan");
          }
          const PartialPlan& plan =
            partials[static_cast<std::size_t>(plan_index)];
          Real* block =
            Lij_out.data() +
            static_cast<std::size_t>(M) * M *
              (static_cast<std::size_t>(first_axis) +
               static_cast<std::size_t>(D) * second_axis);
          for (int row = 0; row < plan.size; ++row)
          {
            block[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(M) * col] =
              channels[plan.value_offset + static_cast<std::size_t>(row)];
          }
        }
      }
    }
  }

  DAGCompatibilityReport dag_compatibility_report() const
  {
    std::vector<Real> Lij_dag;
    std::vector<Real> Li_dag;
    std::vector<Real> L0_dag;
    materialize_partials_from_dag(Lij_dag, Li_dag, L0_dag);

    DAGCompatibilityReport report;
    report.L0_relative = relative_frobenius_error(L0_dag, L0_ref);
    report.Li_relative = relative_frobenius_error(Li_dag, Li_ref);
    report.Lij_relative = relative_frobenius_error(Lij_dag, Lij_ref);
    return report;
  }

  RefSimplexPrecomp(int n_in,
                    int q_pad_in,
                    int q_vol_in,
                    int q_face_in,
                    const Real* kappa_in,
                    const std::string& stencil_folder_in = "stencils")
  {
    if (!kappa_in) { throw std::invalid_argument("RefSimplexPrecomp: null kappa"); }
    if (n_in < 2) { throw std::invalid_argument("RefSimplexPrecomp: require n>=2"); }

    n = n_in;
    stencil_folder = stencil_folder_in.empty()
      ? std::string("stencils")
      : stencil_folder_in;
    q_pad = (q_pad_in > 0) ? q_pad_in : 2;
    q_vol = (q_vol_in > 0) ? q_vol_in : (n + q_pad);
    q_face = (q_face_in > 0) ? q_face_in : q_vol;
    if (q_face > q_vol) { throw std::invalid_argument("RefSimplexPrecomp: require q_face <= q_vol"); }

    for (int i = 0; i <= D; ++i)
    {
      kappa[(std::size_t)i] = kappa_in[i];
      kappa_res[(std::size_t)i] = kappa_in[i] + Real(2);
    }

    // Current common-face convention: all faces use source kappa[0:D].
    dsimplex_common_face_kappa<D,Real>(kappa.data(), kappa_face.data());

    M = Basis<D,Real>::dim_Pi(n);
    m_int = Basis<D,Real>::dim_Pi(n - 2);
    if constexpr (D == 1)
    {
      kf = 1;
      nq_face = 1;
    }
    else
    {
      kf = Basis<D - 1,Real>::dim_Pi(n);
      nq_face = (int)QuadMapped<D - 1,Real>::npoints((unsigned int)q_face);
    }
    nq_vol = (int)QuadMapped<D,Real>::npoints((unsigned int)q_vol);

    build_volume_data();
    build_residual_data();
    build_face_data();
    build_operator_stencil_data();
    build_partial_dag();
    build_derivative_factor_data();
    build_second_partials();
    build_first_partials();
    build_zero_partials();
    build_face_operator_data();
    build_face_sparse_data();
  }

private:
  // Stage-A numerical factors: dense row-major materializations assembled
  // from shared stencil structures. These preserve existing public dense
  // operators while eliminating repeated stencil discovery.
  std::vector<Real> D1_factor_rm;
  std::vector<Real> D2_factor_rm;
  std::vector<std::array<Real, D + 1>> derivative_kappa_1;
  std::map<std::array<int, D + 1>, std::vector<Real>>
    residual_promotion_cache;

  // Finite-degree structural lookup used by the DAG.
  std::map<std::pair<int, int>, int> dag_d_structure_index;
  std::map<std::pair<int, int>, int> dag_k_structure_index;

  void build_volume_data()
  {
    X_vol.assign((std::size_t)nq_vol * D, (Real)0);
    W_vol.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa.data(), X_vol.data(), W_vol.data());
    if (built != nq_vol) { throw std::runtime_error("RefSimplexPrecomp: volume quadrature build failed"); }

    std::vector<int> alpha, tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha, tail, invh);

    V_vol.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n,
      alpha.data(), tail.data(), invh.data(), V_vol.data(), nq_vol, nullptr);

    std::vector<int> alpha_i, tail_i;
    std::vector<Real> invh_i;
    Basis<D,Real>::build_structures(n - 2, kappa.data(), alpha_i, tail_i, invh_i);

    V_vol_int.assign((std::size_t)nq_vol * m_int, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n - 2,
      alpha_i.data(), tail_i.data(), invh_i.data(), V_vol_int.data(), nq_vol, nullptr);
  }

  void build_residual_data()
  {
    X_res.assign((std::size_t)nq_vol * D, (Real)0);
    W_res.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa_res.data(), X_res.data(), W_res.data());
    if (built != nq_vol)
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: residual quadrature build failed");
    }

    std::vector<int> alpha_res, tail_res;
    std::vector<Real> invh_res;
    Basis<D,Real>::build_structures(
      n, kappa_res.data(), alpha_res, tail_res, invh_res);

    V_res.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_res.data(), D, 1, nq_vol, kappa_res.data(), n,
      alpha_res.data(), tail_res.data(), invh_res.data(),
      V_res.data(), nq_vol, nullptr);
  }

  void build_face_data()
  {
    if constexpr (D == 1)
    {
      Y_face.clear();
      W_face.assign(1, (Real)1);
      V_face.assign(1, (Real)1);
    }
    else
    {
      Y_face.assign((std::size_t)nq_face * (D - 1), (Real)0);
      W_face.assign((std::size_t)nq_face, (Real)0);
      const int built = QuadMapped<D - 1,Real>::build_kappa(
        (unsigned int)q_face, kappa_face.data(), Y_face.data(), W_face.data());
      if (built != nq_face) { throw std::runtime_error("RefSimplexPrecomp: face quadrature build failed"); }

      std::vector<int> alpha_f, tail_f;
      std::vector<Real> invh_f;
      Basis<D - 1,Real>::build_structures(n, kappa_face.data(), alpha_f, tail_f, invh_f);

      V_face.assign((std::size_t)nq_face * kf, (Real)0);
      Basis<D - 1,Real>::eval_all(
        Y_face.data(), D - 1, 1, nq_face, kappa_face.data(), n,
        alpha_f.data(), tail_f.data(), invh_f.data(), V_face.data(), nq_face, nullptr);
    }

    face_ref_scale.assign((std::size_t)nface, (Real)0);
    for (int face_id = 0; face_id < nface; ++face_id)
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);
      face_ref_scale[(std::size_t)face_id] =
        dsimplex_reference_face_scale_from_vertex_ids<D,Real>(fv);
    }
  }

  static void copy_csc_raw_structure(
    int rows,
    int cols,
    int degree,
    const std::string& signature,
    int* colptr_raw,
    int* rowind_raw,
    std::size_t nnz,
    SharedOperatorStructure& out)
  {
    std::unique_ptr<int, decltype(&std::free)> colptr_guard(
      colptr_raw, &std::free);
    std::unique_ptr<int, decltype(&std::free)> rowind_guard(
      rowind_raw, &std::free);

    out.rows = rows;
    out.cols = cols;
    out.degree = degree;
    out.signature = signature;
    out.colptr.assign(
      colptr_raw,
      colptr_raw + static_cast<std::size_t>(cols) + 1);
    if (nnz > 0)
    {
      out.rowind.assign(rowind_raw, rowind_raw + nnz);
    }
  }

  void build_operator_stencil_data()
  {
    d_axis_structure.fill(-1);
    k_parameter_structure.fill(-1);
    d_structures.clear();
    k_structures.clear();
    dag_d_structure_index.clear();
    dag_k_structure_index.clear();

    std::map<std::pair<std::string, int>, int>
      d_signature_degree_index;
    std::map<std::pair<std::string, int>, int>
      k_signature_degree_index;

    // One persistent delta-stencil load/discovery per derivative axis.
    for (int axis = 0; axis < D; ++axis)
    {
      DMatStencil stencil{};
      DMat<D,Real>::load_or_discover_natural_stencil(
        static_cast<unsigned int>(q_vol),
        kappa.data(),
        axis,
        &stencil,
        stencil_folder);

      const std::string signature =
        DMat<D,Real>::stencil_signature(stencil);

      for (int degree = n; degree >= n - 1; --degree)
      {
        const std::pair<std::string, int> intern_key{
          signature, degree};
        int structure_index = -1;
        const auto found =
          d_signature_degree_index.find(intern_key);

        if (found != d_signature_degree_index.end())
        {
          structure_index = found->second;
        }
        else
        {
          int* colptr_raw = nullptr;
          int* rowind_raw = nullptr;
          const std::size_t nnz =
            DMat<D,Real>::build_natural_csc_pattern_from_stencil(
              degree,
              stencil,
              &colptr_raw,
              &rowind_raw);

          SharedOperatorStructure structure;
          copy_csc_raw_structure(
            Basis<D,Real>::dim_Pi(degree - 1),
            Basis<D,Real>::dim_Pi(degree),
            degree,
            signature,
            colptr_raw,
            rowind_raw,
            nnz,
            structure);

          structure_index =
            static_cast<int>(d_structures.size());
          d_structures.push_back(std::move(structure));
          d_signature_degree_index.emplace(
            intern_key,
            structure_index);
        }

        dag_d_structure_index.emplace(
          std::make_pair(degree, axis),
          structure_index);
        if (degree == n)
        {
          d_axis_structure[static_cast<std::size_t>(axis)] =
            structure_index;
        }
      }

      stencil.clear();
    }

    // One persistent delta-stencil load/discovery per promoted parameter.
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      KMatStencil stencil{};
      KMat<D,Real>::load_or_discover_natural_stencil(
        static_cast<unsigned int>(q_vol),
        kappa.data(),
        parameter,
        &stencil,
        stencil_folder);

      const std::string signature =
        KMat<D,Real>::stencil_signature(stencil);

      for (int degree = n; degree >= n - 2; --degree)
      {
        const std::pair<std::string, int> intern_key{
          signature, degree};
        int structure_index = -1;
        const auto found =
          k_signature_degree_index.find(intern_key);

        if (found != k_signature_degree_index.end())
        {
          structure_index = found->second;
        }
        else
        {
          int* colptr_raw = nullptr;
          int* rowind_raw = nullptr;
          const std::size_t nnz =
            KMat<D,Real>::build_natural_csc_pattern_from_stencil(
              degree,
              stencil,
              &colptr_raw,
              &rowind_raw);

          SharedOperatorStructure structure;
          const int size = Basis<D,Real>::dim_Pi(degree);
          copy_csc_raw_structure(
            size,
            size,
            degree,
            signature,
            colptr_raw,
            rowind_raw,
            nnz,
            structure);

          structure_index =
            static_cast<int>(k_structures.size());
          k_structures.push_back(std::move(structure));
          k_signature_degree_index.emplace(
            intern_key,
            structure_index);
        }

        dag_k_structure_index.emplace(
          std::make_pair(degree, parameter),
          structure_index);
        if (degree == n)
        {
          k_parameter_structure[
            static_cast<std::size_t>(parameter)] =
            structure_index;
        }
      }

      stencil.clear();
    }
  }

  const SharedOperatorStructure& d_structure(int axis) const
  {
    if (axis < 0 || axis >= D)
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: derivative-axis structure index");
    }
    const int index = d_axis_structure[static_cast<std::size_t>(axis)];
    if (index < 0 || index >= static_cast<int>(d_structures.size()))
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: derivative structure not initialized");
    }
    return d_structures[static_cast<std::size_t>(index)];
  }

  const SharedOperatorStructure& k_structure(int parameter) const
  {
    if (parameter < 0 || parameter > D)
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: promotion-parameter structure index");
    }
    const int index =
      k_parameter_structure[static_cast<std::size_t>(parameter)];
    if (index < 0 || index >= static_cast<int>(k_structures.size()))
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: promotion structure not initialized");
    }
    return k_structures[static_cast<std::size_t>(index)];
  }

  const SharedOperatorStructure& dag_d_structure(
    int degree,
    int axis) const
  {
    const auto found =
      dag_d_structure_index.find(std::make_pair(degree, axis));
    if (found == dag_d_structure_index.end())
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: missing DAG derivative structure");
    }
    return d_structures[static_cast<std::size_t>(found->second)];
  }

  const SharedOperatorStructure& dag_k_structure(
    int degree,
    int parameter) const
  {
    const auto found =
      dag_k_structure_index.find(
        std::make_pair(degree, parameter));
    if (found == dag_k_structure_index.end())
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: missing DAG promotion structure");
    }
    return k_structures[static_cast<std::size_t>(found->second)];
  }

  std::array<Real, D + 1> shifted_kappa(
    const ShiftState& shift) const
  {
    std::array<Real, D + 1> result{};
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      result[static_cast<std::size_t>(parameter)] =
        kappa[static_cast<std::size_t>(parameter)] +
        static_cast<Real>(
          shift.delta[static_cast<std::size_t>(parameter)]);
    }
    return result;
  }

  static int partial_order(
    const std::array<int, D>& alpha)
  {
    int order = 0;
    for (int axis = 0; axis < D; ++axis)
    {
      order += alpha[static_cast<std::size_t>(axis)];
    }
    return order;
  }

  static ShiftState derivative_shift(
    const std::array<int, D>& alpha)
  {
    ShiftState shift;
    int order = 0;
    for (int axis = 0; axis < D; ++axis)
    {
      const int count =
        alpha[static_cast<std::size_t>(axis)];
      shift.delta[static_cast<std::size_t>(axis)] = count;
      order += count;
    }
    shift.delta[static_cast<std::size_t>(D)] = order;
    return shift;
  }

  static void enumerate_exact_order_recursive(
    int axis,
    int remaining,
    std::array<int, D>& alpha,
    std::vector<std::array<int, D>>& result)
  {
    if (axis == D - 1)
    {
      alpha[static_cast<std::size_t>(axis)] = remaining;
      result.push_back(alpha);
      return;
    }

    for (int count = 0; count <= remaining; ++count)
    {
      alpha[static_cast<std::size_t>(axis)] = count;
      enumerate_exact_order_recursive(
        axis + 1,
        remaining - count,
        alpha,
        result);
    }
  }

  static std::vector<std::array<int, D>>
  enumerate_exact_order(int order)
  {
    std::vector<std::array<int, D>> result;
    std::array<int, D> alpha{};
    enumerate_exact_order_recursive(
      0,
      order,
      alpha,
      result);
    return result;
  }

  int get_or_build_dag_d_factor(
    int degree,
    const ShiftState& source_shift,
    int axis)
  {
    for (std::size_t index = 0;
         index < dag_d_factors.size();
         ++index)
    {
      const SparseFactor& factor = dag_d_factors[index];
      if (factor.degree == degree &&
          factor.coordinate == axis &&
          factor.source_shift == source_shift)
      {
        return static_cast<int>(index);
      }
    }

    const SharedOperatorStructure& structure =
      dag_d_structure(degree, axis);
    SparseFactor factor;
    factor.structure_index =
      static_cast<int>(&structure - d_structures.data());
    factor.degree = degree;
    factor.coordinate = axis;
    factor.source_shift = source_shift;
    factor.values.assign(structure.nnz(), Real(0));

    const std::array<Real, D + 1> source =
      shifted_kappa(source_shift);
    DMat<D,Real>::fill_tprod_natural_csc_values(
      degree,
      static_cast<unsigned int>(q_vol),
      source.data(),
      axis,
      structure.colptr.data(),
      structure.rowind.data(),
      factor.values.data());

    dag_d_factors.push_back(std::move(factor));
    return static_cast<int>(dag_d_factors.size()) - 1;
  }

  int get_or_build_dag_k_factor(
    int degree,
    const ShiftState& source_shift,
    int parameter)
  {
    for (std::size_t index = 0;
         index < dag_k_factors.size();
         ++index)
    {
      const SparseFactor& factor = dag_k_factors[index];
      if (factor.degree == degree &&
          factor.coordinate == parameter &&
          factor.source_shift == source_shift)
      {
        return static_cast<int>(index);
      }
    }

    const SharedOperatorStructure& structure =
      dag_k_structure(degree, parameter);
    SparseFactor factor;
    factor.structure_index =
      static_cast<int>(&structure - k_structures.data());
    factor.degree = degree;
    factor.coordinate = parameter;
    factor.source_shift = source_shift;
    factor.values.assign(structure.nnz(), Real(0));

    const std::array<Real, D + 1> source =
      shifted_kappa(source_shift);
    KMat<D,Real>::fill_tprod_natural_csc_values(
      degree,
      static_cast<unsigned int>(q_vol),
      source.data(),
      parameter,
      structure.colptr.data(),
      structure.rowind.data(),
      factor.values.data());

    dag_k_factors.push_back(std::move(factor));
    return static_cast<int>(dag_k_factors.size()) - 1;
  }

  static BatchedFactorGroup& find_or_add_batch(
    std::vector<BatchedFactorGroup>& batches,
    int stage,
    int structure_index)
  {
    for (BatchedFactorGroup& group : batches)
    {
      if (group.stage == stage &&
          group.structure_index == structure_index)
      {
        return group;
      }
    }

    BatchedFactorGroup group;
    group.stage = stage;
    group.structure_index = structure_index;
    batches.push_back(std::move(group));
    return batches.back();
  }

  void pack_batch_values(
    BatchedFactorGroup& group,
    const std::vector<SparseFactor>& factors,
    const std::vector<SharedOperatorStructure>& structures)
  {
    if (group.structure_index < 0 ||
        group.structure_index >=
          static_cast<int>(structures.size()))
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: invalid batch structure index");
    }

    const std::size_t nnz =
      structures[static_cast<std::size_t>(
        group.structure_index)].nnz();
    const std::size_t lane_count = group.lanes.size();
    group.values.assign(nnz * lane_count, Real(0));

    for (std::size_t lane = 0; lane < lane_count; ++lane)
    {
      const int factor_index = group.lanes[lane].factor_index;
      if (factor_index < 0 ||
          factor_index >= static_cast<int>(factors.size()))
      {
        throw std::runtime_error(
          "RefSimplexPrecomp: invalid batch factor index");
      }
      const std::vector<Real>& source =
        factors[static_cast<std::size_t>(factor_index)].values;
      if (source.size() != nnz)
      {
        throw std::runtime_error(
          "RefSimplexPrecomp: factor/stencil nnz mismatch");
      }
      for (std::size_t pos = 0; pos < nnz; ++pos)
      {
        group.values[pos * lane_count + lane] = source[pos];
      }
    }
  }

  void build_partial_batches()
  {
    derivative_batches.clear();
    promotion_forward_batches.clear();
    promotion_transpose_batches.clear();

    for (std::size_t child = 1;
         child < partials.size();
         ++child)
    {
      const PartialPlan& plan = partials[child];
      const SparseFactor& factor =
        dag_d_factors[
          static_cast<std::size_t>(plan.derivative_factor)];
      find_or_add_batch(
        derivative_batches,
        plan.order - 1,
        factor.structure_index)
        .lanes.push_back(FactorLane{
          plan.parent_partial,
          static_cast<int>(child),
          plan.derivative_factor});
    }

    for (std::size_t partial = 0;
         partial < partials.size();
         ++partial)
    {
      const PartialPlan& plan = partials[partial];
      const int count =
        static_cast<int>(plan.promotion_factors.size());
      for (int stage = 0; stage < count; ++stage)
      {
        const int factor_index =
          plan.promotion_factors[
            static_cast<std::size_t>(stage)];
        const SparseFactor& factor =
          dag_k_factors[
            static_cast<std::size_t>(factor_index)];

        find_or_add_batch(
          promotion_forward_batches,
          stage,
          factor.structure_index)
          .lanes.push_back(FactorLane{
            static_cast<int>(partial),
            static_cast<int>(partial),
            factor_index});

        const int reverse_stage = count - 1 - stage;
        find_or_add_batch(
          promotion_transpose_batches,
          reverse_stage,
          factor.structure_index)
          .lanes.push_back(FactorLane{
            static_cast<int>(partial),
            static_cast<int>(partial),
            factor_index});
      }
    }

    const auto batch_order =
      [](const BatchedFactorGroup& left,
         const BatchedFactorGroup& right)
      {
        if (left.stage != right.stage)
        {
          return left.stage < right.stage;
        }
        return left.structure_index < right.structure_index;
      };

    std::sort(
      derivative_batches.begin(),
      derivative_batches.end(),
      batch_order);
    std::sort(
      promotion_forward_batches.begin(),
      promotion_forward_batches.end(),
      batch_order);
    std::sort(
      promotion_transpose_batches.begin(),
      promotion_transpose_batches.end(),
      batch_order);

    for (BatchedFactorGroup& group : derivative_batches)
    {
      pack_batch_values(group, dag_d_factors, d_structures);
    }
    for (BatchedFactorGroup& group : promotion_forward_batches)
    {
      pack_batch_values(group, dag_k_factors, k_structures);
    }
    for (BatchedFactorGroup& group : promotion_transpose_batches)
    {
      pack_batch_values(group, dag_k_factors, k_structures);
    }
  }

  void build_partial_dag()
  {
    dag_d_factors.clear();
    dag_k_factors.clear();
    partials.clear();
    partial_value_count = 0;

    constexpr int maximum_order = 2;

    for (int order = 0; order <= maximum_order; ++order)
    {
      const std::vector<std::array<int, D>> alphas =
        enumerate_exact_order(order);

      for (const std::array<int, D>& alpha : alphas)
      {
        PartialPlan plan;
        plan.alpha = alpha;
        plan.order = order;
        plan.degree = n - order;
        plan.size = Basis<D,Real>::dim_Pi(plan.degree);
        plan.value_offset = partial_value_count;
        partial_value_count +=
          static_cast<std::size_t>(plan.size);

        if (order > 0)
        {
          int axis = D - 1;
          while (axis >= 0 &&
                 alpha[static_cast<std::size_t>(axis)] == 0)
          {
            --axis;
          }
          if (axis < 0)
          {
            throw std::runtime_error(
              "RefSimplexPrecomp: invalid derivative DAG node");
          }

          std::array<int, D> parent_alpha = alpha;
          --parent_alpha[static_cast<std::size_t>(axis)];
          plan.parent_partial = partial_index(parent_alpha);
          if (plan.parent_partial < 0)
          {
            throw std::runtime_error(
              "RefSimplexPrecomp: derivative DAG parent not found");
          }

          const PartialPlan& parent =
            partials[
              static_cast<std::size_t>(plan.parent_partial)];
          plan.derivative_factor =
            get_or_build_dag_d_factor(
              parent.degree,
              derivative_shift(parent_alpha),
              axis);
        }

        ShiftState current_shift = derivative_shift(alpha);
        for (int parameter = 0; parameter <= D; ++parameter)
        {
          while (
            current_shift.delta[
              static_cast<std::size_t>(parameter)] <
            maximum_order)
          {
            const int factor_index =
              get_or_build_dag_k_factor(
                plan.degree,
                current_shift,
                parameter);
            plan.promotion_factors.push_back(factor_index);
            ++current_shift.delta[
              static_cast<std::size_t>(parameter)];
          }
        }

        partials.push_back(std::move(plan));
      }
    }

    build_partial_batches();
  }

  void apply_derivative_group_forward(
    const BatchedFactorGroup& group,
    const std::vector<Real>& input,
    std::vector<Real>& output) const
  {
    const SharedOperatorStructure& structure =
      d_structures[
        static_cast<std::size_t>(group.structure_index)];
    const std::size_t lane_count = group.lanes.size();

    for (int col = 0; col < structure.cols; ++col)
    {
      for (int pos =
             structure.colptr[static_cast<std::size_t>(col)];
           pos <
             structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row =
          structure.rowind[static_cast<std::size_t>(pos)];
        const Real* entry_values =
          group.values.data() +
          static_cast<std::size_t>(pos) * lane_count;

        for (std::size_t lane = 0;
             lane < lane_count;
             ++lane)
        {
          const FactorLane& mapping = group.lanes[lane];
          const PartialPlan& source =
            partials[
              static_cast<std::size_t>(
                mapping.source_partial)];
          const PartialPlan& target =
            partials[
              static_cast<std::size_t>(
                mapping.target_partial)];

          output[target.value_offset +
                 static_cast<std::size_t>(row)] +=
            entry_values[lane] *
            input[source.value_offset +
                  static_cast<std::size_t>(col)];
        }
      }
    }
  }

  void apply_derivative_group_transpose(
    const BatchedFactorGroup& group,
    const std::vector<Real>& input,
    std::vector<Real>& output) const
  {
    const SharedOperatorStructure& structure =
      d_structures[
        static_cast<std::size_t>(group.structure_index)];
    const std::size_t lane_count = group.lanes.size();

    for (int col = 0; col < structure.cols; ++col)
    {
      for (int pos =
             structure.colptr[static_cast<std::size_t>(col)];
           pos <
             structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row =
          structure.rowind[static_cast<std::size_t>(pos)];
        const Real* entry_values =
          group.values.data() +
          static_cast<std::size_t>(pos) * lane_count;

        for (std::size_t lane = 0;
             lane < lane_count;
             ++lane)
        {
          const FactorLane& mapping = group.lanes[lane];
          const PartialPlan& source =
            partials[
              static_cast<std::size_t>(
                mapping.source_partial)];
          const PartialPlan& target =
            partials[
              static_cast<std::size_t>(
                mapping.target_partial)];

          output[source.value_offset +
                 static_cast<std::size_t>(col)] +=
            entry_values[lane] *
            input[target.value_offset +
                  static_cast<std::size_t>(row)];
        }
      }
    }
  }

  void apply_promotion_group_forward(
    const BatchedFactorGroup& group,
    const std::vector<Real>& input,
    std::vector<Real>& output) const
  {
    const SharedOperatorStructure& structure =
      k_structures[
        static_cast<std::size_t>(group.structure_index)];
    const std::size_t lane_count = group.lanes.size();

    for (int col = 0; col < structure.cols; ++col)
    {
      for (int pos =
             structure.colptr[static_cast<std::size_t>(col)];
           pos <
             structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row =
          structure.rowind[static_cast<std::size_t>(pos)];
        const Real* entry_values =
          group.values.data() +
          static_cast<std::size_t>(pos) * lane_count;

        for (std::size_t lane = 0;
             lane < lane_count;
             ++lane)
        {
          const FactorLane& mapping = group.lanes[lane];
          const PartialPlan& plan =
            partials[
              static_cast<std::size_t>(
                mapping.target_partial)];

          output[plan.value_offset +
                 static_cast<std::size_t>(row)] +=
            entry_values[lane] *
            input[plan.value_offset +
                  static_cast<std::size_t>(col)];
        }
      }
    }
  }

  void apply_promotion_group_transpose(
    const BatchedFactorGroup& group,
    const std::vector<Real>& input,
    std::vector<Real>& output) const
  {
    const SharedOperatorStructure& structure =
      k_structures[
        static_cast<std::size_t>(group.structure_index)];
    const std::size_t lane_count = group.lanes.size();

    for (int col = 0; col < structure.cols; ++col)
    {
      for (int pos =
             structure.colptr[static_cast<std::size_t>(col)];
           pos <
             structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row =
          structure.rowind[static_cast<std::size_t>(pos)];
        const Real* entry_values =
          group.values.data() +
          static_cast<std::size_t>(pos) * lane_count;

        for (std::size_t lane = 0;
             lane < lane_count;
             ++lane)
        {
          const FactorLane& mapping = group.lanes[lane];
          const PartialPlan& plan =
            partials[
              static_cast<std::size_t>(
                mapping.target_partial)];

          output[plan.value_offset +
                 static_cast<std::size_t>(col)] +=
            entry_values[lane] *
            input[plan.value_offset +
                  static_cast<std::size_t>(row)];
        }
      }
    }
  }

  static Real relative_frobenius_error(
    const std::vector<Real>& candidate,
    const std::vector<Real>& reference)
  {
    if (candidate.size() != reference.size())
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: compatibility size mismatch");
    }

    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (std::size_t index = 0;
         index < candidate.size();
         ++index)
    {
      const long double difference =
        static_cast<long double>(candidate[index]) -
        static_cast<long double>(reference[index]);
      const long double value =
        static_cast<long double>(reference[index]);
      numerator += difference * difference;
      denominator += value * value;
    }

    const long double scale =
      std::max(denominator, 1.0e-300L);
    return static_cast<Real>(
      std::sqrt(numerator / scale));
  }

  std::vector<Real> build_d_factor_dense(
    const Real* kappa_src,
    int axis) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null derivative source kappa");
    }
    const SharedOperatorStructure& structure = d_structure(axis);
    std::vector<Real> values(structure.nnz(), Real(0));
    DMat<D,Real>::fill_tprod_natural_csc_values(
      n,
      static_cast<unsigned int>(q_vol),
      kappa_src,
      axis,
      structure.colptr.data(),
      structure.rowind.data(),
      values.data());

    // Preserve the legacy padded row-major M x M factor layout.
    std::vector<Real> dense(
      static_cast<std::size_t>(M) * M, Real(0));
    for (int col = 0; col < M; ++col)
    {
      for (int pos = structure.colptr[static_cast<std::size_t>(col)];
           pos < structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row = structure.rowind[static_cast<std::size_t>(pos)];
        dense[static_cast<std::size_t>(row) * M + col] =
          values[static_cast<std::size_t>(pos)];
      }
    }
    return dense;
  }

  std::vector<Real> build_k_factor_dense(
    const Real* kappa_src,
    int promoted_parameter) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null promotion source kappa");
    }
    const SharedOperatorStructure& structure =
      k_structure(promoted_parameter);
    std::vector<Real> values(structure.nnz(), Real(0));
    KMat<D,Real>::fill_tprod_natural_csc_values(
      n,
      static_cast<unsigned int>(q_vol),
      kappa_src,
      promoted_parameter,
      structure.colptr.data(),
      structure.rowind.data(),
      values.data());

    std::vector<Real> dense(
      static_cast<std::size_t>(M) * M, Real(0));
    for (int col = 0; col < M; ++col)
    {
      for (int pos = structure.colptr[static_cast<std::size_t>(col)];
           pos < structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row = structure.rowind[static_cast<std::size_t>(pos)];
        dense[static_cast<std::size_t>(row) * M + col] =
          values[static_cast<std::size_t>(pos)];
      }
    }
    return dense;
  }

  void row_major_product(
    const std::vector<Real>& A,
    const std::vector<Real>& B,
    std::vector<Real>& C) const
  {
    const std::size_t count = static_cast<std::size_t>(M) * M;
    if (A.size() != count || B.size() != count)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: incompatible dense factor dimensions");
    }
    C.assign(count, Real(0));
    detail::BlasGemm<Real>::run(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      M,
      M,
      M,
      Real(1),
      A.data(),
      M,
      B.data(),
      M,
      Real(0),
      C.data(),
      M);
  }

  std::array<int, D + 1> residual_promotion_remaining(
    const Real* kappa_src) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null residual-promotion source kappa");
    }
    std::array<int, D + 1> remaining{};
    const Real tolerance = Real(1.0e-12);
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      const Real difference =
        kappa_res[static_cast<std::size_t>(parameter)] -
        kappa_src[parameter];
      const long long rounded = std::llround(
        static_cast<long double>(difference));
      if (rounded < 0 ||
          std::abs(difference - static_cast<Real>(rounded)) > tolerance)
      {
        throw std::invalid_argument(
          "RefSimplexPrecomp: residual promotion is not a nonnegative "
          "integer parameter shift");
      }
      remaining[static_cast<std::size_t>(parameter)] =
        static_cast<int>(rounded);
    }
    return remaining;
  }

  const std::vector<Real>& build_residual_promotion_from_remaining(
    const std::array<int, D + 1>& remaining)
  {
    const auto found = residual_promotion_cache.find(remaining);
    if (found != residual_promotion_cache.end())
    {
      return found->second;
    }

    bool identity = true;
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      if (remaining[static_cast<std::size_t>(parameter)] != 0)
      {
        identity = false;
        break;
      }
    }
    if (identity)
    {
      std::vector<Real> I(
        static_cast<std::size_t>(M) * M, Real(0));
      for (int i = 0; i < M; ++i)
      {
        I[static_cast<std::size_t>(i) * M + i] = Real(1);
      }
      const auto inserted = residual_promotion_cache.emplace(
        remaining, std::move(I));
      return inserted.first->second;
    }

    int parameter = -1;
    for (int r = 0; r <= D; ++r)
    {
      if (remaining[static_cast<std::size_t>(r)] > 0)
      {
        parameter = r;
        break;
      }
    }
    if (parameter < 0)
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: invalid residual-promotion DAG state");
    }

    std::array<Real, D + 1> source{};
    for (int r = 0; r <= D; ++r)
    {
      source[static_cast<std::size_t>(r)] =
        kappa_res[static_cast<std::size_t>(r)] -
        static_cast<Real>(remaining[static_cast<std::size_t>(r)]);
    }

    std::array<int, D + 1> next_remaining = remaining;
    --next_remaining[static_cast<std::size_t>(parameter)];

    const std::vector<Real> natural_factor =
      build_k_factor_dense(source.data(), parameter);
    const std::vector<Real>& tail =
      build_residual_promotion_from_remaining(next_remaining);

    std::vector<Real> result;
    row_major_product(tail, natural_factor, result);
    const auto inserted = residual_promotion_cache.emplace(
      remaining, std::move(result));
    return inserted.first->second;
  }

  const std::vector<Real>& build_residual_promotion(
    const Real* kappa_src)
  {
    return build_residual_promotion_from_remaining(
      residual_promotion_remaining(kappa_src));
  }

  void copy_row_major_to_column_major(
    const std::vector<Real>& source,
    Real* destination) const
  {
    if (!destination ||
        source.size() != static_cast<std::size_t>(M) * M)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: invalid dense layout conversion");
    }
    for (int col = 0; col < M; ++col)
    {
      for (int row = 0; row < M; ++row)
      {
        destination[static_cast<std::size_t>(row) +
                    static_cast<std::size_t>(M) * col] =
          source[static_cast<std::size_t>(row) * M + col];
      }
    }
  }

  void build_derivative_factor_data()
  {
    D1_factor_rm.assign(
      static_cast<std::size_t>(D) * M * M, Real(0));
    D2_factor_rm.assign(
      static_cast<std::size_t>(D) * D * M * M, Real(0));
    derivative_kappa_1.resize(static_cast<std::size_t>(D));

    for (int first_axis = 0; first_axis < D; ++first_axis)
    {
      Basis<D,Real>::derivative_kappa_shift(
        kappa.data(),
        first_axis,
        derivative_kappa_1[static_cast<std::size_t>(first_axis)].data());

      const std::vector<Real> first =
        build_d_factor_dense(kappa.data(), first_axis);
      std::copy(
        first.begin(),
        first.end(),
        D1_factor_rm.begin() +
          static_cast<std::size_t>(first_axis) * M * M);

      for (int second_axis = 0; second_axis < D; ++second_axis)
      {
        const std::vector<Real> second = build_d_factor_dense(
          derivative_kappa_1[static_cast<std::size_t>(first_axis)].data(),
          second_axis);
        std::copy(
          second.begin(),
          second.end(),
          D2_factor_rm.begin() + static_cast<std::size_t>(M) * M *
            (static_cast<std::size_t>(second_axis) +
             static_cast<std::size_t>(D) * first_axis));
      }
    }
  }

  void build_second_partials()
  {
    Lij_ref.assign(
      static_cast<std::size_t>(M) * M * D * D, Real(0));

    for (int first_axis = 0; first_axis < D; ++first_axis)
    {
      const Real* first = D1_factor_rm.data() +
        static_cast<std::size_t>(first_axis) * M * M;
      const std::vector<Real> first_matrix(
        first,
        first + static_cast<std::size_t>(M) * M);

      for (int second_axis = 0; second_axis < D; ++second_axis)
      {
        const Real* second = D2_factor_rm.data() +
          static_cast<std::size_t>(M) * M *
          (static_cast<std::size_t>(second_axis) +
           static_cast<std::size_t>(D) * first_axis);
        const std::vector<Real> second_matrix(
          second,
          second + static_cast<std::size_t>(M) * M);

        std::vector<Real> derivative_product;
        row_major_product(
          second_matrix,
          first_matrix,
          derivative_product);

        std::array<Real, D + 1> kappa_second{};
        Basis<D,Real>::derivative_kappa_shift(
          derivative_kappa_1[static_cast<std::size_t>(first_axis)].data(),
          second_axis,
          kappa_second.data());

        const std::vector<Real>& promotion =
          build_residual_promotion(kappa_second.data());
        std::vector<Real> result;
        row_major_product(promotion, derivative_product, result);

        Real* output = Lij_ref.data() +
          static_cast<std::size_t>(M) * M *
          (static_cast<std::size_t>(first_axis) +
           static_cast<std::size_t>(D) * second_axis);
        copy_row_major_to_column_major(result, output);
      }
    }
  }

  void build_first_partials()
  {
    Li_ref.assign(
      static_cast<std::size_t>(M) * M * D, Real(0));

    for (int axis = 0; axis < D; ++axis)
    {
      const Real* derivative = D1_factor_rm.data() +
        static_cast<std::size_t>(axis) * M * M;
      const std::vector<Real> derivative_matrix(
        derivative,
        derivative + static_cast<std::size_t>(M) * M);

      const std::vector<Real>& promotion = build_residual_promotion(
        derivative_kappa_1[static_cast<std::size_t>(axis)].data());
      std::vector<Real> result;
      row_major_product(promotion, derivative_matrix, result);

      copy_row_major_to_column_major(
        result,
        Li_ref.data() + static_cast<std::size_t>(M) * M * axis);
    }
  }

  void build_zero_partials()
  {
    L0_ref.assign(static_cast<std::size_t>(M) * M, Real(0));
    const std::vector<Real>& promotion =
      build_residual_promotion(kappa.data());
    copy_row_major_to_column_major(promotion, L0_ref.data());
  }

  void embed_face_points(int face_id, const int* sigma, std::vector<Real>& Xf) const
  {
    Xf.assign((std::size_t)nq_face * D, (Real)0);

    if constexpr (D == 1)
    {
      Xf[0] = (face_id == 0) ? (Real)1 : (Real)0;
      (void)sigma;
    }
    else
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);

      for (int q = 0; q < nq_face; ++q)
      {
        Real mu[D];
        Real sum_y = (Real)0;
        for (int a = 1; a < D; ++a)
        {
          const Real y = Y_face[(std::size_t)q * (D - 1) + (a - 1)];
          mu[a] = y;
          sum_y += y;
        }
        mu[0] = (Real)1 - sum_y;

        Real lambda[D + 1];
        for (int a = 0; a <= D; ++a) { lambda[a] = (Real)0; }

        // sigma is local-face-position -> canonical-face-position.
        for (int local = 0; local < D; ++local)
        {
          const int canonical = sigma[local];
          lambda[fv[local]] = mu[canonical];
        }

        for (int a = 0; a < D; ++a)
        {
          Xf[(std::size_t)q * D + a] = lambda[a + 1];
        }
      }
    }
  }

  void build_face_operator_data()
  {
    T_ref.assign((std::size_t)kf * M * nsigma * nface, (Real)0);
    Fgrad_ref.assign((std::size_t)kf * M * D * nsigma * nface, (Real)0);
    Mface_ref.assign((std::size_t)kf * kf * nface, (Real)0);

    std::vector<int> alpha_vol, tail_vol;
    std::vector<Real> invh_vol;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha_vol, tail_vol, invh_vol);

    std::array<std::array<Real, D + 1>, D> k_deriv{};
    std::array<std::vector<int>, D> alpha_deriv;
    std::array<std::vector<int>, D> tail_deriv;
    std::array<std::vector<Real>, D> invh_deriv;
    for (int a = 0; a < D; ++a)
    {
      Basis<D,Real>::derivative_kappa_shift(kappa.data(), a, k_deriv[(std::size_t)a].data());
      Basis<D,Real>::build_structures(n, k_deriv[(std::size_t)a].data(),
                                      alpha_deriv[(std::size_t)a],
                                      tail_deriv[(std::size_t)a],
                                      invh_deriv[(std::size_t)a]);
    }

    for (int face_id = 0; face_id < nface; ++face_id)
    {
      const Real fscale = face_ref_scale[(std::size_t)face_id];

      // Face mass matrix for this embedded reference face:
      //   Mface_ref = V_face^T * diag(W_face * fscale) * V_face.
      std::vector<Real> WV_face((std::size_t)nq_face * kf, Real(0));
      for (int col = 0; col < kf; ++col)
      {
        for (int q = 0; q < nq_face; ++q)
        {
          WV_face[(std::size_t)q + (std::size_t)nq_face * col] =
            (W_face[(std::size_t)q] * fscale) *
            V_face[(std::size_t)q + (std::size_t)nq_face * col];
        }
      }
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        kf,
        kf,
        nq_face,
        Real(1),
        V_face.data(),
        nq_face,
        WV_face.data(),
        nq_face,
        Real(0),
        &Mface_ref[(std::size_t)kf * kf * face_id],
        kf);

      for (int sigma_index = 0; sigma_index < nsigma; ++sigma_index)
      {
        int sigma[D];
        lehmer_index_to_perm<D>(sigma_index, sigma);

        std::vector<Real> Xf;
        embed_face_points(face_id, sigma, Xf);

        std::vector<Real> Vv((std::size_t)nq_face * M, (Real)0);
        Basis<D,Real>::eval_all(
          Xf.data(), D, 1, nq_face, kappa.data(), n,
          alpha_vol.data(), tail_vol.data(), invh_vol.data(), Vv.data(), nq_face, nullptr);

        // T_ref block:
        //   T_ref = V_face^T * diag(W_face * fscale) * Vv.
        std::vector<Real> WVv((std::size_t)nq_face * M, Real(0));
        for (int col = 0; col < M; ++col)
        {
          for (int q = 0; q < nq_face; ++q)
          {
            WVv[(std::size_t)q + (std::size_t)nq_face * col] =
              (W_face[(std::size_t)q] * fscale) *
              Vv[(std::size_t)q + (std::size_t)nq_face * col];
          }
        }
        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasTrans,
          CblasNoTrans,
          kf,
          M,
          nq_face,
          Real(1),
          V_face.data(),
          nq_face,
          WVv.data(),
          nq_face,
          Real(0),
          &T_ref[idx_T(0, 0, sigma_index, face_id)],
          kf);

        for (int a = 0; a < D; ++a)
        {
          std::vector<Real> Vrng((std::size_t)nq_face * M, (Real)0);
          Basis<D,Real>::eval_all(
            Xf.data(), D, 1, nq_face, k_deriv[(std::size_t)a].data(), n,
            alpha_deriv[(std::size_t)a].data(),
            tail_deriv[(std::size_t)a].data(),
            invh_deriv[(std::size_t)a].data(),
            Vrng.data(), nq_face, nullptr);

          std::vector<Real> dV((std::size_t)nq_face * M, Real(0));
          // dV = Vrng * D_a.  Vrng is column-major nq_face x M.
          // D_a is stored row-major M x M; interpreted as column-major it is
          // D_a^T, so use transB = CblasTrans.
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasNoTrans,
            CblasTrans,
            nq_face,
            M,
            M,
            Real(1),
            Vrng.data(),
            nq_face,
            D1_factor_rm.data() + (std::size_t)a * M * M,
            M,
            Real(0),
            dV.data(),
            nq_face);

          // Fgrad_ref block:
          //   Fgrad_ref = V_face^T * diag(W_face * fscale) * dV.
          std::vector<Real> WdV((std::size_t)nq_face * M, Real(0));
          for (int col = 0; col < M; ++col)
          {
            for (int q = 0; q < nq_face; ++q)
            {
              WdV[(std::size_t)q + (std::size_t)nq_face * col] =
                (W_face[(std::size_t)q] * fscale) *
                dV[(std::size_t)q + (std::size_t)nq_face * col];
            }
          }
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasTrans,
            CblasNoTrans,
            kf,
            M,
            nq_face,
            Real(1),
            V_face.data(),
            nq_face,
            WdV.data(),
            nq_face,
            Real(0),
            &Fgrad_ref[idx_F(0, 0, a, sigma_index, face_id)],
            kf);
        }
      }
    }
  }

  CSCBlock make_pruned_csc_block(int rows,
                                 int cols,
                                 const Real* A_colmajor) const
  {
    if (rows < 0 || cols < 0 || !A_colmajor)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: invalid dense block for CSC conversion");
    }

    long double norm2 = 0.0L;
    const std::size_t count = (std::size_t)rows * cols;
    for (std::size_t k = 0; k < count; ++k)
    {
      const long double v = static_cast<long double>(A_colmajor[k]);
      norm2 += v * v;
    }
    const Real threshold = boundary_csc_rel_prune *
      static_cast<Real>(std::sqrt(norm2));

    std::vector<Real> pruned(count, Real(0));
    for (std::size_t k = 0; k < count; ++k)
    {
      const Real v = A_colmajor[k];
      if (std::abs(v) > threshold) pruned[k] = v;
    }

    int* colptr_raw = nullptr;
    int* rowind_raw = nullptr;
    Real* values_raw = nullptr;
    const std::size_t nnz = detail::compress_dense_to_csc(
      rows,
      cols,
      pruned.data(),
      rows,
      true,
      &colptr_raw,
      &rowind_raw,
      &values_raw);

    std::unique_ptr<int, decltype(&std::free)> colptr_guard(
      colptr_raw, &std::free);
    std::unique_ptr<int, decltype(&std::free)> rowind_guard(
      rowind_raw, &std::free);
    std::unique_ptr<Real, decltype(&std::free)> values_guard(
      values_raw, &std::free);

    CSCBlock out;
    out.rows = rows;
    out.cols = cols;
    out.colptr.assign(colptr_raw, colptr_raw + (std::size_t)cols + 1);
    if (nnz > 0)
    {
      out.rowind.assign(rowind_raw, rowind_raw + nnz);
      out.values.assign(values_raw, values_raw + nnz);
    }
    return out;
  }

  void build_face_sparse_data()
  {
    T_ref_csc.resize((std::size_t)nsigma * nface);
    Fgrad_ref_csc.resize((std::size_t)D * nsigma * nface);
    Mface_ref_csc.resize((std::size_t)nface);

    for (int face = 0; face < nface; ++face)
    {
      Mface_ref_csc[(std::size_t)face] = make_pruned_csc_block(
        kf,
        kf,
        Mface_ref.data() + (std::size_t)kf * kf * face);

      for (int sigma = 0; sigma < nsigma; ++sigma)
      {
        T_ref_csc[(std::size_t)sigma + (std::size_t)nsigma * face] =
          make_pruned_csc_block(
            kf,
            M,
            T_ref.data() + idx_T(0, 0, sigma, face));

        for (int axis = 0; axis < D; ++axis)
        {
          Fgrad_ref_csc[(std::size_t)axis + (std::size_t)D *
            ((std::size_t)sigma + (std::size_t)nsigma * face)] =
            make_pruned_csc_block(
              kf,
              M,
              Fgrad_ref.data() + idx_F(0, 0, axis, sigma, face));
        }
      }
    }
  }

  std::size_t idx_T(int row, int col, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)sigma + (std::size_t)nsigma * face));
  }

  std::size_t idx_F(int row, int col, int axis, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)axis + (std::size_t)D *
      ((std::size_t)sigma + (std::size_t)nsigma * face)));
  }
};

} // namespace jsimplex

#endif // JPRECOMP_HH
