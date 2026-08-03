#ifndef JNODE_HH
#define JNODE_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <jdetail.hh>
#include <jelliptic.hh>
#include <jmesh.hh>
#include <jleaf.hh>
#ifdef TIMING
#include <timer.hh>
#endif

namespace jsimplex {

template<int D, class Real>
struct NodeMergeData
{
  using FaceKey = DSimplexFaceKey<D>;

  int child_A = -1;
  int child_B = -1;

  // Face-key partitions used for this merge.  Parent ordering is EA followed
  // by EB in this first implementation.
  std::vector<FaceKey> EA_faces;
  std::vector<FaceKey> EB_faces;
  std::vector<FaceKey> I_faces;

  // Scalar indices into child maps/traces, after expanding by kf.
  std::vector<int> A_E_idx;
  std::vector<int> A_I_idx;
  std::vector<int> B_E_idx;
  std::vector<int> B_I_idx;

  // gamma = R_A lambda_AE + R_B lambda_BE + r.
  // Shapes are column-major:
  //   R_A: nI x nAE
  //   R_B: nI x nBE
  //   r:   nI
  std::vector<Real> R_A;
  std::vector<Real> R_B;
  std::vector<Real> r;

  // Reusable source-upward maps for sI=b_A,I+b_B,I:
  //   r       = R_f sI,
  //   b_P,EA  = b_A,E + P_A sI,
  //   b_P,EB  = b_B,E + P_B sI.
  // Shapes are column-major:
  //   R_f: nI x nI
  //   P_A: nAE x nI
  //   P_B: nBE x nI
  std::vector<Real> R_f;
  std::vector<Real> P_A;
  std::vector<Real> P_B;

  int nAE = 0;
  int nBE = 0;
  int nI = 0;
};

#ifdef TIMING
struct LeafTiming
{
  double interior_scale_seconds = 0.0;
  double ulam_solve_seconds = 0.0;
  double homogeneous_boundary_map_seconds = 0.0;
  double source_solve_seconds = 0.0;
  double source_boundary_map_seconds = 0.0;

  long long ulam_lsmr_iterations = 0;
  long long source_lsmr_iterations = 0;
  EllipticActionTimings interior_scale_actions{};
  EllipticActionTimings ulam_actions{};
  EllipticActionTimings source_actions{};
};
#endif


template<int D, class Real = double>
struct Node
{
  using FaceKey = DSimplexFaceKey<D>;
  using MergeData = NodeMergeData<D,Real>;

  int kf = 0;
  std::vector<FaceKey> boundary_faces;

  // DtN map on boundary_faces:
  //   mu = S lambda + b.
  // S is nb x nb, column-major, nb=kf*boundary_faces.size().
  std::vector<Real> S;
  std::vector<Real> b;

  bool is_leaf = false;
  int leaf_element_id = -1;
  int vol_dim = 0;
  int source_dim = 0;

  // Reusable leaf response maps:
  //   c      = Ulam lambda + Uf f_int,
  //   mu_hat = S lambda    + Gf f_int.
  // Ulam is vol_dim x nb, Uf is vol_dim x source_dim,
  // Gf is nb x source_dim, all column-major.
  std::vector<Real> Ulam;
  std::vector<Real> Uf;
  std::vector<Real> Gf;

  // Current source-state vectors. They are updated from Uf/Gf without any
  // local least-squares solve and consumed by the normal upward/downward pass.
  std::vector<Real> cf;

#ifdef TIMING
  // Leaf-only profiling information. Internal merge nodes keep zero values.
  LeafTiming leaf_timing{};
#endif

  bool is_merge = false;
  MergeData merge;

  int nfaces() const
  {
    return static_cast<int>(boundary_faces.size());
  }

  int nb() const
  {
    return kf * nfaces();
  }

  void validate() const
  {
    if (kf <= 0)
    {
      throw std::runtime_error("Node::validate: kf must be positive");
    }

    const int n = nb();
    if (static_cast<int>(S.size()) != n * n)
    {
      throw std::runtime_error("Node::validate: S has wrong size");
    }
    if (static_cast<int>(b.size()) != n)
    {
      throw std::runtime_error("Node::validate: b has wrong size");
    }

    if (is_leaf)
    {
      if (vol_dim < 0)
      {
        throw std::runtime_error("Node::validate: negative vol_dim");
      }
      if (static_cast<int>(Ulam.size()) != vol_dim * n)
      {
        throw std::runtime_error("Node::validate: Ulam has wrong size");
      }
      if (source_dim < 0)
      {
        throw std::runtime_error("Node::validate: negative source_dim");
      }
      if (static_cast<int>(Uf.size()) != vol_dim * source_dim)
      {
        throw std::runtime_error("Node::validate: Uf has wrong size");
      }
      if (static_cast<int>(Gf.size()) != n * source_dim)
      {
        throw std::runtime_error("Node::validate: Gf has wrong size");
      }
      if (static_cast<int>(cf.size()) != vol_dim)
      {
        throw std::runtime_error("Node::validate: cf has wrong size");
      }
    }

    if (is_merge)
    {
      if (merge.nAE < 0 || merge.nBE < 0 || merge.nI <= 0)
      {
        throw std::runtime_error("Node::validate: invalid merge dimensions");
      }
      if (static_cast<int>(merge.R_A.size()) != merge.nI * merge.nAE
          || static_cast<int>(merge.R_B.size()) != merge.nI * merge.nBE
          || static_cast<int>(merge.r.size()) != merge.nI
          || static_cast<int>(merge.R_f.size()) != merge.nI * merge.nI
          || static_cast<int>(merge.P_A.size()) != merge.nAE * merge.nI
          || static_cast<int>(merge.P_B.size()) != merge.nBE * merge.nI)
      {
        throw std::runtime_error("Node::validate: merge map size mismatch");
      }
    }
  }
};

namespace node_detail {

template<int D, class Real>
inline Node<D,Real> materialize_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const Leaf<D,Real>& leaf,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace_in = nullptr)
{
  const auto& elem = mesh.element(element_id);

  Node<D,Real> node;
  node.kf = leaf.kf;
  node.boundary_faces.assign(
    elem.face_keys.begin(),
    elem.face_keys.end());
  node.is_leaf = true;
  node.leaf_element_id = element_id;
  node.vol_dim = leaf.M;
  node.source_dim = leaf.m_int;

  const int nb = node.nb();
  const int M = node.vol_dim;
  const int m_int = node.source_dim;
  const int nrhs = nb + m_int;
  if (nb != leaf.nb)
  {
    throw std::runtime_error(
      "materialize_homogeneous_leaf_node: Node/Leaf boundary size mismatch");
  }
  for (int face = 0;
       face < leaf.nface;
       ++face)
  {
    if (node.boundary_faces[(std::size_t)face]
        != leaf.face_key(face))
    {
      throw std::runtime_error(
        "materialize_homogeneous_leaf_node: Mesh/Leaf face ordering mismatch");
    }
  }

  node.S.assign(
    (std::size_t)nb * nb,
    Real(0));
  node.b.assign(
    (std::size_t)nb,
    Real(0));
  node.Ulam.assign(
    (std::size_t)M * nb,
    Real(0));
  node.Uf.assign(
    (std::size_t)M * m_int,
    Real(0));
  node.Gf.assign(
    (std::size_t)nb * m_int,
    Real(0));
  node.cf.assign(
    (std::size_t)M,
    Real(0));

  typename Leaf<D,Real>::SolveWorkspace local_workspace;
  typename Leaf<D,Real>::SolveWorkspace& solve_workspace =
    solve_workspace_in
    ? *solve_workspace_in
    : local_workspace;
  if (!solve_workspace.compatible(leaf))
  {
    solve_workspace.reset(leaf);
  }

#ifdef TIMING
  node.leaf_timing.interior_scale_seconds =
    leaf.timing_interior_scale_seconds;
  node.leaf_timing.interior_scale_actions =
    leaf.timing_interior_scale_actions;
  leaf.reset_elliptic_action_timings();

  timer response_timer;
  response_timer.tic();
#endif

  std::vector<Real> response_map(
    (std::size_t)M * nrhs,
    Real(0));
  long long response_lsmr_iterations = 0;
  leaf.solve_response_maps(
    response_map.data(),
    M,
    solve_workspace,
    &response_lsmr_iterations);

  for (int column = 0; column < nb; ++column)
  {
    std::copy_n(
      response_map.data()
        + (std::size_t)M * (std::size_t)column,
      M,
      node.Ulam.data()
        + (std::size_t)M * (std::size_t)column);
  }
  for (int column = 0; column < m_int; ++column)
  {
    std::copy_n(
      response_map.data()
        + (std::size_t)M * (std::size_t)(nb + column),
      M,
      node.Uf.data()
        + (std::size_t)M * (std::size_t)column);
  }

#ifdef TIMING
  node.leaf_timing.ulam_solve_seconds =
    response_timer.toc();
  node.leaf_timing.ulam_lsmr_iterations =
    response_lsmr_iterations;
  node.leaf_timing.ulam_actions =
    leaf.elliptic_action_timings();

  timer boundary_timer;
  boundary_timer.tic();
#endif

  // Apply trace and flux once to the complete response panel. Dense mode uses
  // two GEMMs over all boundary and source columns; action modes retain their
  // existing panel kernels.
  std::vector<Real> trace_response(
    (std::size_t)nb * nrhs,
    Real(0));
  std::vector<Real> flux_response(
    (std::size_t)nb * nrhs,
    Real(0));
  leaf.apply_trace_columns(
    response_map.data(),
    M,
    nrhs,
    trace_response.data(),
    nb);
  leaf.apply_flux_columns(
    response_map.data(),
    M,
    nrhs,
    flux_response.data(),
    nb);

#ifdef TIMING
  node.leaf_timing.homogeneous_boundary_map_seconds =
    boundary_timer.toc();
#endif

  // Boundary response: S=(F+tau*T)Ulam-tau*I.
  for (int column = 0; column < nb; ++column)
  {
    for (int row = 0; row < nb; ++row)
    {
      const std::size_t panel_index =
        (std::size_t)row
        + (std::size_t)nb * (std::size_t)column;
      const Real identity = row == column ? Real(1) : Real(0);
      node.S[panel_index] =
        flux_response[panel_index]
        + leaf.tau_rows[(std::size_t)row]
          * (trace_response[panel_index] - identity);
    }
  }

  // Source response: Gf=(F+tau*T)Uf.
  for (int column = 0; column < m_int; ++column)
  {
    const int panel_column = nb + column;
    for (int row = 0; row < nb; ++row)
    {
      const std::size_t panel_index =
        (std::size_t)row
        + (std::size_t)nb * (std::size_t)panel_column;
      node.Gf[
        (std::size_t)row
        + (std::size_t)nb * (std::size_t)column] =
        flux_response[panel_index]
        + leaf.tau_rows[(std::size_t)row]
          * trace_response[panel_index];
    }
  }

  node.validate();
  return node;
}

} // namespace node_detail

/*
  Materialize reusable elliptic leaf maps

    c(lambda,f)       = Ulam lambda + Uf f,
    mu_hat(lambda,f)  = S lambda + Gf f,

  using a caller-owned shared elliptic plan and caller-owned mutable workspace.
  The coefficient view is consumed only while Leaf materializes its dense local
  operator; neither Leaf nor Node retains the coefficient pointers.
*/
template<int D, class Real>
inline Node<D,Real> make_elliptic_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const RefSimplexPrecomp<D,Real>& pre,
  const EllipticPlan<D,Real>& elliptic_plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticWorkspace<D,Real>& elliptic_work,
  Leaf<D,Real>& leaf_out,
  Real tau_C_base = Real(1),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace = nullptr)
{
  if (!(tau_C_base > Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tau_C_base must be positive");
  }
  if (!(atol >= Real(0)) || !(btol >= Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tolerances must be nonnegative");
  }
  if (itnlim <= 0)
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: itnlim must be positive");
  }

  const auto& elem = mesh.element(element_id);

  typename Leaf<D,Real>::LsmrOptions opts{};
  opts.atol = atol;
  opts.btol = btol;
  opts.itnlim = itnlim;

  leaf_out.reset(
    pre,
    elem.V_phys.data(),
    elem.global_vids.data(),
    elliptic_plan,
    coeffs,
    elliptic_work,
    tau_C_base,
    opts);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out,
    solve_workspace);
}

/*
  Mode-aware elliptic leaf materialization with a caller-owned dense assembly
  workspace. DenseSparse uses this workspace only while assembling L; Verify
  uses it for the retained dense comparison backend.
*/
template<int D, class Real>
inline Node<D,Real> make_elliptic_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const RefSimplexPrecomp<D,Real>& pre,
  const EllipticPlan<D,Real>& elliptic_plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  EllipticDenseWorkspace<D,Real>& elliptic_work,
  Leaf<D,Real>& leaf_out,
  const typename Leaf<D,Real>::Options& leaf_options,
  Real tau_C_base = Real(1),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace = nullptr)
{
  if (!(tau_C_base > Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tau_C_base must be positive");
  }
  if (!(atol >= Real(0)) || !(btol >= Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tolerances must be nonnegative");
  }
  if (itnlim <= 0)
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: itnlim must be positive");
  }

  const auto& elem = mesh.element(element_id);

  typename Leaf<D,Real>::LsmrOptions opts{};
  opts.atol = atol;
  opts.btol = btol;
  opts.itnlim = itnlim;

  leaf_out.reset(
    pre,
    elem.V_phys.data(),
    elem.global_vids.data(),
    elliptic_plan,
    coeffs,
    elliptic_work,
    tau_C_base,
    opts,
    leaf_options);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out,
    solve_workspace);
}

/*
  Mode-aware elliptic leaf materialization.

  Dense mode preserves the explicit local matrices. DenseSparse retains only
  the composed dense interior matrix and sparse boundary actions. MatrixFree
  stores only the action workspace and the final HPS-facing Ulam/S maps.
  Verify retains both leaf backends and compares them according to
  leaf_options.

  MatrixFree and Verify may copy enabled coefficient arrays while the local
  response maps are materialized. Once Ulam, Uf, S, and Gf have been formed,
  source application is map-based and the Leaf operator may be released.
*/
template<int D, class Real>
inline Node<D,Real> make_elliptic_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const RefSimplexPrecomp<D,Real>& pre,
  const EllipticPlan<D,Real>& elliptic_plan,
  const EllipticElementCoefficientsView<D,Real>& coeffs,
  Leaf<D,Real>& leaf_out,
  const typename Leaf<D,Real>::Options& leaf_options,
  Real tau_C_base = Real(1),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace = nullptr)
{
  if (!(tau_C_base > Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tau_C_base must be positive");
  }
  if (!(atol >= Real(0)) || !(btol >= Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tolerances must be nonnegative");
  }
  if (itnlim <= 0)
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: itnlim must be positive");
  }

  const auto& elem = mesh.element(element_id);

  typename Leaf<D,Real>::LsmrOptions opts{};
  opts.atol = atol;
  opts.btol = btol;
  opts.itnlim = itnlim;

  leaf_out.reset(
    pre,
    elem.V_phys.data(),
    elem.global_vids.data(),
    elliptic_plan,
    coeffs,
    tau_C_base,
    opts,
    leaf_options);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out,
    solve_workspace);
}



/*
  Materialize reusable Poisson leaf maps

    c(lambda,f)       = Ulam lambda + Uf f,
    mu_hat(lambda,f)  = S lambda + Gf f,

  where mu_hat is Leaf::apply's augmented flux

    mu_hat = F c + tau (T c - lambda).

  This compatibility path preserves the existing caller interface.  The
  no-elliptic-arguments Leaf::reset currently assembles A=I, b=0, c=0 through
  the elliptic implementation.
*/
template<int D, class Real>
inline Node<D,Real> make_poisson_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const RefSimplexPrecomp<D,Real>& pre,
  Leaf<D,Real>& leaf_out,
  Real tau_C = Real(10),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace = nullptr)
{
  if (!(tau_C > Real(0)))
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: tau_C must be positive");
  }
  if (!(atol >= Real(0)) || !(btol >= Real(0)))
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: tolerances must be nonnegative");
  }
  if (itnlim <= 0)
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: itnlim must be positive");
  }

  const auto& elem = mesh.element(element_id);

  typename Leaf<D,Real>::LsmrOptions opts{};
  opts.atol = atol;
  opts.btol = btol;
  opts.itnlim = itnlim;

  leaf_out.reset(
    pre,
    elem.V_phys.data(),
    elem.global_vids.data(),
    tau_C,
    opts);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out,
    solve_workspace);
}

/*
  Mode-aware Poisson leaf materialization. The legacy overload above remains
  Dense by default; this overload permits DenseSparse, MatrixFree, and Verify
  without changing old callers.
*/
template<int D, class Real>
inline Node<D,Real> make_poisson_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const RefSimplexPrecomp<D,Real>& pre,
  Leaf<D,Real>& leaf_out,
  const typename Leaf<D,Real>::Options& leaf_options,
  Real tau_C = Real(10),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000,
  typename Leaf<D,Real>::SolveWorkspace* solve_workspace = nullptr)
{
  if (!(tau_C > Real(0)))
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: tau_C must be positive");
  }
  if (!(atol >= Real(0)) || !(btol >= Real(0)))
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: tolerances must be nonnegative");
  }
  if (itnlim <= 0)
  {
    throw std::invalid_argument(
      "make_poisson_homogeneous_leaf_node: itnlim must be positive");
  }

  const auto& elem = mesh.element(element_id);

  typename Leaf<D,Real>::LsmrOptions opts{};
  opts.atol = atol;
  opts.btol = btol;
  opts.itnlim = itnlim;

  leaf_out.reset(
    pre,
    elem.V_phys.data(),
    elem.global_vids.data(),
    tau_C,
    opts,
    leaf_options);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out,
    solve_workspace);
}



/*
  Set the source-dependent affine terms on an already materialized homogeneous
  leaf:

    c(0,f)       = cf,
    mu_hat(0,f)  = b.

  S and Ulam are left unchanged.  Internal-node b vectors and merge trace
  offsets must subsequently be recomputed by the source upward pass.
*/
template<int D, class Real>
inline void apply_leaf_source_columns(
  const Node<D,Real>& node,
  const Real* f_int,
  int ldf,
  int nrhs,
  Real* cf_out,
  int ldcf,
  Real* b_out,
  int ldb)
{
  node.validate();
  if (!node.is_leaf)
  {
    throw std::invalid_argument(
      "apply_leaf_source_columns: node is not a leaf");
  }
  if (nrhs < 0)
  {
    throw std::invalid_argument(
      "apply_leaf_source_columns: negative nrhs");
  }
  if (nrhs == 0)
  {
    return;
  }
  if (!cf_out || !b_out)
  {
    throw std::invalid_argument(
      "apply_leaf_source_columns: null output");
  }
  if (node.source_dim > 0 && !f_int)
  {
    throw std::invalid_argument(
      "apply_leaf_source_columns: null source panel");
  }
  if (ldf < std::max(node.source_dim, 1)
      || ldcf < std::max(node.vol_dim, 1)
      || ldb < std::max(node.nb(), 1))
  {
    throw std::invalid_argument(
      "apply_leaf_source_columns: invalid leading dimension");
  }

  if (node.source_dim == 0)
  {
    std::fill_n(
      cf_out,
      (std::size_t)ldcf * (std::size_t)nrhs,
      Real(0));
    std::fill_n(
      b_out,
      (std::size_t)ldb * (std::size_t)nrhs,
      Real(0));
    return;
  }

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    node.vol_dim,
    nrhs,
    node.source_dim,
    Real(1),
    node.Uf.data(),
    node.vol_dim,
    f_int,
    ldf,
    Real(0),
    cf_out,
    ldcf);

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    node.nb(),
    nrhs,
    node.source_dim,
    Real(1),
    node.Gf.data(),
    node.nb(),
    f_int,
    ldf,
    Real(0),
    b_out,
    ldb);
}

template<int D, class Real>
inline void apply_leaf_response_columns(
  const Node<D,Real>& node,
  const Real* lambda,
  int ldlambda,
  const Real* f_int,
  int ldf,
  int nrhs,
  Real* c_out,
  int ldc,
  Real* mu_hat_out,
  int ldmu)
{
  node.validate();
  if (!node.is_leaf)
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: node is not a leaf");
  }
  if (nrhs < 0)
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: negative nrhs");
  }
  if (nrhs == 0)
    return;
  if (!c_out || !mu_hat_out)
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: null output");
  }
  if (ldc < std::max(node.vol_dim, 1)
      || ldmu < std::max(node.nb(), 1))
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: invalid output leading dimension");
  }
  if (lambda && ldlambda < std::max(node.nb(), 1))
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: invalid lambda leading dimension");
  }
  if (f_int && ldf < std::max(node.source_dim, 1))
  {
    throw std::invalid_argument(
      "apply_leaf_response_columns: invalid source leading dimension");
  }

  std::fill_n(
    c_out,
    (std::size_t)ldc * (std::size_t)nrhs,
    Real(0));
  std::fill_n(
    mu_hat_out,
    (std::size_t)ldmu * (std::size_t)nrhs,
    Real(0));

  if (lambda && node.nb() > 0)
  {
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      node.vol_dim,
      nrhs,
      node.nb(),
      Real(1),
      node.Ulam.data(),
      node.vol_dim,
      lambda,
      ldlambda,
      Real(0),
      c_out,
      ldc);

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      node.nb(),
      nrhs,
      node.nb(),
      Real(1),
      node.S.data(),
      node.nb(),
      lambda,
      ldlambda,
      Real(0),
      mu_hat_out,
      ldmu);
  }

  if (f_int && node.source_dim > 0)
  {
    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      node.vol_dim,
      nrhs,
      node.source_dim,
      Real(1),
      node.Uf.data(),
      node.vol_dim,
      f_int,
      ldf,
      Real(1),
      c_out,
      ldc);

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      node.nb(),
      nrhs,
      node.source_dim,
      Real(1),
      node.Gf.data(),
      node.nb(),
      f_int,
      ldf,
      Real(1),
      mu_hat_out,
      ldmu);
  }
}

template<int D, class Real>
inline void set_leaf_source(
  Node<D,Real>& node,
  const Real* f_int)
{
  node.validate();
  if (node.source_dim > 0 && !f_int)
  {
    throw std::invalid_argument(
      "set_leaf_source: null f_int");
  }

#ifdef TIMING
  timer source_apply_timer;
  source_apply_timer.tic();
#endif

  const Real dummy = Real(0);
  apply_leaf_source_columns<D,Real>(
    node,
    f_int ? f_int : &dummy,
    std::max(node.source_dim, 1),
    1,
    node.cf.data(),
    std::max(node.vol_dim, 1),
    node.b.data(),
    std::max(node.nb(), 1));

#ifdef TIMING
  node.leaf_timing.source_solve_seconds = 0.0;
  node.leaf_timing.source_lsmr_iterations = 0;
  node.leaf_timing.source_actions.reset();
  node.leaf_timing.source_boundary_map_seconds =
    source_apply_timer.toc();
#endif

  node.validate();
}

/* Backward-compatible overload. The Leaf is now used only for consistency
   checks; source application is entirely map-based. */
template<int D, class Real>
inline void set_leaf_source(
  Node<D,Real>& node,
  const Leaf<D,Real>& leaf,
  const Real* f_int)
{
  if (node.kf != leaf.kf
      || node.nb() != leaf.nb
      || node.vol_dim != leaf.M
      || node.source_dim != leaf.m_int)
  {
    throw std::invalid_argument(
      "set_leaf_source: Node/Leaf dimension mismatch");
  }
  if (node.nfaces() != leaf.nface)
  {
    throw std::invalid_argument(
      "set_leaf_source: Node/Leaf face-count mismatch");
  }
  for (int face = 0; face < leaf.nface; ++face)
  {
    if (node.boundary_faces[(std::size_t)face]
        != leaf.face_key(face))
    {
      throw std::invalid_argument(
        "set_leaf_source: Node/Leaf face ordering mismatch");
    }
  }
  set_leaf_source<D,Real>(node, f_int);
}

/* Preserve the current Poisson calling path. */
template<int D, class Real>
inline void set_poisson_leaf_source(
  Node<D,Real>& node,
  const Real* f_int)
{
  set_leaf_source<D,Real>(node, f_int);
}

template<int D, class Real>
inline void set_poisson_leaf_source(
  Node<D,Real>& node,
  const Leaf<D,Real>& leaf,
  const Real* f_int)
{
  set_leaf_source<D,Real>(node, leaf, f_int);
}

template<class Real>
inline Real node_uniform_random(std::mt19937& rng)
{
  std::uniform_real_distribution<Real> dist(Real(-1), Real(1));
  return dist(rng);
}

template<int D, class Real>
inline Node<D,Real> make_dummy_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  int kf,
  int vol_dim,
  unsigned int seed = 1
)
{
  if (kf <= 0)
  {
    throw std::invalid_argument("make_dummy_leaf_node: kf must be positive");
  }
  if (vol_dim < 0)
  {
    throw std::invalid_argument("make_dummy_leaf_node: vol_dim must be nonnegative");
  }

  const auto& elem = mesh.element(element_id);

  Node<D,Real> node;
  node.kf = kf;
  node.boundary_faces.assign(elem.face_keys.begin(), elem.face_keys.end());
  node.is_leaf = true;
  node.leaf_element_id = element_id;
  node.vol_dim = vol_dim;
  node.source_dim = 0;

  const int nb = node.nb();
  node.S.assign((std::size_t)nb * nb, Real(0));
  node.b.assign((std::size_t)nb, Real(0));
  node.Ulam.assign((std::size_t)vol_dim * nb, Real(0));
  node.Uf.clear();
  node.Gf.clear();
  node.cf.assign((std::size_t)vol_dim, Real(0));

  std::mt19937 rng(seed + 7919u * static_cast<unsigned int>(element_id + 1));

  // Build a deterministic SPD-ish dummy DtN matrix S = A^T A + delta I.
  std::vector<Real> A((std::size_t)nb * nb, Real(0));
  for (int j = 0; j < nb; ++j)
  {
    for (int i = 0; i < nb; ++i)
    {
      A[(std::size_t)i + (std::size_t)nb * j] =
        node_uniform_random<Real>(rng);
    }
  }

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasTrans,
    CblasNoTrans,
    nb,
    nb,
    nb,
    Real(1),
    A.data(),
    nb,
    A.data(),
    nb,
    Real(0),
    node.S.data(),
    nb);

  const Real delta = Real(1) + Real(0.1) * Real(nb);
  for (int i = 0; i < nb; ++i)
  {
    node.S[(std::size_t)i + (std::size_t)nb * i] += delta;
  }

  for (int i = 0; i < nb; ++i)
  {
    node.b[(std::size_t)i] = node_uniform_random<Real>(rng);
  }

  for (int j = 0; j < nb; ++j)
  {
    for (int i = 0; i < vol_dim; ++i)
    {
      node.Ulam[(std::size_t)i + (std::size_t)vol_dim * j] =
        Real(0.25) * node_uniform_random<Real>(rng);
    }
  }

  for (int i = 0; i < vol_dim; ++i)
  {
    node.cf[(std::size_t)i] = Real(0.25) * node_uniform_random<Real>(rng);
  }

  node.validate();
  return node;
}

template<int D, class Real>
inline void apply_node_dtn(
  const Node<D,Real>& node,
  const Real* lambda,
  Real* mu
)
{
  if (!lambda || !mu)
  {
    throw std::invalid_argument("apply_node_dtn: null input/output");
  }
  node.validate();

  const int nb = node.nb();
  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    nb,
    1,
    nb,
    Real(1),
    node.S.data(),
    nb,
    lambda,
    nb,
    Real(0),
    mu,
    nb);

  for (int i = 0; i < nb; ++i)
  {
    mu[i] += node.b[(std::size_t)i];
  }
}

template<int D, class Real>
inline void reconstruct_leaf_volume(
  const Node<D,Real>& leaf,
  const Real* lambda,
  Real* c_out
)
{
  if (!lambda || !c_out)
  {
    throw std::invalid_argument("reconstruct_leaf_volume: null input/output");
  }
  leaf.validate();
  if (!leaf.is_leaf)
  {
    throw std::invalid_argument("reconstruct_leaf_volume: node is not a leaf");
  }

  const int nb = leaf.nb();
  const int M = leaf.vol_dim;
  if (M == 0)
  {
    return;
  }

  detail::BlasGemm<Real>::run(
    CblasColMajor,
    CblasNoTrans,
    CblasNoTrans,
    M,
    1,
    nb,
    Real(1),
    leaf.Ulam.data(),
    M,
    lambda,
    nb,
    Real(0),
    c_out,
    M);

  for (int i = 0; i < M; ++i)
  {
    c_out[i] += leaf.cf[(std::size_t)i];
  }
}

} // namespace jsimplex

#endif // JNODE_HH
