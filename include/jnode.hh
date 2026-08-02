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

  int nAE = 0;
  int nBE = 0;
  int nI = 0;
};

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

  // Leaf reconstruction map:
  //   c = Ulam lambda + cf.
  // Ulam is vol_dim x nb, column-major.  Some notes may call this Wlam.
  std::vector<Real> Ulam;
  std::vector<Real> cf;

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
      if (static_cast<int>(cf.size()) != vol_dim)
      {
        throw std::runtime_error("Node::validate: cf has wrong size");
      }
    }
  }
};

namespace node_detail {

template<int D, class Real>
inline Node<D,Real> materialize_homogeneous_leaf_node(
  const Mesh<D,Real>& mesh,
  int element_id,
  const Leaf<D,Real>& leaf)
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

  const int nb = node.nb();
  const int M = node.vol_dim;
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
  node.cf.assign(
    (std::size_t)M,
    Real(0));

  std::vector<Real> lambda(
    (std::size_t)nb,
    Real(0));

  typename Leaf<D,Real>::SolveWorkspace
    solve_workspace(leaf);

  /*
    First construct only the coefficient map Ulam. This performs nb LSMR
    solves while reusing all reverse-communication vectors and does not
    traverse the trace or flux maps inside the column loop.
  */
  for (int column = 0;
       column < nb;
       ++column)
  {
    lambda[(std::size_t)column] =
      Real(1);

    leaf.solve_coefficients_zero_source(
      lambda.data(),
      node.Ulam.data()
        + (std::size_t)M
          * (std::size_t)column,
      solve_workspace);

    lambda[(std::size_t)column] =
      Real(0);
  }

  /*
    Apply the boundary maps once to the complete dense coefficient panel.
    Dense mode uses GEMM; MatrixFree and Verify use the existing CSC x dense
    traversals.
  */
  std::vector<Real> trace_map(
    (std::size_t)nb * nb,
    Real(0));

  leaf.apply_trace_columns(
    node.Ulam.data(),
    M,
    nb,
    trace_map.data(),
    nb);
  leaf.apply_flux_columns(
    node.Ulam.data(),
    M,
    nb,
    node.S.data(),
    nb);

  // S already contains F Ulam. Add tau (T Ulam - I).
  for (int column = 0;
       column < nb;
       ++column)
  {
    for (int row = 0;
         row < nb;
         ++row)
    {
      const std::size_t index =
        (std::size_t)row
        + (std::size_t)nb
          * (std::size_t)column;

      const Real identity =
        row == column
        ? Real(1)
        : Real(0);

      node.S[index] +=
        leaf.tau_rows[(std::size_t)row]
        * (
            trace_map[index]
            - identity
          );
    }
  }

  node.validate();
  return node;
}

} // namespace node_detail

/*
  Materialize source-independent elliptic leaf maps

    c(lambda,0)       = Ulam lambda,
    mu_hat(lambda,0)  = S lambda,

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
  Real tau_C = Real(10),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000)
{
  if (!(tau_C > Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tau_C must be positive");
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
    tau_C,
    opts);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out);
}

/*
  Mode-aware elliptic leaf materialization.

  Dense mode preserves the explicit local matrices. MatrixFree stores only
  the action workspace and the final HPS-facing Ulam/S maps. Verify retains
  both leaf backends and compares them according to leaf_options.

  In MatrixFree and Verify modes Leaf copies the enabled coefficient arrays,
  because set_leaf_source and homogeneous-map materialization continue to
  apply the local operator after this function returns.
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
  Real tau_C = Real(10),
  Real atol = Real(1e-14),
  Real btol = Real(1e-14),
  int itnlim = 5000)
{
  if (!(tau_C > Real(0)))
  {
    throw std::invalid_argument(
      "make_elliptic_homogeneous_leaf_node: tau_C must be positive");
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
    tau_C,
    opts,
    leaf_options);

  return node_detail::materialize_homogeneous_leaf_node<D,Real>(
    mesh,
    element_id,
    leaf_out);
}



/*
  Materialize the source-independent Poisson leaf maps

    c(lambda,0)       = Ulam lambda,
    mu_hat(lambda,0)  = S lambda,

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
  int itnlim = 5000)
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
    leaf_out);
}

/*
  Mode-aware Poisson leaf materialization. The legacy overload above remains
  Dense by default; this overload permits MatrixFree and Verify without
  changing old callers.
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
  int itnlim = 5000)
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
    leaf_out);
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
inline void set_leaf_source(
  Node<D,Real>& node,
  const Leaf<D,Real>& leaf,
  const Real* f_int)
{
  node.validate();
  if (!node.is_leaf)
  {
    throw std::invalid_argument(
      "set_leaf_source: node is not a leaf");
  }
  if (node.kf != leaf.kf
      || node.nb() != leaf.nb
      || node.vol_dim != leaf.M)
  {
    throw std::invalid_argument(
      "set_leaf_source: Node/Leaf dimension mismatch");
  }
  if (node.nfaces() != leaf.nface)
  {
    throw std::invalid_argument(
      "set_leaf_source: Node/Leaf face-count mismatch");
  }
  for (int face = 0;
       face < leaf.nface;
       ++face)
  {
    if (node.boundary_faces[(std::size_t)face]
        != leaf.face_key(face))
    {
      throw std::invalid_argument(
        "set_leaf_source: Node/Leaf face ordering mismatch");
    }
  }
  if (!f_int && leaf.m_int > 0)
  {
    throw std::invalid_argument(
      "set_leaf_source: null f_int");
  }

  const int nb = node.nb();
  const int M = node.vol_dim;

  std::vector<Real> lambda0(
    (std::size_t)nb,
    Real(0));
  std::vector<Real> trace(
    (std::size_t)nb,
    Real(0));

  // Keep the pointer non-null for the degree range where m_int=0.
  const Real f_dummy = Real(0);
  const Real* f_ptr =
    f_int ? f_int : &f_dummy;

  std::vector<Real> cf_new(
    (std::size_t)M,
    Real(0));
  std::vector<Real> b_new(
    (std::size_t)nb,
    Real(0));

  typename Leaf<D,Real>::SolveWorkspace
    solve_workspace(leaf);

  leaf.solve_coefficients(
    lambda0.data(),
    f_ptr,
    cf_new.data(),
    solve_workspace);

  leaf.apply_trace_columns(
    cf_new.data(),
    M,
    1,
    trace.data(),
    nb);
  leaf.apply_flux_columns(
    cf_new.data(),
    M,
    1,
    b_new.data(),
    nb);

  // b_new already contains F cf. Since lambda=0, add tau T cf.
  for (int row = 0;
       row < nb;
       ++row)
  {
    b_new[(std::size_t)row] +=
      leaf.tau_rows[(std::size_t)row]
      * trace[(std::size_t)row];
  }

  node.cf.swap(cf_new);
  node.b.swap(b_new);
  node.validate();
}

/* Preserve the current Poisson calling path. */
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

  const int nb = node.nb();
  node.S.assign((std::size_t)nb * nb, Real(0));
  node.b.assign((std::size_t)nb, Real(0));
  node.Ulam.assign((std::size_t)vol_dim * nb, Real(0));
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
