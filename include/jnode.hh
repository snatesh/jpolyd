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

/*
  Materialize the source-independent Poisson leaf maps

    c(lambda,0)       = Ulam lambda,
    mu_hat(lambda,0)  = S lambda,

  where mu_hat is Leaf::apply's augmented flux

    mu_hat = F c + tau (T c - lambda).

  leaf_out is initialized here and is intended to be retained by the caller so
  that set_poisson_leaf_source() can reuse the same geometry, local operators,
  tau scaling, and LSMR configuration for multiple right-hand sides.
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

  Node<D,Real> node;
  node.kf = leaf_out.kf;
  node.boundary_faces.assign(elem.face_keys.begin(), elem.face_keys.end());
  node.is_leaf = true;
  node.leaf_element_id = element_id;
  node.vol_dim = leaf_out.M;

  const int nb = node.nb();
  const int M = node.vol_dim;
  if (nb != leaf_out.nb)
  {
    throw std::runtime_error(
      "make_poisson_homogeneous_leaf_node: node/Leaf boundary size mismatch");
  }
  for (int f = 0; f < leaf_out.nface; ++f)
  {
    if (node.boundary_faces[(std::size_t)f] != leaf_out.face_key(f))
    {
      throw std::runtime_error(
        "make_poisson_homogeneous_leaf_node: Mesh/Leaf face ordering mismatch");
    }
  }

  node.S.assign((std::size_t)nb * nb, Real(0));
  node.b.assign((std::size_t)nb, Real(0));
  node.Ulam.assign((std::size_t)M * nb, Real(0));
  node.cf.assign((std::size_t)M, Real(0));

  std::vector<Real> lambda((std::size_t)nb, Real(0));
  std::vector<Real> c((std::size_t)M, Real(0));
  std::vector<Real> trace((std::size_t)nb, Real(0));
  std::vector<Real> raw_flux((std::size_t)nb, Real(0));
  std::vector<Real> aug_flux((std::size_t)nb, Real(0));

  // Leaf::apply currently requires a non-null f_int pointer even when m_int=0.
  std::vector<Real> f0((std::size_t)std::max(1, leaf_out.m_int), Real(0));

  for (int j = 0; j < nb; ++j)
  {
    std::fill(lambda.begin(), lambda.end(), Real(0));
    lambda[(std::size_t)j] = Real(1);

    std::fill(c.begin(), c.end(), Real(0));
    leaf_out.apply(
      lambda.data(),
      f0.data(),
      c.data(),
      trace.data(),
      raw_flux.data(),
      aug_flux.data());

    for (int i = 0; i < M; ++i)
    {
      node.Ulam[(std::size_t)i + (std::size_t)M * j] = c[(std::size_t)i];
    }
    for (int i = 0; i < nb; ++i)
    {
      node.S[(std::size_t)i + (std::size_t)nb * j] = aug_flux[(std::size_t)i];
    }
  }

  node.validate();
  return node;
}

/*
  Set the source-dependent affine terms on an already materialized homogeneous
  Poisson leaf:

    c(0,f)       = cf,
    mu_hat(0,f)  = b.

  S and Ulam are left unchanged.  Internal-node b vectors and merge trace
  offsets must subsequently be recomputed by the source upward pass.
*/
template<int D, class Real>
inline void set_poisson_leaf_source(
  Node<D,Real>& node,
  const Leaf<D,Real>& leaf,
  const Real* f_int)
{
  node.validate();
  if (!node.is_leaf)
  {
    throw std::invalid_argument(
      "set_poisson_leaf_source: node is not a leaf");
  }
  if (node.kf != leaf.kf || node.nb() != leaf.nb || node.vol_dim != leaf.M)
  {
    throw std::invalid_argument(
      "set_poisson_leaf_source: Node/Leaf dimension mismatch");
  }
  if (node.nfaces() != leaf.nface)
  {
    throw std::invalid_argument(
      "set_poisson_leaf_source: Node/Leaf face-count mismatch");
  }
  for (int f = 0; f < leaf.nface; ++f)
  {
    if (node.boundary_faces[(std::size_t)f] != leaf.face_key(f))
    {
      throw std::invalid_argument(
        "set_poisson_leaf_source: Node/Leaf face ordering mismatch");
    }
  }
  if (!f_int && leaf.m_int > 0)
  {
    throw std::invalid_argument(
      "set_poisson_leaf_source: null f_int");
  }

  const int nb = node.nb();
  const int M = node.vol_dim;

  std::vector<Real> lambda0((std::size_t)nb, Real(0));
  std::vector<Real> trace((std::size_t)nb, Real(0));
  std::vector<Real> raw_flux((std::size_t)nb, Real(0));

  // Keep the pointer non-null for the degree range where m_int=0.
  const Real f_dummy = Real(0);
  const Real* f_ptr = f_int ? f_int : &f_dummy;

  std::vector<Real> cf_new((std::size_t)M, Real(0));
  std::vector<Real> b_new((std::size_t)nb, Real(0));

  leaf.apply(
    lambda0.data(),
    f_ptr,
    cf_new.data(),
    trace.data(),
    raw_flux.data(),
    b_new.data());

  node.cf.swap(cf_new);
  node.b.swap(b_new);
  node.validate();
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
