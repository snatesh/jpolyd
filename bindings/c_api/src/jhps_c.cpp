#include <jhps_c.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include <jelliptic.hh>
#include <jmesh.hh>
#include <jprecomp.hh>
#include <jnode.hh>
#include <jmerge.hh>

namespace {

template<class Real>
Real inf_norm(const std::vector<Real>& x)
{
  Real m = Real(0);
  for (Real v : x)
  {
    m = std::max(m, std::abs(v));
  }
  return m;
}

template<class Real>
jsimplex::LeafOptions<Real> leaf_options_from_c(
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve)
{
  jsimplex::LeafOptions<Real> options;

  switch (leaf_operator_mode)
  {
    case JHPS_LEAF_OPERATOR_DENSE:
      options.operator_mode =
        jsimplex::LeafOperatorMode::Dense;
      break;

    case JHPS_LEAF_OPERATOR_MATRIX_FREE:
      options.operator_mode =
        jsimplex::LeafOperatorMode::MatrixFree;
      break;

    case JHPS_LEAF_OPERATOR_VERIFY:
      options.operator_mode =
        jsimplex::LeafOperatorMode::Verify;
      break;

    default:
      throw std::invalid_argument(
        "HPS mesh-tree solve: invalid leaf_operator_mode");
  }

  if (leaf_verify_tolerance > 0.0)
  {
    options.verify_tolerance =
      static_cast<Real>(leaf_verify_tolerance);
  }

  options.verify_each_solve =
    leaf_verify_each_solve != 0;

  return options;
}

inline const char* leaf_operator_mode_name(
  jsimplex::LeafOperatorMode mode)
{
  switch (mode)
  {
    case jsimplex::LeafOperatorMode::Dense:
      return "Dense";
    case jsimplex::LeafOperatorMode::MatrixFree:
      return "MatrixFree";
    case jsimplex::LeafOperatorMode::Verify:
      return "Verify";
  }
  return "Unknown";
}

template<int D, class Real>
void validate_leaf_backend_storage(
  const jsimplex::Leaf<D,Real>& leaf)
{
  if (leaf.operator_mode ==
      jsimplex::LeafOperatorMode::MatrixFree)
  {
    if (leaf.has_dense_local_operator()
        || !leaf.L.empty()
        || !leaf.T.empty()
        || !leaf.F.empty()
        || !leaf.A_tau.empty())
    {
      throw std::runtime_error(
        "HPS mesh-tree solve: MatrixFree leaf retained dense local matrices");
    }
  }
  else if (!leaf.has_dense_local_operator())
  {
    throw std::runtime_error(
      "HPS mesh-tree solve: Dense/Verify leaf is missing dense fallback matrices");
  }
}


template<class Real>
void add_scaled_identity_colmajor(std::vector<Real>& A, int n, Real alpha)
{
  if (static_cast<int>(A.size()) != n * n)
  {
    throw std::invalid_argument("add_scaled_identity_colmajor: size mismatch");
  }
  for (int i = 0; i < n; ++i)
  {
    A[(std::size_t)i + (std::size_t)n * i] += alpha;
  }
}

template<int D>
jsimplex::Mesh<D,double> build_mesh_from_c(
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor)
{
  if (nverts <= 0 || !vertex_ids || !coords_rowmajor)
  {
    throw std::invalid_argument("build_mesh_from_c: invalid vertices");
  }
  if (nelem <= 0 || !simplices_rowmajor)
  {
    throw std::invalid_argument("build_mesh_from_c: invalid simplices");
  }

  jsimplex::Mesh<D,double> mesh;

  for (int v = 0; v < nverts; ++v)
  {
    std::array<double,D> x{};
    for (int r = 0; r < D; ++r)
    {
      x[(std::size_t)r] = coords_rowmajor[(std::size_t)v * D + r];
    }
    mesh.add_vertex(vertex_ids[v], x);
  }

  for (int e = 0; e < nelem; ++e)
  {
    std::array<int,D + 1> simplex{};
    for (int a = 0; a < D + 1; ++a)
    {
      simplex[(std::size_t)a] = simplices_rowmajor[(std::size_t)e * (D + 1) + a];
    }
    mesh.add_simplex(simplex);
  }

  mesh.build(true);
  return mesh;
}

template<int D>
void print_face_key(const jsimplex::DSimplexFaceKey<D>& key)
{
  std::cout << "(";
  for (int i = 0; i < D; ++i)
  {
    if (i) std::cout << ",";
    std::cout << key[(std::size_t)i];
  }
  std::cout << ")";
}

template<int D>
void print_node_faces(const char* name, const jsimplex::Node<D,double>& node)
{
  std::cout << name << " boundary_faces=[";
  for (int i = 0; i < static_cast<int>(node.boundary_faces.size()); ++i)
  {
    if (i) std::cout << ", ";
    print_face_key<D>(node.boundary_faces[(std::size_t)i]);
  }
  std::cout << "]\n";
}

template<int D>
double g_value(const jsimplex::DSimplexFaceKey<D>& key, int r)
{
  double phase = 0.19 * double(r + 1);
  for (int i = 0; i < D; ++i)
  {
    phase += 0.113 * double(i + 1) * double(key[(std::size_t)i] + 1);
  }
  return 0.5 + std::sin(phase);
}

template<int D, class Real>
void fill_root_g(const jsimplex::Node<D,Real>& root, std::vector<Real>& g)
{
  const int kf = root.kf;
  const int nb = root.nb();
  g.assign((std::size_t)nb, Real(0));
  for (int f = 0; f < root.nfaces(); ++f)
  {
    for (int r = 0; r < kf; ++r)
    {
      g[(std::size_t)f * kf + r] = Real(g_value<D>(root.boundary_faces[(std::size_t)f], r));
    }
  }
}


template<int D, class Real>
void fill_root_g_from_face_data(
  const jsimplex::Mesh<D,Real>& mesh,
  const jsimplex::Node<D,Real>& root,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const Real* boundary_g_rowmajor,
  std::vector<Real>& g)
{
  using FaceKey = jsimplex::DSimplexFaceKey<D>;

  if (nboundary_faces <= 0)
  {
    throw std::invalid_argument(
      "fill_root_g_from_face_data: nboundary_faces must be positive");
  }
  if (!boundary_face_keys_rowmajor || !boundary_g_rowmajor)
  {
    throw std::invalid_argument(
      "fill_root_g_from_face_data: null boundary data");
  }
  if (root.nfaces() != nboundary_faces)
  {
    throw std::invalid_argument(
      "fill_root_g_from_face_data: boundary face count does not match root");
  }

  std::map<FaceKey,int> input_pos;
  for (int f = 0; f < nboundary_faces; ++f)
  {
    FaceKey key{};
    for (int i = 0; i < D; ++i)
    {
      key[(std::size_t)i] =
        boundary_face_keys_rowmajor[(std::size_t)f * D + i];
    }
    std::sort(key.begin(), key.end());

    if (!mesh.is_boundary_face(key))
    {
      throw std::invalid_argument(
        "fill_root_g_from_face_data: supplied face key is not a mesh boundary face");
    }
    if (!input_pos.emplace(key, f).second)
    {
      throw std::invalid_argument(
        "fill_root_g_from_face_data: duplicate boundary face key");
    }
  }

  const int kf = root.kf;
  g.assign((std::size_t)root.nb(), Real(0));
  for (int rf = 0; rf < root.nfaces(); ++rf)
  {
    const FaceKey& key = root.boundary_faces[(std::size_t)rf];
    const auto it = input_pos.find(key);
    if (it == input_pos.end())
    {
      throw std::invalid_argument(
        "fill_root_g_from_face_data: missing root boundary face data");
    }

    const int input_f = it->second;
    for (int r = 0; r < kf; ++r)
    {
      g[(std::size_t)rf * kf + r] =
        boundary_g_rowmajor[(std::size_t)input_f * kf + r];
    }
  }
}

template<int D, class Real>
std::map<jsimplex::DSimplexFaceKey<D>,int> face_position_map(
  const std::vector<jsimplex::DSimplexFaceKey<D>>& faces)
{
  std::map<jsimplex::DSimplexFaceKey<D>,int> pos;
  for (int i = 0; i < static_cast<int>(faces.size()); ++i)
  {
    pos[faces[(std::size_t)i]] = i;
  }
  return pos;
}

template<int D, class Real>
int node_face_position(const jsimplex::Node<D,Real>& node, const jsimplex::DSimplexFaceKey<D>& key)
{
  for (int i = 0; i < node.nfaces(); ++i)
  {
    if (node.boundary_faces[(std::size_t)i] == key)
    {
      return i;
    }
  }
  throw std::runtime_error("node_face_position: face not found on node");
}

template<int D, class Real>
std::vector<Real> solve_root_robin(
  const jsimplex::Node<D,Real>& root,
  Real alpha,
  Real beta,
  const std::vector<Real>& g,
  Real& root_res_inf)
{
  const int n = root.nb();
  if (static_cast<int>(g.size()) != n)
  {
    throw std::invalid_argument("solve_root_robin: g size mismatch");
  }

  std::vector<Real> M = root.S;
  for (Real& v : M)
  {
    v *= beta;
  }
  add_scaled_identity_colmajor(M, n, alpha);

  std::vector<Real> lambda = g;
  for (int i = 0; i < n; ++i)
  {
    lambda[(std::size_t)i] -= beta * root.b[(std::size_t)i];
  }
  jsimplex::hps_detail::dense_solve_in_place<Real>(n, 1, M, lambda);

  std::vector<Real> mu((std::size_t)n, Real(0));
  jsimplex::apply_node_dtn<D,Real>(root, lambda.data(), mu.data());

  std::vector<Real> res((std::size_t)n, Real(0));
  for (int i = 0; i < n; ++i)
  {
    res[(std::size_t)i] = alpha * lambda[(std::size_t)i] + beta * mu[(std::size_t)i] - g[(std::size_t)i];
  }
  root_res_inf = inf_norm(res);
  return lambda;
}

template<int D, class Real>
void check_merge_residuals(
  const jsimplex::Node<D,Real>& parent,
  const jsimplex::Node<D,Real>& A,
  const jsimplex::Node<D,Real>& B,
  const std::vector<Real>& lambda_parent,
  const std::vector<Real>& lambda_A,
  const std::vector<Real>& lambda_B,
  Real& interface_res_max,
  Real& parent_consistency_max)
{
  const auto& md = parent.merge;
  const int nP = parent.nb();
  const int nAE = md.nAE;
  const int nBE = md.nBE;
  const int nI = md.nI;

  std::vector<Real> mu_P((std::size_t)nP, Real(0));
  std::vector<Real> mu_A((std::size_t)A.nb(), Real(0));
  std::vector<Real> mu_B((std::size_t)B.nb(), Real(0));

  jsimplex::apply_node_dtn<D,Real>(parent, lambda_parent.data(), mu_P.data());
  jsimplex::apply_node_dtn<D,Real>(A, lambda_A.data(), mu_A.data());
  jsimplex::apply_node_dtn<D,Real>(B, lambda_B.data(), mu_B.data());

  std::vector<Real> iface((std::size_t)nI, Real(0));
  for (int i = 0; i < nI; ++i)
  {
    iface[(std::size_t)i] =
      mu_A[(std::size_t)md.A_I_idx[(std::size_t)i]]
      + mu_B[(std::size_t)md.B_I_idx[(std::size_t)i]];
  }
  interface_res_max = std::max(interface_res_max, inf_norm(iface));

  std::vector<Real> mu_children((std::size_t)nP, Real(0));
  for (int i = 0; i < nAE; ++i)
  {
    mu_children[(std::size_t)i] = mu_A[(std::size_t)md.A_E_idx[(std::size_t)i]];
  }
  for (int i = 0; i < nBE; ++i)
  {
    mu_children[(std::size_t)nAE + i] = mu_B[(std::size_t)md.B_E_idx[(std::size_t)i]];
  }

  std::vector<Real> diff((std::size_t)nP, Real(0));
  for (int i = 0; i < nP; ++i)
  {
    diff[(std::size_t)i] = mu_P[(std::size_t)i] - mu_children[(std::size_t)i];
  }
  parent_consistency_max = std::max(parent_consistency_max, inf_norm(diff));
}

template<int D, class Real>
std::vector<Real> solve_monolithic_skeleton(
  const jsimplex::Mesh<D,Real>& mesh,
  const std::vector<jsimplex::Node<D,Real>>& leaves,
  int kf,
  Real alpha,
  Real beta)
{
  using FaceKey = jsimplex::DSimplexFaceKey<D>;

  std::map<FaceKey,int> fpos;
  int nf = 0;
  for (const auto& kv : mesh.face_incidence)
  {
    fpos[kv.first] = nf++;
  }

  const int n = nf * kf;
  std::vector<Real> M((std::size_t)n * n, Real(0));
  std::vector<Real> rhs((std::size_t)n, Real(0));

  auto add_coeff = [&](int row, int col, Real val)
  {
    M[(std::size_t)row + (std::size_t)n * col] += val;
  };

  for (const auto& kv : mesh.face_incidence)
  {
    const FaceKey& face = kv.first;
    const auto& incs = kv.second;
    const int fp = fpos[face];

    if (incs.size() == 1)
    {
      const int e = incs[0].element_id;
      const auto& leaf = leaves[(std::size_t)e];
      const int lf = node_face_position<D,Real>(leaf, face);

      for (int r = 0; r < kf; ++r)
      {
        const int row = fp * kf + r;
        const int local_row = lf * kf + r;

        add_coeff(row, row, alpha);

        for (int cf = 0; cf < leaf.nfaces(); ++cf)
        {
          const int gfp = fpos[leaf.boundary_faces[(std::size_t)cf]];
          for (int cr = 0; cr < kf; ++cr)
          {
            const int local_col = cf * kf + cr;
            const int gcol = gfp * kf + cr;
            add_coeff(row, gcol,
              beta * leaf.S[(std::size_t)local_row + (std::size_t)leaf.nb() * local_col]);
          }
        }

        rhs[(std::size_t)row] = Real(g_value<D>(face, r)) - beta * leaf.b[(std::size_t)local_row];
      }
    }
    else if (incs.size() == 2)
    {
      for (int r = 0; r < kf; ++r)
      {
        const int row = fp * kf + r;
        Real brhs = Real(0);

        for (const auto& inc : incs)
        {
          const int e = inc.element_id;
          const auto& leaf = leaves[(std::size_t)e];
          const int lf = node_face_position<D,Real>(leaf, face);
          const int local_row = lf * kf + r;

          for (int cf = 0; cf < leaf.nfaces(); ++cf)
          {
            const int gfp = fpos[leaf.boundary_faces[(std::size_t)cf]];
            for (int cr = 0; cr < kf; ++cr)
            {
              const int local_col = cf * kf + cr;
              const int gcol = gfp * kf + cr;
              add_coeff(row, gcol,
                leaf.S[(std::size_t)local_row + (std::size_t)leaf.nb() * local_col]);
            }
          }

          brhs -= leaf.b[(std::size_t)local_row];
        }

        rhs[(std::size_t)row] = brhs;
      }
    }
    else
    {
      throw std::runtime_error("solve_monolithic_skeleton: non-manifold incidence count");
    }
  }

  jsimplex::hps_detail::dense_solve_in_place<Real>(n, 1, M, rhs);
  return rhs;
}

template<int D, class Real>
Real monolithic_trace_diff(
  const jsimplex::Mesh<D,Real>& mesh,
  const std::vector<jsimplex::Node<D,Real>>& leaves,
  const std::vector<std::vector<Real>>& leaf_lambdas,
  const std::vector<Real>& lambda_mono,
  int kf)
{
  std::map<jsimplex::DSimplexFaceKey<D>,int> fpos;
  int nf = 0;
  for (const auto& kv : mesh.face_incidence)
  {
    fpos[kv.first] = nf++;
  }

  Real maxdiff = Real(0);
  for (int e = 0; e < static_cast<int>(leaves.size()); ++e)
  {
    const auto& leaf = leaves[(std::size_t)e];
    const auto& lam = leaf_lambdas[(std::size_t)e];
    for (int f = 0; f < leaf.nfaces(); ++f)
    {
      const int gp = fpos[leaf.boundary_faces[(std::size_t)f]];
      for (int r = 0; r < kf; ++r)
      {
        const Real d = lam[(std::size_t)f * kf + r] - lambda_mono[(std::size_t)gp * kf + r];
        maxdiff = std::max(maxdiff, std::abs(d));
      }
    }
  }
  return maxdiff;
}

template<int D, class Real>
Real leaf_volume_norm(
  const std::vector<jsimplex::Node<D,Real>>& leaves,
  const std::vector<std::vector<Real>>& leaf_lambdas,
  int vol_dim)
{
  Real out = Real(0);
  if (vol_dim <= 0)
  {
    return out;
  }

  for (int e = 0; e < static_cast<int>(leaves.size()); ++e)
  {
    std::vector<Real> c((std::size_t)vol_dim, Real(0));
    jsimplex::reconstruct_leaf_volume<D,Real>(
      leaves[(std::size_t)e], leaf_lambdas[(std::size_t)e].data(), c.data());
    out = std::max(out, inf_norm(c));
  }
  return out;
}

template<int D>
void validate_common_inputs(int nelem, int expected_nelem, int kf, int vol_dim, double alpha, double beta)
{
  if (nelem != expected_nelem)
  {
    throw std::invalid_argument("HPS dummy test: unexpected element count");
  }
  if (kf <= 0)
  {
    throw std::invalid_argument("HPS dummy test: kf must be positive");
  }
  if (vol_dim < 0)
  {
    throw std::invalid_argument("HPS dummy test: vol_dim must be nonnegative");
  }
  if (alpha == 0.0 && beta == 0.0)
  {
    throw std::invalid_argument("HPS dummy test: alpha and beta cannot both be zero");
  }
}

template<int D>
void validate_mesh_tree_inputs(
  int nelem,
  int nmerge,
  const int* merge_pairs_rowmajor,
  int kf,
  int vol_dim,
  double alpha,
  double beta)
{
  if (nelem <= 0)
  {
    throw std::invalid_argument("HPS dummy mesh-tree test: nelem must be positive");
  }
  if (nmerge != nelem - 1)
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: nmerge must equal nelem - 1");
  }
  if (nmerge > 0 && !merge_pairs_rowmajor)
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: null merge_pairs_rowmajor");
  }
  if (kf <= 0)
  {
    throw std::invalid_argument("HPS dummy mesh-tree test: kf must be positive");
  }
  if (vol_dim < 0)
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: vol_dim must be nonnegative");
  }
  if (alpha == 0.0 && beta == 0.0)
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: alpha and beta cannot both be zero");
  }
}

template<int D>
int run_mesh_tree_test_impl(
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  validate_mesh_tree_inputs<D>(
    nelem, nmerge, merge_pairs_rowmajor, kf, vol_dim, alpha, beta);

  using Real = double;
  using Node = jsimplex::Node<D,Real>;

  const auto mesh = build_mesh_from_c<D>(
    nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  std::vector<Node> leaves;
  leaves.reserve((std::size_t)nelem);
  for (int e = 0; e < nelem; ++e)
  {
    leaves.push_back(jsimplex::make_dummy_leaf_node<D,Real>(
      mesh, e, kf, vol_dim, seed + 101u * unsigned(e + 1)));
  }

  const int total_nodes = nelem + nmerge;
  std::vector<Node> nodes((std::size_t)total_nodes);
  std::vector<unsigned char> built((std::size_t)total_nodes, 0);
  std::vector<unsigned char> active((std::size_t)total_nodes, 0);

  for (int e = 0; e < nelem; ++e)
  {
    nodes[(std::size_t)e] = leaves[(std::size_t)e];
    built[(std::size_t)e] = 1;
    active[(std::size_t)e] = 1;
  }

  int eliminated_nb = 0;
  for (int m = 0; m < nmerge; ++m)
  {
    const int parent_id = nelem + m;
    const int child_A = merge_pairs_rowmajor[(std::size_t)2 * m];
    const int child_B = merge_pairs_rowmajor[(std::size_t)2 * m + 1];

    if (child_A < 0 || child_A >= parent_id ||
        child_B < 0 || child_B >= parent_id)
    {
      throw std::invalid_argument(
        "HPS dummy mesh-tree test: merge child id is not a previously built node");
    }
    if (child_A == child_B)
    {
      throw std::invalid_argument(
        "HPS dummy mesh-tree test: a merge cannot use the same child twice");
    }
    if (!built[(std::size_t)child_A] || !built[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS dummy mesh-tree test: merge references an unbuilt child");
    }
    if (!active[(std::size_t)child_A] || !active[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS dummy mesh-tree test: merge child is not an active subtree root");
    }

    nodes[(std::size_t)parent_id] = jsimplex::merge_nodes<D,Real>(
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      child_A,
      child_B);

    built[(std::size_t)parent_id] = 1;
    active[(std::size_t)child_A] = 0;
    active[(std::size_t)child_B] = 0;
    active[(std::size_t)parent_id] = 1;
    eliminated_nb += nodes[(std::size_t)parent_id].merge.nI;

    if (verbose > 1)
    {
      const auto& parent = nodes[(std::size_t)parent_id];
      std::cout << "  merge " << m
                << ": parent=" << parent_id
                << " children=(" << child_A << "," << child_B << ")"
                << " interface_faces=" << parent.merge.I_faces.size()
                << " interface_nb=" << parent.merge.nI
                << " parent_faces=" << parent.nfaces() << "\n";
    }
  }

  const int root_id = total_nodes - 1;
  int active_count = 0;
  int active_id = -1;
  for (int node_id = 0; node_id < total_nodes; ++node_id)
  {
    if (active[(std::size_t)node_id])
    {
      ++active_count;
      active_id = node_id;
    }
  }
  if (active_count != 1 || active_id != root_id ||
      !built[(std::size_t)root_id])
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: merge pairs do not form one complete binary tree");
  }

  const Node& root = nodes[(std::size_t)root_id];
  if (root.nb() <= 0)
  {
    throw std::invalid_argument(
      "HPS dummy mesh-tree test: root has no exterior trace unknowns");
  }

  std::vector<Real> g;
  fill_root_g(root, g);
  Real root_res_inf = 0;
  std::vector<std::vector<Real>> node_lambdas((std::size_t)total_nodes);
  node_lambdas[(std::size_t)root_id] = solve_root_robin(
    root, Real(alpha), Real(beta), g, root_res_inf);

  Real iface_res_inf = 0;
  Real parent_res_inf = 0;
  for (int m = nmerge - 1; m >= 0; --m)
  {
    const int parent_id = nelem + m;
    const Node& parent = nodes[(std::size_t)parent_id];
    const int child_A = parent.merge.child_A;
    const int child_B = parent.merge.child_B;

    jsimplex::merge_reconstruct_child_traces<D,Real>(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id].data(),
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B]);

    check_merge_residuals(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id],
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B],
      iface_res_inf,
      parent_res_inf);
  }

  std::vector<std::vector<Real>> leaf_lambdas((std::size_t)nelem);
  for (int e = 0; e < nelem; ++e)
  {
    if (static_cast<int>(node_lambdas[(std::size_t)e].size()) !=
        leaves[(std::size_t)e].nb())
    {
      throw std::runtime_error(
        "HPS dummy mesh-tree test: downward pass did not recover a leaf trace");
    }
    leaf_lambdas[(std::size_t)e] = node_lambdas[(std::size_t)e];
  }

  const std::vector<Real> lambda_mono = solve_monolithic_skeleton(
    mesh, leaves, kf, Real(alpha), Real(beta));
  const Real mono_diff = monolithic_trace_diff(
    mesh, leaves, leaf_lambdas, lambda_mono, kf);
  const Real vol_norm = leaf_volume_norm(leaves, leaf_lambdas, vol_dim);

  if (verbose)
  {
    std::cout << "HPS dummy external mesh-tree test D=" << D
              << " nelem=" << nelem
              << " nmerge=" << nmerge
              << " kf=" << kf
              << " vol_dim=" << vol_dim << "\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    std::cout << "  root_id=" << root_id
              << " root_faces=" << root.nfaces()
              << " root_nb=" << root.nb()
              << " eliminated_nb=" << eliminated_nb << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  monolithic trace diff inf   = " << mono_diff << "\n";
    std::cout << "  leaf volume coeff norm inf  = " << vol_norm << "\n";
  }

  if (root_robin_residual_inf_out) *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out) *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out) *parent_consistency_residual_inf_out = parent_res_inf;
  if (monolithic_trace_residual_inf_out) *monolithic_trace_residual_inf_out = mono_diff;
  if (leaf_volume_norm_inf_out) *leaf_volume_norm_inf_out = vol_norm;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = eliminated_nb;
  return 0;
}


template<int D>
void validate_poisson_mesh_tree_inputs(
  int n,
  int nverts,
  int nelem,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* kappa,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  double* leaf_coeffs_elementmajor)
{
  if (n < 2)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: require polynomial degree n>=2");
  }
  if (nverts <= 0)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: nverts must be positive");
  }
  if (nelem <= 0)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: nelem must be positive");
  }
  if (nmerge != nelem - 1)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: nmerge must equal nelem - 1");
  }
  if (nmerge > 0 && !merge_pairs_rowmajor)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: null merge_pairs_rowmajor");
  }
  if (!kappa)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: null kappa");
  }
  if (!f_int_elementmajor)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: null f_int_elementmajor");
  }
  if (nboundary_faces <= 0 ||
      !boundary_face_keys_rowmajor || !boundary_g_rowmajor)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: invalid boundary data");
  }
  if (!(tau_C > 0.0))
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: tau_C must be positive");
  }
  if (alpha == 0.0 && beta == 0.0)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: alpha and beta cannot both be zero");
  }
  if (alpha == 0.0)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: pure Neumann requires the compatibility/gauge branch, which is not implemented in this entry point");
  }
  if (!leaf_coeffs_elementmajor)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: null leaf_coeffs_elementmajor");
  }
}

template<int D>
int run_mesh_tree_poisson_impl(
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  validate_poisson_mesh_tree_inputs<D>(
    n, nverts, nelem, nmerge, merge_pairs_rowmajor, kappa,
    f_int_elementmajor, nboundary_faces, boundary_face_keys_rowmajor,
    boundary_g_rowmajor, tau_C, alpha, beta, leaf_coeffs_elementmajor);

  using Real = double;
  using Node = jsimplex::Node<D,Real>;
  using Leaf = jsimplex::Leaf<D,Real>;

  const typename Leaf::Options leaf_options =
    leaf_options_from_c<Real>(
      leaf_operator_mode,
      leaf_verify_tolerance,
      leaf_verify_each_solve);

  const jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, q_pad, q_vol, q_face, kappa);

  const auto mesh = build_mesh_from_c<D>(
    nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  const int mesh_boundary_faces =
    static_cast<int>(mesh.boundary_face_keys().size());
  if (nboundary_faces != mesh_boundary_faces)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: supplied boundary face count does not match mesh");
  }

  std::vector<Leaf> poisson_leaves((std::size_t)nelem);
  std::vector<Node> leaves;
  leaves.reserve((std::size_t)nelem);

  for (int e = 0; e < nelem; ++e)
  {
    Node leaf_node;

    if (leaf_options.operator_mode ==
        jsimplex::LeafOperatorMode::Dense)
    {
      // Preserve the legacy dense call path exactly.
      leaf_node =
        jsimplex::make_poisson_homogeneous_leaf_node<D,Real>(
          mesh,
          e,
          pre,
          poisson_leaves[(std::size_t)e],
          Real(tau_C),
          Real(1e-14),
          Real(1e-14),
          5000);
    }
    else
    {
      leaf_node =
        jsimplex::make_poisson_homogeneous_leaf_node<D,Real>(
          mesh,
          e,
          pre,
          poisson_leaves[(std::size_t)e],
          leaf_options,
          Real(tau_C),
          Real(1e-14),
          Real(1e-14),
          5000);
    }

    validate_leaf_backend_storage(
      poisson_leaves[(std::size_t)e]);

    const Real* f_e =
      f_int_elementmajor + (std::size_t)e * pre.m_int;

    jsimplex::set_poisson_leaf_source<D,Real>(
      leaf_node,
      poisson_leaves[(std::size_t)e],
      f_e);

    leaves.push_back(std::move(leaf_node));
  }

  const int total_nodes = nelem + nmerge;
  std::vector<Node> nodes((std::size_t)total_nodes);
  std::vector<unsigned char> built((std::size_t)total_nodes, 0);
  std::vector<unsigned char> active((std::size_t)total_nodes, 0);

  for (int e = 0; e < nelem; ++e)
  {
    nodes[(std::size_t)e] = std::move(leaves[(std::size_t)e]);
    built[(std::size_t)e] = 1;
    active[(std::size_t)e] = 1;
  }

  int eliminated_nb = 0;
  for (int m = 0; m < nmerge; ++m)
  {
    const int parent_id = nelem + m;
    const int child_A = merge_pairs_rowmajor[(std::size_t)2 * m];
    const int child_B = merge_pairs_rowmajor[(std::size_t)2 * m + 1];

    if (child_A < 0 || child_A >= parent_id ||
        child_B < 0 || child_B >= parent_id)
    {
      throw std::invalid_argument(
        "HPS Poisson mesh-tree solve: merge child id is not a previously built node");
    }
    if (child_A == child_B)
    {
      throw std::invalid_argument(
        "HPS Poisson mesh-tree solve: a merge cannot use the same child twice");
    }
    if (!built[(std::size_t)child_A] || !built[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS Poisson mesh-tree solve: merge references an unbuilt child");
    }
    if (!active[(std::size_t)child_A] || !active[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS Poisson mesh-tree solve: merge child is not an active subtree root");
    }

    nodes[(std::size_t)parent_id] = jsimplex::merge_nodes<D,Real>(
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      child_A,
      child_B);

    built[(std::size_t)parent_id] = 1;
    active[(std::size_t)child_A] = 0;
    active[(std::size_t)child_B] = 0;
    active[(std::size_t)parent_id] = 1;
    eliminated_nb += nodes[(std::size_t)parent_id].merge.nI;

    if (verbose > 1)
    {
      const Node& parent = nodes[(std::size_t)parent_id];
      std::cout << "  merge " << m
                << ": parent=" << parent_id
                << " children=(" << child_A << "," << child_B << ")"
                << " interface_faces=" << parent.merge.I_faces.size()
                << " interface_nb=" << parent.merge.nI
                << " parent_faces=" << parent.nfaces() << "\n";
    }
  }

  const int root_id = total_nodes - 1;
  int active_count = 0;
  int active_id = -1;
  for (int node_id = 0; node_id < total_nodes; ++node_id)
  {
    if (active[(std::size_t)node_id])
    {
      ++active_count;
      active_id = node_id;
    }
  }
  if (active_count != 1 || active_id != root_id ||
      !built[(std::size_t)root_id])
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: merge pairs do not form one complete binary tree");
  }

  const Node& root = nodes[(std::size_t)root_id];
  if (root.kf != pre.kf)
  {
    throw std::runtime_error(
      "HPS Poisson mesh-tree solve: root/precompute kf mismatch");
  }
  if (root.nb() <= 0)
  {
    throw std::invalid_argument(
      "HPS Poisson mesh-tree solve: root has no exterior trace unknowns");
  }

  std::vector<Real> g;
  fill_root_g_from_face_data(
    mesh,
    root,
    nboundary_faces,
    boundary_face_keys_rowmajor,
    boundary_g_rowmajor,
    g);

  Real root_res_inf = Real(0);
  std::vector<std::vector<Real>> node_lambdas((std::size_t)total_nodes);
  node_lambdas[(std::size_t)root_id] = solve_root_robin(
    root, Real(alpha), Real(beta), g, root_res_inf);

  Real iface_res_inf = Real(0);
  Real parent_res_inf = Real(0);
  for (int m = nmerge - 1; m >= 0; --m)
  {
    const int parent_id = nelem + m;
    const Node& parent = nodes[(std::size_t)parent_id];
    const int child_A = parent.merge.child_A;
    const int child_B = parent.merge.child_B;

    jsimplex::merge_reconstruct_child_traces<D,Real>(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id].data(),
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B]);

    check_merge_residuals(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id],
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B],
      iface_res_inf,
      parent_res_inf);
  }

  Real leaf_coeff_norm_inf = Real(0);
  for (int e = 0; e < nelem; ++e)
  {
    const Node& leaf_node = nodes[(std::size_t)e];
    const std::vector<Real>& lambda = node_lambdas[(std::size_t)e];
    if (static_cast<int>(lambda.size()) != leaf_node.nb())
    {
      throw std::runtime_error(
        "HPS Poisson mesh-tree solve: downward pass did not recover a leaf trace");
    }
    if (leaf_node.vol_dim != pre.M)
    {
      throw std::runtime_error(
        "HPS Poisson mesh-tree solve: leaf/precompute volume dimension mismatch");
    }

    Real* c_e =
      leaf_coeffs_elementmajor + (std::size_t)e * pre.M;
    jsimplex::reconstruct_leaf_volume<D,Real>(
      leaf_node,
      lambda.data(),
      c_e);

    for (int i = 0; i < pre.M; ++i)
    {
      leaf_coeff_norm_inf = std::max(
        leaf_coeff_norm_inf, std::abs(c_e[(std::size_t)i]));
    }
  }

  if (verbose)
  {
    std::cout << "HPS Poisson external mesh-tree solve D=" << D
              << " n=" << pre.n
              << " nelem=" << nelem
              << " nmerge=" << nmerge
              << " M=" << pre.M
              << " m_int=" << pre.m_int
              << " kf=" << pre.kf << "\n";
    std::cout << "  leaf operator mode="
              << leaf_operator_mode_name(
                   leaf_options.operator_mode)
              << "\n";
    std::cout << "  quadrature q_vol=" << pre.q_vol
              << " q_face=" << pre.q_face
              << " tau_C=" << tau_C
              << " LSMR(atol=1e-14, btol=1e-14, maxiter=5000)\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    std::cout << "  root_id=" << root_id
              << " root_faces=" << root.nfaces()
              << " root_nb=" << root.nb()
              << " eliminated_nb=" << eliminated_nb << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  leaf coefficient norm inf   = " << leaf_coeff_norm_inf << "\n";
  }

  if (root_robin_residual_inf_out)
    *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out)
    *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out)
    *parent_consistency_residual_inf_out = parent_res_inf;
  if (M_out) *M_out = pre.M;
  if (m_int_out) *m_int_out = pre.m_int;
  if (kf_out) *kf_out = pre.kf;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = eliminated_nb;
  return 0;
}

template<int D>
void validate_elliptic_mesh_tree_inputs(
  int n,
  int p2,
  int p1,
  int p0,
  const double* A_coeffs_elementmajor,
  const double* b_coeffs_elementmajor,
  const double* c_coeffs_elementmajor,
  int nverts,
  int nelem,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* kappa,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  double* leaf_coeffs_elementmajor)
{
  if (n < 2)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: require polynomial degree n>=2");
  }
  if (p2 < 0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: p2 must be nonnegative");
  }
  if (p1 < -1 || p0 < -1)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: p1 and p0 must be >=-1");
  }
  if (!A_coeffs_elementmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null A coefficients");
  }
  if (p1 >= 0 && !b_coeffs_elementmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null b coefficients for enabled first-order term");
  }
  if (p0 >= 0 && !c_coeffs_elementmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null c coefficients for enabled zero-order term");
  }
  if (nverts <= 0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: nverts must be positive");
  }
  if (nelem <= 0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: nelem must be positive");
  }
  if (nmerge != nelem - 1)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: nmerge must equal nelem - 1");
  }
  if (nmerge > 0 && !merge_pairs_rowmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null merge_pairs_rowmajor");
  }
  if (!kappa)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null kappa");
  }
  if (!f_int_elementmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null f_int_elementmajor");
  }
  if (nboundary_faces <= 0 ||
      !boundary_face_keys_rowmajor || !boundary_g_rowmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: invalid boundary data");
  }
  if (!(tau_C > 0.0))
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: tau_C must be positive");
  }
  if (alpha == 0.0 && beta == 0.0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: alpha and beta cannot both be zero");
  }
  if (alpha == 0.0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: pure Neumann requires the compatibility/gauge branch, which is not implemented in this entry point");
  }
  if (!leaf_coeffs_elementmajor)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: null leaf_coeffs_elementmajor");
  }
}

template<int D>
int run_mesh_tree_elliptic_impl(
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int p2,
  int p1,
  int p0,
  int assume_symmetric,
  const double* A_coeffs_elementmajor,
  const double* b_coeffs_elementmajor,
  const double* c_coeffs_elementmajor,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out,
  int* leaf_threads_used_out)
{
  validate_elliptic_mesh_tree_inputs<D>(
    n, p2, p1, p0,
    A_coeffs_elementmajor,
    b_coeffs_elementmajor,
    c_coeffs_elementmajor,
    nverts, nelem, nmerge, merge_pairs_rowmajor, kappa,
    f_int_elementmajor, nboundary_faces, boundary_face_keys_rowmajor,
    boundary_g_rowmajor, tau_C, alpha, beta, leaf_coeffs_elementmajor);

  using Real = double;
  using Node = jsimplex::Node<D,Real>;
  using Leaf = jsimplex::Leaf<D,Real>;
  using Plan = jsimplex::EllipticPlan<D,Real>;
  using DenseWorkspace =
    jsimplex::EllipticDenseWorkspace<D,Real>;
  using CoeffView =
    jsimplex::EllipticElementCoefficientsView<D,Real>;

  const typename Leaf::Options leaf_options =
    leaf_options_from_c<Real>(
      leaf_operator_mode,
      leaf_verify_tolerance,
      leaf_verify_each_solve);

  const jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, q_pad, q_vol, q_face, kappa);

  jsimplex::EllipticDegreeSpec degree_spec;
  degree_spec.p2 = p2;
  degree_spec.p1 = p1;
  degree_spec.p0 = p0;

  const Plan elliptic_plan(
    pre,
    degree_spec,
    assume_symmetric != 0);

  const auto mesh = build_mesh_from_c<D>(
    nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  const int mesh_boundary_faces =
    static_cast<int>(mesh.boundary_face_keys().size());
  if (nboundary_faces != mesh_boundary_faces)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: supplied boundary face count does not match mesh");
  }

  const int Mp2 = elliptic_plan.coefficient_size(2);
  const int Mp1 = elliptic_plan.coefficient_size(1);
  const int Mp0 = elliptic_plan.coefficient_size(0);

  const std::size_t A_stride =
    (std::size_t)D * (std::size_t)D * (std::size_t)Mp2;
  const std::size_t b_stride =
    (p1 >= 0) ? (std::size_t)D * (std::size_t)Mp1 : 0;
  const std::size_t c_stride =
    (p0 >= 0) ? (std::size_t)Mp0 : 0;

  std::vector<Leaf> elliptic_leaves((std::size_t)nelem);
  std::vector<Node> leaves((std::size_t)nelem);

  const int max_threads = std::max(1, omp_get_max_threads());

  // Dense preserves the old per-thread reusable assembly workspace. The
  // MatrixFree path owns only one action workspace per Leaf. Verify creates
  // its dense diagnostic workspace inside the mode-aware Leaf reset.
  std::vector<std::unique_ptr<DenseWorkspace>>
    dense_workspaces;
  if (leaf_options.operator_mode ==
      jsimplex::LeafOperatorMode::Dense)
  {
    dense_workspaces.reserve(
      (std::size_t)max_threads);
    for (int t = 0; t < max_threads; ++t)
    {
      dense_workspaces.emplace_back(
        std::make_unique<DenseWorkspace>(
          elliptic_plan));
    }
  }

  std::vector<unsigned char> thread_touched(
    (std::size_t)max_threads, 0);
  std::exception_ptr first_leaf_exception;

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();

    #pragma omp for schedule(static)
    for (int e = 0; e < nelem; ++e)
    {
      try
      {
        thread_touched[(std::size_t)tid] = 1;

        CoeffView coeffs;
        coeffs.A =
          A_coeffs_elementmajor + (std::size_t)e * A_stride;
        coeffs.b =
          (p1 >= 0)
          ? b_coeffs_elementmajor + (std::size_t)e * b_stride
          : nullptr;
        coeffs.c =
          (p0 >= 0)
          ? c_coeffs_elementmajor + (std::size_t)e * c_stride
          : nullptr;

        Node leaf_node;

        if (leaf_options.operator_mode ==
            jsimplex::LeafOperatorMode::Dense)
        {
          DenseWorkspace& work =
            *dense_workspaces[(std::size_t)tid];

          // Preserve the legacy dense call path and per-thread workspace reuse.
          leaf_node =
            jsimplex::make_elliptic_homogeneous_leaf_node<D,Real>(
              mesh,
              e,
              pre,
              elliptic_plan,
              coeffs,
              work,
              elliptic_leaves[(std::size_t)e],
              Real(tau_C),
              Real(1e-14),
              Real(1e-14),
              5000);
        }
        else
        {
          leaf_node =
            jsimplex::make_elliptic_homogeneous_leaf_node<D,Real>(
              mesh,
              e,
              pre,
              elliptic_plan,
              coeffs,
              elliptic_leaves[(std::size_t)e],
              leaf_options,
              Real(tau_C),
              Real(1e-14),
              Real(1e-14),
              5000);
        }

        validate_leaf_backend_storage(
          elliptic_leaves[(std::size_t)e]);

        const Real* f_e =
          f_int_elementmajor + (std::size_t)e * pre.m_int;

        jsimplex::set_leaf_source<D,Real>(
          leaf_node,
          elliptic_leaves[(std::size_t)e],
          f_e);

        leaves[(std::size_t)e] = std::move(leaf_node);
      }
      catch (const std::exception& ex)
      {
        const std::exception_ptr ep = std::make_exception_ptr(
          std::runtime_error(
            std::string("HPS elliptic mesh-tree solve: leaf ")
            + std::to_string(e)
            + " construction failed: "
            + ex.what()));

  #pragma omp critical(jhps_elliptic_leaf_exception)
        {
          if (!first_leaf_exception)
            first_leaf_exception = ep;
        }
      }
      catch (...)
      {
        const std::exception_ptr ep = std::make_exception_ptr(
          std::runtime_error(
            std::string("HPS elliptic mesh-tree solve: leaf ")
            + std::to_string(e)
            + " construction failed with an unknown exception"));

  #pragma omp critical(jhps_elliptic_leaf_exception)
        {
          if (!first_leaf_exception)
            first_leaf_exception = ep;
        }
      }
    }
  }

  if (first_leaf_exception)
    std::rethrow_exception(first_leaf_exception);

  int leaf_threads_used = 0;
  for (unsigned char used : thread_touched)
    leaf_threads_used += (used != 0);

  const int total_nodes = nelem + nmerge;
  std::vector<Node> nodes((std::size_t)total_nodes);
  std::vector<unsigned char> built((std::size_t)total_nodes, 0);
  std::vector<unsigned char> active((std::size_t)total_nodes, 0);

  for (int e = 0; e < nelem; ++e)
  {
    nodes[(std::size_t)e] = std::move(leaves[(std::size_t)e]);
    built[(std::size_t)e] = 1;
    active[(std::size_t)e] = 1;
  }

  int eliminated_nb = 0;
  for (int m = 0; m < nmerge; ++m)
  {
    const int parent_id = nelem + m;
    const int child_A = merge_pairs_rowmajor[(std::size_t)2 * m];
    const int child_B = merge_pairs_rowmajor[(std::size_t)2 * m + 1];

    if (child_A < 0 || child_A >= parent_id ||
        child_B < 0 || child_B >= parent_id)
    {
      throw std::invalid_argument(
        "HPS elliptic mesh-tree solve: merge child id is not a previously built node");
    }
    if (child_A == child_B)
    {
      throw std::invalid_argument(
        "HPS elliptic mesh-tree solve: a merge cannot use the same child twice");
    }
    if (!built[(std::size_t)child_A] || !built[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS elliptic mesh-tree solve: merge references an unbuilt child");
    }
    if (!active[(std::size_t)child_A] || !active[(std::size_t)child_B])
    {
      throw std::invalid_argument(
        "HPS elliptic mesh-tree solve: merge child is not an active subtree root");
    }

    nodes[(std::size_t)parent_id] = jsimplex::merge_nodes<D,Real>(
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      child_A,
      child_B);

    built[(std::size_t)parent_id] = 1;
    active[(std::size_t)child_A] = 0;
    active[(std::size_t)child_B] = 0;
    active[(std::size_t)parent_id] = 1;
    eliminated_nb += nodes[(std::size_t)parent_id].merge.nI;

    if (verbose > 1)
    {
      const Node& parent = nodes[(std::size_t)parent_id];
      std::cout << "  merge " << m
                << ": parent=" << parent_id
                << " children=(" << child_A << "," << child_B << ")"
                << " interface_faces=" << parent.merge.I_faces.size()
                << " interface_nb=" << parent.merge.nI
                << " parent_faces=" << parent.nfaces() << "\n";
    }
  }

  const int root_id = total_nodes - 1;
  int active_count = 0;
  int active_id = -1;
  for (int node_id = 0; node_id < total_nodes; ++node_id)
  {
    if (active[(std::size_t)node_id])
    {
      ++active_count;
      active_id = node_id;
    }
  }
  if (active_count != 1 || active_id != root_id ||
      !built[(std::size_t)root_id])
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: merge pairs do not form one complete binary tree");
  }

  const Node& root = nodes[(std::size_t)root_id];
  if (root.kf != pre.kf)
  {
    throw std::runtime_error(
      "HPS elliptic mesh-tree solve: root/precompute kf mismatch");
  }
  if (root.nb() <= 0)
  {
    throw std::invalid_argument(
      "HPS elliptic mesh-tree solve: root has no exterior trace unknowns");
  }

  std::vector<Real> g;
  fill_root_g_from_face_data(
    mesh,
    root,
    nboundary_faces,
    boundary_face_keys_rowmajor,
    boundary_g_rowmajor,
    g);

  Real root_res_inf = Real(0);
  std::vector<std::vector<Real>> node_lambdas((std::size_t)total_nodes);
  node_lambdas[(std::size_t)root_id] = solve_root_robin(
    root, Real(alpha), Real(beta), g, root_res_inf);

  Real iface_res_inf = Real(0);
  Real parent_res_inf = Real(0);
  for (int m = nmerge - 1; m >= 0; --m)
  {
    const int parent_id = nelem + m;
    const Node& parent = nodes[(std::size_t)parent_id];
    const int child_A = parent.merge.child_A;
    const int child_B = parent.merge.child_B;

    jsimplex::merge_reconstruct_child_traces<D,Real>(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id].data(),
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B]);

    check_merge_residuals(
      parent,
      nodes[(std::size_t)child_A],
      nodes[(std::size_t)child_B],
      node_lambdas[(std::size_t)parent_id],
      node_lambdas[(std::size_t)child_A],
      node_lambdas[(std::size_t)child_B],
      iface_res_inf,
      parent_res_inf);
  }

  Real leaf_coeff_norm_inf = Real(0);
  for (int e = 0; e < nelem; ++e)
  {
    const Node& leaf_node = nodes[(std::size_t)e];
    const std::vector<Real>& lambda = node_lambdas[(std::size_t)e];
    if (static_cast<int>(lambda.size()) != leaf_node.nb())
    {
      throw std::runtime_error(
        "HPS elliptic mesh-tree solve: downward pass did not recover a leaf trace");
    }
    if (leaf_node.vol_dim != pre.M)
    {
      throw std::runtime_error(
        "HPS elliptic mesh-tree solve: leaf/precompute volume dimension mismatch");
    }

    Real* c_e =
      leaf_coeffs_elementmajor + (std::size_t)e * pre.M;
    jsimplex::reconstruct_leaf_volume<D,Real>(
      leaf_node,
      lambda.data(),
      c_e);

    for (int i = 0; i < pre.M; ++i)
    {
      leaf_coeff_norm_inf = std::max(
        leaf_coeff_norm_inf, std::abs(c_e[(std::size_t)i]));
    }
  }

  if (verbose)
  {
    std::cout << "HPS elliptic external mesh-tree solve D=" << D
              << " n=" << pre.n
              << " nelem=" << nelem
              << " nmerge=" << nmerge
              << " M=" << pre.M
              << " m_int=" << pre.m_int
              << " kf=" << pre.kf << "\n";
    std::cout << "  leaf operator mode="
              << leaf_operator_mode_name(
                   leaf_options.operator_mode)
              << "\n";
    std::cout << "  coefficient degrees p2=" << p2
              << " p1=" << p1
              << " p0=" << p0
              << " Clenshaw_assume_symmetric="
              << (assume_symmetric != 0) << "\n";
    std::cout << "  quadrature q_vol=" << pre.q_vol
              << " q_face=" << pre.q_face
              << " tau_C=" << tau_C
              << " LSMR(atol=1e-14, btol=1e-14, maxiter=5000)\n";
    std::cout << "  leaf OpenMP threads used=" << leaf_threads_used
              << " max_threads=" << max_threads << "\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    std::cout << "  root_id=" << root_id
              << " root_faces=" << root.nfaces()
              << " root_nb=" << root.nb()
              << " eliminated_nb=" << eliminated_nb << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  leaf coefficient norm inf   = " << leaf_coeff_norm_inf << "\n";
  }

  if (root_robin_residual_inf_out)
    *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out)
    *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out)
    *parent_consistency_residual_inf_out = parent_res_inf;
  if (M_out) *M_out = pre.M;
  if (m_int_out) *m_int_out = pre.m_int;
  if (kf_out) *kf_out = pre.kf;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = eliminated_nb;
  if (leaf_threads_used_out) *leaf_threads_used_out = leaf_threads_used;
  return 0;
}

template<int D>
int run_two_leaf_test_impl(
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  validate_common_inputs<D>(nelem, 2, kf, vol_dim, alpha, beta);
  using Real = double;
  using Node = jsimplex::Node<D,Real>;

  const auto mesh = build_mesh_from_c<D>(nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  std::vector<Node> leaves;
  for (int e = 0; e < nelem; ++e)
  {
    leaves.push_back(jsimplex::make_dummy_leaf_node<D,Real>(mesh, e, kf, vol_dim, seed + 101u * unsigned(e + 1)));
  }

  Node root = jsimplex::merge_nodes<D,Real>(leaves[0], leaves[1], 0, 1);
  std::vector<Real> g;
  fill_root_g(root, g);
  Real root_res_inf = 0;
  std::vector<Real> lambda_root = solve_root_robin(root, Real(alpha), Real(beta), g, root_res_inf);

  std::vector<Real> lambda_A, lambda_B;
  jsimplex::merge_reconstruct_child_traces<D,Real>(root, leaves[0], leaves[1], lambda_root.data(), lambda_A, lambda_B);

  Real iface_res_inf = 0;
  Real parent_res_inf = 0;
  check_merge_residuals(root, leaves[0], leaves[1], lambda_root, lambda_A, lambda_B, iface_res_inf, parent_res_inf);

  std::vector<std::vector<Real>> leaf_lambdas = {lambda_A, lambda_B};
  const std::vector<Real> lambda_mono = solve_monolithic_skeleton(mesh, leaves, kf, Real(alpha), Real(beta));
  const Real mono_diff = monolithic_trace_diff(mesh, leaves, leaf_lambdas, lambda_mono, kf);
  const Real vol_norm = leaf_volume_norm(leaves, leaf_lambdas, vol_dim);

  if (verbose)
  {
    std::cout << "HPS dummy two-leaf test D=" << D
              << " nelem=" << nelem << " kf=" << kf
              << " vol_dim=" << vol_dim << "\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    print_node_faces<D>("  A", leaves[0]);
    print_node_faces<D>("  B", leaves[1]);
    print_node_faces<D>("  P", root);
    std::cout << "  root_nb=" << root.nb() << " eliminated_nb=" << root.merge.nI << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  monolithic trace diff inf   = " << mono_diff << "\n";
    std::cout << "  leaf volume coeff norm inf  = " << vol_norm << "\n";
  }

  if (root_robin_residual_inf_out) *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out) *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out) *parent_consistency_residual_inf_out = parent_res_inf;
  if (monolithic_trace_residual_inf_out) *monolithic_trace_residual_inf_out = mono_diff;
  if (leaf_volume_norm_inf_out) *leaf_volume_norm_inf_out = vol_norm;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = root.merge.nI;
  return 0;
}

template<int D>
int run_three_leaf_chain_test_impl(
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  validate_common_inputs<D>(nelem, 3, kf, vol_dim, alpha, beta);
  using Real = double;
  using Node = jsimplex::Node<D,Real>;

  const auto mesh = build_mesh_from_c<D>(nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  std::vector<Node> leaves;
  for (int e = 0; e < nelem; ++e)
  {
    leaves.push_back(jsimplex::make_dummy_leaf_node<D,Real>(mesh, e, kf, vol_dim, seed + 101u * unsigned(e + 1)));
  }

  Node P01 = jsimplex::merge_nodes<D,Real>(leaves[0], leaves[1], 0, 1);
  Node root = jsimplex::merge_nodes<D,Real>(P01, leaves[2], 10, 2);

  std::vector<Real> g;
  fill_root_g(root, g);
  Real root_res_inf = 0;
  std::vector<Real> lambda_root = solve_root_robin(root, Real(alpha), Real(beta), g, root_res_inf);

  std::vector<Real> lambda_P01, lambda_2;
  jsimplex::merge_reconstruct_child_traces<D,Real>(root, P01, leaves[2], lambda_root.data(), lambda_P01, lambda_2);

  std::vector<Real> lambda_0, lambda_1;
  jsimplex::merge_reconstruct_child_traces<D,Real>(P01, leaves[0], leaves[1], lambda_P01.data(), lambda_0, lambda_1);

  Real iface_res_inf = 0;
  Real parent_res_inf = 0;
  check_merge_residuals(root, P01, leaves[2], lambda_root, lambda_P01, lambda_2, iface_res_inf, parent_res_inf);
  check_merge_residuals(P01, leaves[0], leaves[1], lambda_P01, lambda_0, lambda_1, iface_res_inf, parent_res_inf);

  std::vector<std::vector<Real>> leaf_lambdas = {lambda_0, lambda_1, lambda_2};
  const std::vector<Real> lambda_mono = solve_monolithic_skeleton(mesh, leaves, kf, Real(alpha), Real(beta));
  const Real mono_diff = monolithic_trace_diff(mesh, leaves, leaf_lambdas, lambda_mono, kf);
  const Real vol_norm = leaf_volume_norm(leaves, leaf_lambdas, vol_dim);
  const int eliminated_nb = P01.merge.nI + root.merge.nI;

  if (verbose)
  {
    std::cout << "HPS dummy three-leaf chain test D=" << D
              << " nelem=" << nelem << " kf=" << kf
              << " vol_dim=" << vol_dim << "\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    print_node_faces<D>("  K0", leaves[0]);
    print_node_faces<D>("  K1", leaves[1]);
    print_node_faces<D>("  K2", leaves[2]);
    print_node_faces<D>("  P01", P01);
    print_node_faces<D>("  Root", root);
    std::cout << "  root_nb=" << root.nb() << " eliminated_nb=" << eliminated_nb << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  monolithic trace diff inf   = " << mono_diff << "\n";
    std::cout << "  leaf volume coeff norm inf  = " << vol_norm << "\n";
  }

  if (root_robin_residual_inf_out) *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out) *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out) *parent_consistency_residual_inf_out = parent_res_inf;
  if (monolithic_trace_residual_inf_out) *monolithic_trace_residual_inf_out = mono_diff;
  if (leaf_volume_norm_inf_out) *leaf_volume_norm_inf_out = vol_norm;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = eliminated_nb;
  return 0;
}

template<int D>
int run_four_leaf_balanced_test_impl(
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  validate_common_inputs<D>(nelem, 4, kf, vol_dim, alpha, beta);
  using Real = double;
  using Node = jsimplex::Node<D,Real>;

  const auto mesh = build_mesh_from_c<D>(nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor);

  std::vector<Node> leaves;
  for (int e = 0; e < nelem; ++e)
  {
    leaves.push_back(jsimplex::make_dummy_leaf_node<D,Real>(mesh, e, kf, vol_dim, seed + 101u * unsigned(e + 1)));
  }

  Node P01 = jsimplex::merge_nodes<D,Real>(leaves[0], leaves[1], 0, 1);
  Node P23 = jsimplex::merge_nodes<D,Real>(leaves[2], leaves[3], 2, 3);
  Node root = jsimplex::merge_nodes<D,Real>(P01, P23, 10, 23);

  std::vector<Real> g;
  fill_root_g(root, g);
  Real root_res_inf = 0;
  std::vector<Real> lambda_root = solve_root_robin(root, Real(alpha), Real(beta), g, root_res_inf);

  std::vector<Real> lambda_P01, lambda_P23;
  jsimplex::merge_reconstruct_child_traces<D,Real>(root, P01, P23, lambda_root.data(), lambda_P01, lambda_P23);

  std::vector<Real> lambda_0, lambda_1, lambda_2, lambda_3;
  jsimplex::merge_reconstruct_child_traces<D,Real>(P01, leaves[0], leaves[1], lambda_P01.data(), lambda_0, lambda_1);
  jsimplex::merge_reconstruct_child_traces<D,Real>(P23, leaves[2], leaves[3], lambda_P23.data(), lambda_2, lambda_3);

  Real iface_res_inf = 0;
  Real parent_res_inf = 0;
  check_merge_residuals(root, P01, P23, lambda_root, lambda_P01, lambda_P23, iface_res_inf, parent_res_inf);
  check_merge_residuals(P01, leaves[0], leaves[1], lambda_P01, lambda_0, lambda_1, iface_res_inf, parent_res_inf);
  check_merge_residuals(P23, leaves[2], leaves[3], lambda_P23, lambda_2, lambda_3, iface_res_inf, parent_res_inf);

  std::vector<std::vector<Real>> leaf_lambdas = {lambda_0, lambda_1, lambda_2, lambda_3};
  const std::vector<Real> lambda_mono = solve_monolithic_skeleton(mesh, leaves, kf, Real(alpha), Real(beta));
  const Real mono_diff = monolithic_trace_diff(mesh, leaves, leaf_lambdas, lambda_mono, kf);
  const Real vol_norm = leaf_volume_norm(leaves, leaf_lambdas, vol_dim);
  const int eliminated_nb = P01.merge.nI + P23.merge.nI + root.merge.nI;

  if (verbose)
  {
    std::cout << "HPS dummy four-leaf balanced test D=" << D
              << " nelem=" << nelem << " kf=" << kf
              << " vol_dim=" << vol_dim << "\n";
    std::cout << "  mesh vertices=" << mesh.num_vertices()
              << " elements=" << mesh.num_elements()
              << " faces=" << mesh.face_incidence.size() << "\n";
    print_node_faces<D>("  K0", leaves[0]);
    print_node_faces<D>("  K1", leaves[1]);
    print_node_faces<D>("  K2", leaves[2]);
    print_node_faces<D>("  K3", leaves[3]);
    print_node_faces<D>("  P01", P01);
    print_node_faces<D>("  P23", P23);
    print_node_faces<D>("  Root", root);
    std::cout << "  root_nb=" << root.nb() << " eliminated_nb=" << eliminated_nb << "\n";
    std::cout << "  root Robin residual inf      = " << root_res_inf << "\n";
    std::cout << "  interface flux residual inf = " << iface_res_inf << "\n";
    std::cout << "  parent consistency inf      = " << parent_res_inf << "\n";
    std::cout << "  monolithic trace diff inf   = " << mono_diff << "\n";
    std::cout << "  leaf volume coeff norm inf  = " << vol_norm << "\n";
  }

  if (root_robin_residual_inf_out) *root_robin_residual_inf_out = root_res_inf;
  if (interface_flux_residual_inf_out) *interface_flux_residual_inf_out = iface_res_inf;
  if (parent_consistency_residual_inf_out) *parent_consistency_residual_inf_out = parent_res_inf;
  if (monolithic_trace_residual_inf_out) *monolithic_trace_residual_inf_out = mono_diff;
  if (leaf_volume_norm_inf_out) *leaf_volume_norm_inf_out = vol_norm;
  if (root_nb_out) *root_nb_out = root.nb();
  if (interface_nb_out) *interface_nb_out = eliminated_nb;
  return 0;
}

template<int D>
int dispatch_test(
  int kind,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  if (kind == 2)
  {
    return run_two_leaf_test_impl<D>(nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  if (kind == 3)
  {
    return run_three_leaf_chain_test_impl<D>(nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  if (kind == 4)
  {
    return run_four_leaf_balanced_test_impl<D>(nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  throw std::invalid_argument("dispatch_test: unknown kind");
}

int entry_dispatch(
  int kind,
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  switch (D)
  {
    case 1:
      return dispatch_test<1>(kind, nverts, vertex_ids, coords_rowmajor, nelem,
        simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 2:
      return dispatch_test<2>(kind, nverts, vertex_ids, coords_rowmajor, nelem,
        simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 3:
      return dispatch_test<3>(kind, nverts, vertex_ids, coords_rowmajor, nelem,
        simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 4:
      return dispatch_test<4>(kind, nverts, vertex_ids, coords_rowmajor, nelem,
        simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 5:
      return dispatch_test<5>(kind, nverts, vertex_ids, coords_rowmajor, nelem,
        simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    default:
      std::cerr << "jhps_dummy test: unsupported D=" << D
                << "; supported range is 1..5\n";
      return -2;
  }
}

int entry_mesh_tree_dispatch(
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  switch (D)
  {
    case 1:
      return run_mesh_tree_test_impl<1>(
        nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
        nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 2:
      return run_mesh_tree_test_impl<2>(
        nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
        nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 3:
      return run_mesh_tree_test_impl<3>(
        nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
        nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 4:
      return run_mesh_tree_test_impl<4>(
        nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
        nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    case 5:
      return run_mesh_tree_test_impl<5>(
        nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
        nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
        root_robin_residual_inf_out, interface_flux_residual_inf_out,
        parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
        leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
    default:
      std::cerr << "jhps_dummy_mesh_tree_test: unsupported D=" << D
                << "; supported range is 1..5\n";
      return -2;
  }
}


int entry_poisson_mesh_tree_dispatch(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
#define JHPS_POISSON_DISPATCH_CASE(DIM) \
  case DIM: \
    return run_mesh_tree_poisson_impl<DIM>( \
      n, q_pad, q_vol, q_face, kappa, \
      nverts, vertex_ids, coords_rowmajor, \
      nelem, simplices_rowmajor, nmerge, merge_pairs_rowmajor, \
      f_int_elementmajor, nboundary_faces, boundary_face_keys_rowmajor, \
      boundary_g_rowmajor, tau_C, alpha, beta, verbose, \
      leaf_operator_mode, leaf_verify_tolerance, leaf_verify_each_solve, \
      leaf_coeffs_elementmajor, root_robin_residual_inf_out, \
      interface_flux_residual_inf_out, parent_consistency_residual_inf_out, \
      M_out, m_int_out, kf_out, root_nb_out, interface_nb_out)

  switch (D)
  {
    JHPS_POISSON_DISPATCH_CASE(1);
    JHPS_POISSON_DISPATCH_CASE(2);
    JHPS_POISSON_DISPATCH_CASE(3);
    JHPS_POISSON_DISPATCH_CASE(4);
    JHPS_POISSON_DISPATCH_CASE(5);
    default:
      std::cerr << "jhps_poisson_mesh_tree_solve: unsupported D=" << D
                << "; supported range is 1..5\n";
      return -2;
  }

#undef JHPS_POISSON_DISPATCH_CASE
}

int entry_elliptic_mesh_tree_dispatch(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int p2,
  int p1,
  int p0,
  int assume_symmetric,
  const double* A_coeffs_elementmajor,
  const double* b_coeffs_elementmajor,
  const double* c_coeffs_elementmajor,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out,
  int* leaf_threads_used_out)
{
#define JHPS_ELLIPTIC_DISPATCH_CASE(DIM) \
  case DIM: \
    return run_mesh_tree_elliptic_impl<DIM>( \
      n, q_pad, q_vol, q_face, kappa, \
      p2, p1, p0, assume_symmetric, \
      A_coeffs_elementmajor, b_coeffs_elementmajor, c_coeffs_elementmajor, \
      nverts, vertex_ids, coords_rowmajor, \
      nelem, simplices_rowmajor, nmerge, merge_pairs_rowmajor, \
      f_int_elementmajor, nboundary_faces, boundary_face_keys_rowmajor, \
      boundary_g_rowmajor, tau_C, alpha, beta, verbose, \
      leaf_operator_mode, leaf_verify_tolerance, leaf_verify_each_solve, \
      leaf_coeffs_elementmajor, root_robin_residual_inf_out, \
      interface_flux_residual_inf_out, parent_consistency_residual_inf_out, \
      M_out, m_int_out, kf_out, root_nb_out, interface_nb_out, \
      leaf_threads_used_out)

  switch (D)
  {
    JHPS_ELLIPTIC_DISPATCH_CASE(1);
    JHPS_ELLIPTIC_DISPATCH_CASE(2);
    JHPS_ELLIPTIC_DISPATCH_CASE(3);
    JHPS_ELLIPTIC_DISPATCH_CASE(4);
    JHPS_ELLIPTIC_DISPATCH_CASE(5);
    default:
      std::cerr << "jhps_elliptic_mesh_tree_solve: unsupported D=" << D
                << "; supported range is 1..5\n";
      return -2;
  }

#undef JHPS_ELLIPTIC_DISPATCH_CASE
}

} // namespace

extern "C" int jhps_dummy_two_leaf_test(
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_dispatch(2, D, nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_dummy_two_leaf_test failed: " << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_dummy_two_leaf_test failed: unknown exception\n";
    return -1;
  }
}

extern "C" int jhps_dummy_three_leaf_chain_test(
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_dispatch(3, D, nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_dummy_three_leaf_chain_test failed: " << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_dummy_three_leaf_chain_test failed: unknown exception\n";
    return -1;
  }
}

extern "C" int jhps_dummy_four_leaf_balanced_test(
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_dispatch(4, D, nverts, vertex_ids, coords_rowmajor, nelem,
      simplices_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_dummy_four_leaf_balanced_test failed: " << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_dummy_four_leaf_balanced_test failed: unknown exception\n";
    return -1;
  }
}

extern "C" int jhps_dummy_mesh_tree_test(
  int D,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  int kf,
  int vol_dim,
  unsigned int seed,
  double alpha,
  double beta,
  int verbose,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  double* monolithic_trace_residual_inf_out,
  double* leaf_volume_norm_inf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_mesh_tree_dispatch(
      D, nverts, vertex_ids, coords_rowmajor, nelem, simplices_rowmajor,
      nmerge, merge_pairs_rowmajor, kf, vol_dim, seed, alpha, beta, verbose,
      root_robin_residual_inf_out, interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out, monolithic_trace_residual_inf_out,
      leaf_volume_norm_inf_out, root_nb_out, interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_dummy_mesh_tree_test failed: " << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_dummy_mesh_tree_test failed: unknown exception\n";
    return -1;
  }
}



extern "C" int jhps_poisson_mesh_tree_solve(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_poisson_mesh_tree_dispatch(
      D, n, q_pad, q_vol, q_face, kappa,
      nverts, vertex_ids, coords_rowmajor,
      nelem, simplices_rowmajor, nmerge, merge_pairs_rowmajor,
      f_int_elementmajor,
      nboundary_faces, boundary_face_keys_rowmajor, boundary_g_rowmajor,
      tau_C, alpha, beta, verbose,
      JHPS_LEAF_OPERATOR_DENSE,
      0.0,
      0,
      leaf_coeffs_elementmajor,
      root_robin_residual_inf_out,
      interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out,
      M_out, m_int_out, kf_out, root_nb_out, interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_poisson_mesh_tree_solve failed: " << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_poisson_mesh_tree_solve failed: unknown exception\n";
    return -1;
  }
}

extern "C" int jhps_elliptic_mesh_tree_solve(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int p2,
  int p1,
  int p0,
  int assume_symmetric,
  const double* A_coeffs_elementmajor,
  const double* b_coeffs_elementmajor,
  const double* c_coeffs_elementmajor,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out,
  int* leaf_threads_used_out)
{
  try
  {
    return entry_elliptic_mesh_tree_dispatch(
      D, n, q_pad, q_vol, q_face, kappa,
      p2, p1, p0, assume_symmetric,
      A_coeffs_elementmajor,
      b_coeffs_elementmajor,
      c_coeffs_elementmajor,
      nverts, vertex_ids, coords_rowmajor,
      nelem, simplices_rowmajor, nmerge, merge_pairs_rowmajor,
      f_int_elementmajor,
      nboundary_faces, boundary_face_keys_rowmajor, boundary_g_rowmajor,
      tau_C, alpha, beta, verbose,
      JHPS_LEAF_OPERATOR_DENSE,
      0.0,
      0,
      leaf_coeffs_elementmajor,
      root_robin_residual_inf_out,
      interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out,
      M_out, m_int_out, kf_out, root_nb_out, interface_nb_out,
      leaf_threads_used_out);
  }
  catch (const std::exception& e)
  {
    std::cerr << "jhps_elliptic_mesh_tree_solve failed: "
              << e.what() << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr << "jhps_elliptic_mesh_tree_solve failed: unknown exception\n";
    return -1;
  }
}
extern "C" int jhps_poisson_mesh_tree_solve_with_leaf_mode(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out)
{
  try
  {
    return entry_poisson_mesh_tree_dispatch(
      D, n, q_pad, q_vol, q_face, kappa,
      nverts, vertex_ids, coords_rowmajor,
      nelem, simplices_rowmajor,
      nmerge, merge_pairs_rowmajor,
      f_int_elementmajor,
      nboundary_faces,
      boundary_face_keys_rowmajor,
      boundary_g_rowmajor,
      tau_C,
      alpha,
      beta,
      verbose,
      leaf_operator_mode,
      leaf_verify_tolerance,
      leaf_verify_each_solve,
      leaf_coeffs_elementmajor,
      root_robin_residual_inf_out,
      interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out,
      M_out,
      m_int_out,
      kf_out,
      root_nb_out,
      interface_nb_out);
  }
  catch (const std::exception& e)
  {
    std::cerr
      << "jhps_poisson_mesh_tree_solve_with_leaf_mode failed: "
      << e.what()
      << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr
      << "jhps_poisson_mesh_tree_solve_with_leaf_mode failed: "
      << "unknown exception\n";
    return -1;
  }
}


extern "C" int jhps_elliptic_mesh_tree_solve_with_leaf_mode(
  int D,
  int n,
  int q_pad,
  int q_vol,
  int q_face,
  const double* kappa,
  int p2,
  int p1,
  int p0,
  int assume_symmetric,
  const double* A_coeffs_elementmajor,
  const double* b_coeffs_elementmajor,
  const double* c_coeffs_elementmajor,
  int nverts,
  const int* vertex_ids,
  const double* coords_rowmajor,
  int nelem,
  const int* simplices_rowmajor,
  int nmerge,
  const int* merge_pairs_rowmajor,
  const double* f_int_elementmajor,
  int nboundary_faces,
  const int* boundary_face_keys_rowmajor,
  const double* boundary_g_rowmajor,
  double tau_C,
  double alpha,
  double beta,
  int verbose,
  int leaf_operator_mode,
  double leaf_verify_tolerance,
  int leaf_verify_each_solve,
  double* leaf_coeffs_elementmajor,
  double* root_robin_residual_inf_out,
  double* interface_flux_residual_inf_out,
  double* parent_consistency_residual_inf_out,
  int* M_out,
  int* m_int_out,
  int* kf_out,
  int* root_nb_out,
  int* interface_nb_out,
  int* leaf_threads_used_out)
{
  try
  {
    return entry_elliptic_mesh_tree_dispatch(
      D, n, q_pad, q_vol, q_face, kappa,
      p2, p1, p0, assume_symmetric,
      A_coeffs_elementmajor,
      b_coeffs_elementmajor,
      c_coeffs_elementmajor,
      nverts, vertex_ids, coords_rowmajor,
      nelem, simplices_rowmajor,
      nmerge, merge_pairs_rowmajor,
      f_int_elementmajor,
      nboundary_faces,
      boundary_face_keys_rowmajor,
      boundary_g_rowmajor,
      tau_C,
      alpha,
      beta,
      verbose,
      leaf_operator_mode,
      leaf_verify_tolerance,
      leaf_verify_each_solve,
      leaf_coeffs_elementmajor,
      root_robin_residual_inf_out,
      interface_flux_residual_inf_out,
      parent_consistency_residual_inf_out,
      M_out,
      m_int_out,
      kf_out,
      root_nb_out,
      interface_nb_out,
      leaf_threads_used_out);
  }
  catch (const std::exception& e)
  {
    std::cerr
      << "jhps_elliptic_mesh_tree_solve_with_leaf_mode failed: "
      << e.what()
      << "\n";
    return -1;
  }
  catch (...)
  {
    std::cerr
      << "jhps_elliptic_mesh_tree_solve_with_leaf_mode failed: "
      << "unknown exception\n";
    return -1;
  }
}
