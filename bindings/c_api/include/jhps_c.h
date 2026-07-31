#ifndef JHPS_C_H
#define JHPS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  Run HPS skeleton tests using dummy leaf DtN maps.

  Inputs:
    D                  spatial/simplex dimension, supported 1..5
    nverts             number of mesh vertices
    vertex_ids         length nverts
    coords_rowmajor    length nverts*D, coords_rowmajor[v*D + r]
    nelem              number of simplices
    simplices_rowmajor length nelem*(D+1), global vertex ids
    kf                 face trace block size
    vol_dim            fake leaf volume coefficient count
    seed               deterministic fake-map seed
    alpha,beta         scalar Robin coefficients for fake root solve
    verbose            nonzero prints a short report to stdout

  Outputs may be nullptr.  Residuals are infinity norms.
*/
int jhps_dummy_two_leaf_test(
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
  int* interface_nb_out
);

int jhps_dummy_three_leaf_chain_test(
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
  int* interface_nb_out
);

int jhps_dummy_four_leaf_balanced_test(
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
  int* interface_nb_out
);

/*
  Run the same dummy HPS test on an arbitrary simplex mesh and an externally
  supplied bottom-up binary merge tree.

  Mesh inputs use the same conventions as the fixed-topology tests above.

  Tree numbering:
    leaves                         0, ..., nelem - 1
    parent created by merge m      nelem + m
    merge_pairs_rowmajor[2*m + 0]  first child node id
    merge_pairs_rowmajor[2*m + 1]  second child node id

  The merge rows must be topologically ordered: both child ids must be less
  than nelem + m.  For nelem > 1, nmerge must equal nelem - 1.  Every child
  must be an active subtree root when consumed, and the final root is node
  nelem + nmerge - 1.  For nelem == 1, pass nmerge == 0; merge_pairs_rowmajor
  may then be nullptr.

  interface_nb_out is the total number of scalar trace unknowns eliminated
  over all merges, i.e. the sum of parent.merge.nI.

  Outputs may be nullptr.  Residuals are infinity norms.
*/
int jhps_dummy_mesh_tree_test(
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
  int* interface_nb_out
);


/*
  Solve the constant-coefficient Poisson problem on an arbitrary conforming
  simplex mesh using an externally supplied bottom-up HPS merge tree.

  The local polynomial and quadrature data are defined by
    n                  volume polynomial degree, n >= 2
    q_pad              quadrature padding; <=0 selects RefSimplexPrecomp default
    q_vol              volume quadrature order; <=0 selects n+q_pad
    q_face             face quadrature order; <=0 selects q_vol
    kappa              length D+1 Jacobi parameter vector

  Source data:
    f_int_elementmajor has shape nelem x m_int, row/element-major, where
      m_int = dim Pi_{n-2}^D.

  Boundary data:
    boundary_face_keys_rowmajor has shape nboundary_faces x D.  Each row is a
    global codimension-one face key; rows may be unsorted because the wrapper
    canonicalizes them.  boundary_g_rowmajor has shape nboundary_faces x kf,
    where kf=1 for D=1 and kf=dim Pi_n^{D-1} otherwise.  The wrapper reorders
    these blocks into the final root-node face ordering.

  Tree numbering is identical to jhps_dummy_mesh_tree_test:
    leaves                     0, ..., nelem-1
    parent created by merge m  nelem+m
    merge_pairs_rowmajor       nmerge x 2, with nmerge=nelem-1

  Complete source-dependent affine leaf nodes are built before the upward
  pass, so merge_nodes propagates both S and b.  Homogeneous materialization
  uses LSMR atol=btol=1e-14 and maxiter=5000.

  Robin data are scalar constants alpha,beta with
    alpha*lambda + beta*mu_hat = g
  on the root boundary.  This entry point currently requires alpha != 0; the
  pure-Neumann compatibility/gauge branch will be added separately.  The
  initial manufactured-solution path may use alpha=1, beta=0.

  leaf_coeffs_elementmajor is required and has shape nelem x M, where
    M=dim Pi_n^D.  It receives c_e=Ulam_e*lambda_e+cf_e after the downward pass.

  Residual and integer outputs may be nullptr.
*/
int jhps_poisson_mesh_tree_solve(
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
  int* interface_nb_out
);


/*
  Solve a variable-coefficient nondivergence-form elliptic problem on an
  arbitrary conforming simplex mesh using an externally supplied bottom-up
  HPS merge tree.

  The volume operator is

    L u = sum_{r,s=0}^{D-1} A_rs(x) d_{x_r x_s} u
        + sum_{r=0}^{D-1} b_r(x) d_{x_r} u
        + c(x) u,

  and the leaf equation uses the existing convention

    L u = -f.

  The reference precompute is constructed once in C++ from
    D,n,q_pad,q_vol,q_face,kappa.

  Coefficient degrees:
    p2 >= 0 is the common modal degree of every A_rs field.
    p1 >= 0 is the common modal degree of every b_r field; use -1 to disable.
    p0 >= 0 is the modal degree of c; use -1 to disable.

  All coefficient vectors are modal coefficients of the affine element
  pullbacks in the common residual Jacobi family kappa_res=kappa+2.
  Storage is element-major with modal index fastest:

    A_coeffs_elementmajor:
      shape (nelem,D,D,Mp2),
      A[(((e*D+r)*D+s)*Mp2) + alpha]

    b_coeffs_elementmajor:
      shape (nelem,D,Mp1),
      b[((e*D+r)*Mp1) + alpha]
      and may be nullptr only when p1==-1

    c_coeffs_elementmajor:
      shape (nelem,Mp0),
      c[e*Mp0 + alpha]
      and may be nullptr only when p0==-1

  Here Mpk=dim Pi_{pk}^D.  The full D x D principal tensor is supplied even
  when it is symmetric.  assume_symmetric controls the internal Clenshaw
  multiplication-plan optimization; it does not change coefficient layout or
  verify A=A^T.

  The source, boundary, tree, Robin, and output conventions are identical to
  jhps_poisson_mesh_tree_solve.  Artificial interfaces currently enforce
  continuity of trace and ordinary normal derivative through the existing
  augmented-flux merge.  Pure Neumann is not implemented in this entry point.

  Leaf construction is parallelized with OpenMP.  One immutable EllipticPlan
  is shared by the team and one EllipticWorkspace is allocated per possible
  OpenMP worker.  leaf_threads_used_out, when non-null, receives the number of
  distinct OpenMP threads that processed at least one leaf.
*/
int jhps_elliptic_mesh_tree_solve(
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
  int* leaf_threads_used_out
);

#ifdef __cplusplus
}
#endif

#endif // JHPS_C_H
