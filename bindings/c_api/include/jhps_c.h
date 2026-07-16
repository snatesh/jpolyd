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

#ifdef __cplusplus
}
#endif

#endif // JHPS_C_H
