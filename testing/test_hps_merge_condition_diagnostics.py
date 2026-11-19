"""Smoke test for opt-in HPS merge and root conditioning diagnostics.

Run with the rebuilt JPolyD shared library and Python bindings:
  python3 testing/test_hps_merge_condition_diagnostics.py

This tests constructor-time records without requiring a PDE solve.
"""

import numpy as np

from jhps import HpsDivergenceSolver, HpsLeafLeastSquaresSolver
from jprecomp import RefSimplexPrecomp
from test_hps_divergence_pure_neumann import cube_simplicial_mesh


def main():
  D, n = 2, 3
  ids, X, E, merges, keys, *_ = cube_simplicial_mesh(D, 1)
  pc = RefSimplexPrecomp(D, n, np.full(D + 1, 0.5))

  _, w = pc.volume_quad()
  V = pc.volume_basis()
  const = V.T @ w
  A = np.zeros((len(E), D, D, 1))
  for direction in range(D):
    A[:, direction, direction, 0] = const[0]

  kwargs = dict(
    p_A=0, p_b=-1, p_c=-1,
    alpha=1.0, beta=0.0,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys,
    collect_merge_diagnostics=True, **kwargs,
  ) as solver:
    rows = solver.merge_condition_diagnostics()
    assert len(rows) == len(merges)
    assert np.array_equal(rows['merge_index'], np.arange(len(merges)))
    assert np.array_equal(rows['parent_id'], len(E) + np.arange(len(merges)))
    assert np.array_equal(rows['child_A_id'], merges[:, 0])
    assert np.array_equal(rows['child_B_id'], merges[:, 1])
    assert np.all(rows['level'] >= 1)
    assert np.all(rows['interface_faces'] >= 1)
    assert np.all(rows['interface_dofs'] == rows['interface_faces'] * pc.kf)
    assert np.array_equal(
      rows['parent_boundary_dofs'],
      rows['exterior_A_dofs'] + rows['exterior_B_dofs'],
    )
    assert np.all(np.isfinite(rows['H_norm_1']))
    assert np.all(rows['H_norm_1'] > 0)
    assert np.all((rows['H_rcond_1'] > 0) & (rows['H_rcond_1'] <= 1))

    root = solver.root_condition_diagnostics()
    assert root['dimension'] == solver.root_nb
    assert not root['singular']
    assert np.isclose(root['reciprocal_condition_1'], 1, atol=1e-10)

    # Rebuilding just the root Robin factorization must refresh its record.
    _, wf, Vf = pc.face_quad_basis()
    face_constant = (Vf.T @ wf)[0]
    alpha = np.full((len(keys), 1), face_constant)
    beta = np.zeros_like(alpha)
    solver.update_variable_robin(alpha, beta, p_alpha=0, p_beta=0)
    updated = solver.root_condition_diagnostics()
    assert updated['dimension'] == solver.root_nb
    assert updated['reciprocal_condition_1'] > 0
    assert np.array_equal(rows, solver.merge_condition_diagnostics())

  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys,
    collect_merge_diagnostics=False, **kwargs,
  ) as solver:
    try:
      solver.merge_condition_diagnostics()
    except RuntimeError:
      pass
    else:
      raise AssertionError('Diagnostics must be opt-in')

  print('PASS: HPS merge conditioning records and root update')


if __name__ == '__main__':
  main()
