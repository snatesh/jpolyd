import numpy as np

from jmult import JMultClenshaw
from test_jmult_clenshaw import build_basis_structs, project_to_coeffs


def relative_inf_error(x, y):
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  scale = max(1.0, float(np.linalg.norm(y, ord=np.inf)))
  return float(np.linalg.norm(x - y, ord=np.inf) / scale)


def make_test_pairs(q_base, c_base):
  """
  Build deterministic, distinct inputs with the same coefficient-space sizes.
  The first pair is the original projected multiplication from the existing
  smoke test; the remaining pairs exercise workspace reuse/state separation.
  """
  q_base = np.asarray(q_base, dtype=np.float64)
  c_base = np.asarray(c_base, dtype=np.float64)

  iq = np.arange(q_base.size, dtype=np.float64)
  ic = np.arange(c_base.size, dtype=np.float64)

  q1 = 0.75 * q_base + 1.0e-2 * np.cos(0.37 * (iq + 1.0))
  c1 = -0.40 * c_base + 2.0e-2 * np.sin(0.19 * (ic + 1.0))

  q2 = -0.25 * q_base + 1.5e-2 * np.sin(0.23 * (iq + 1.0))
  c2 = 1.20 * c_base + 1.0e-2 * np.cos(0.31 * (ic + 1.0))

  return [
    (q_base.copy(), c_base.copy()),
    (q1, c1),
    (q2, c2),
  ]


def run_case(D, N, p, kappa, nquad, f_fun, q_fun,
             rtol=5.0e-14, atol=5.0e-14):
  K = N + p

  c_fN, _ = project_to_coeffs(D, N, kappa, nquad, f_fun)
  q_cp, _ = project_to_coeffs(D, p, kappa, nquad, q_fun)

  alpha_p, _, _, Mp = build_basis_structs(D, p, kappa)
  _, _, _, MK = build_basis_structs(D, K, kappa)

  if q_cp.size != Mp:
    raise AssertionError(f"q size mismatch: got {q_cp.size}, expected {Mp}")

  c_fK = np.zeros(MK, dtype=np.float64)
  c_fK[:c_fN.size] = c_fN

  solver = JMultClenshaw(
    D,
    p,
    K,
    kappa,
    alpha_p,
    assume_symmetric=True,
  )

  work0 = solver.create_workspace()
  work1 = solver.create_workspace()

  try:
    pairs = make_test_pairs(q_cp, c_fK)

    # Legacy references use the original plan-owned compatibility workspace.
    references = [
      solver.apply(q, c)
      for q, c in pairs
    ]

    # Alternate two independent workspaces and revisit earlier inputs. This
    # catches state leakage between calls as well as API/path mismatches.
    sequence = [
      (work0, 0),
      (work1, 1),
      (work0, 2),
      (work1, 0),
      (work0, 1),
      (work1, 2),
      (work0, 0),
      (work1, 1),
    ]

    max_rel = 0.0
    max_abs = 0.0

    for workspace, pair_id in sequence:
      q, c = pairs[pair_id]
      reference = references[pair_id]

      out = np.empty_like(c)
      result = solver.apply_workspace(workspace, q, c, out=out)

      if result is not out:
        raise AssertionError("apply_workspace did not return the supplied output array")

      rel = relative_inf_error(result, reference)
      abs_err = float(np.linalg.norm(result - reference, ord=np.inf))

      max_rel = max(max_rel, rel)
      max_abs = max(max_abs, abs_err)

      if not np.allclose(result, reference, rtol=rtol, atol=atol):
        raise AssertionError(
          f"D={D} pair={pair_id}: explicit-workspace result differs from "
          f"legacy result; rel_inf={rel:.3e}, abs_inf={abs_err:.3e}"
        )

    # Directly compare the two explicit workspaces on the same fresh input.
    q_last, c_last = pairs[2]
    y0 = solver.apply_workspace(work0, q_last, c_last)
    y1 = solver.apply_workspace(work1, q_last, c_last)

    rel_between = relative_inf_error(y0, y1)
    abs_between = float(np.linalg.norm(y0 - y1, ord=np.inf))

    max_rel = max(max_rel, rel_between)
    max_abs = max(max_abs, abs_between)

    if not np.allclose(y0, y1, rtol=rtol, atol=atol):
      raise AssertionError(
        f"D={D}: work0/work1 disagreement; "
        f"rel_inf={rel_between:.3e}, abs_inf={abs_between:.3e}"
      )

    print(
      f"[jmult multiworkspace] D={D} N={N} p={p} K={K} "
      f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )

  finally:
    solver.destroy_workspace(work1)
    solver.destroy_workspace(work0)
    solver.close()


def run_multiworkspace_smoke():
  run_case(
    D=1,
    N=5,
    p=10,
    kappa=np.array([0.8, 1.7], dtype=np.float64),
    nquad=26,
    f_fun=lambda X: X[:, 0]**5,
    q_fun=lambda X: np.sin(X[:, 0]),
  )

  run_case(
    D=2,
    N=5,
    p=10,
    kappa=np.array([0.8, 1.7, 2.3], dtype=np.float64),
    nquad=26,
    f_fun=lambda X: X[:, 0]**5,
    q_fun=lambda X: np.sin(X[:, 0] + X[:, 1]),
  )

  run_case(
    D=3,
    N=8,
    p=10,
    kappa=np.array([0.8, 1.7, 2.3, 1.1], dtype=np.float64),
    nquad=29,
    f_fun=lambda X: X[:, 0]**4,
    q_fun=lambda X: np.sin(X[:, 0] + X[:, 1] + 0.5 * X[:, 2]),
  )

  print("\nAll explicit-workspace compatibility smoke tests passed.")


if __name__ == "__main__":
  run_multiworkspace_smoke()
