import numpy as np

from jbasis import *
from jquad_tprod import *
from jmult import *

def build_basis_structs(D, n, kappa):
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
  M = alpha_table.shape[0]
  return alpha_table, tail_deg, inv_h, M

def degree_offsets_from_alpha(alpha_table, p):
  deg = np.sum(alpha_table, axis=1).astype(np.int64)
  M = alpha_table.shape[0]
  off = np.zeros(p + 2, dtype=np.int64)
  k = 0
  for j in range(p + 1):
    off[j] = k
    while k < M and deg[k] == j:
      k += 1
  off[p + 1] = k
  if off[p + 1] != M:
    raise ValueError(f"alpha_table not truncated to p={p}? off[p+1]={off[p+1]} M={M}")
  return off

def embed_coeffs_prefix(c_small, Mbig):
  out = np.zeros(Mbig, dtype=np.float64)
  out[:c_small.size] = c_small
  return out

def project_to_coeffs(D, n, kappa, nquad, fun):
  alpha, tail, invh, M = build_basis_structs(D, n, kappa)
  X, W = jquad_mapped_build_kappa(D, nquad, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha, tail, invh, D)
  fx = fun(X)
  c = V.T @ (W * fx)
  return c, (X, W, V, alpha, tail, invh)

def multbyq_apply_clenshaw_c(D, N, p, kappa, c_fN, q_cp, K=None, assume_symmetric=True):
  """
  C-backed version of multbyq_apply_clenshaw from multby_clenshaw_opt.py:

    solver = MultByQClenshaw(D, p, K, kappa)
    c_fK = embed_coeffs_prefix(c_fN, solver.MK)
    y_K = solver.apply(q_cp, c_fK)
    return y_K[:dimPi(N)]

  Here:
    - alpha_p is computed using build_basis_structs (same ordering as your python reference)
    - MK is inferred from c_fK.size (dimPi(K))
  """
  if K is None:
    K = N + p

  kappa = np.asarray(kappa, dtype=np.float64)
  c_fN = np.asarray(c_fN, dtype=np.float64)
  q_cp = np.asarray(q_cp, dtype=np.float64)

  # Build alpha table for degree p (must match the basis ordering used in C)
  alpha_p, tail_p, invh_p, Mp = build_basis_structs(D, p, kappa)

  # Embed c_fN into Pi_K by prefix rule (same as python test)
  # embed_coeffs_prefix expects the output length (dimPi(K)).
  # We can get dimPi(K) by building basis structs for degree K (cheap for test).
  alpha_K, tail_K, invh_K, MK = build_basis_structs(D, K, kappa)

  c_fK = embed_coeffs_prefix(c_fN, MK)

  # Create C solver (plan)
  solver_c = JMultClenshaw(D, p, K, kappa, alpha_p, assume_symmetric=assume_symmetric)

  # Apply y = M_q c
  y_K = solver_c.apply(q_cp, c_fK)

  # Return only the first dimPi(N) coefficients, consistent with original helper
  return y_K[:c_fN.size].copy()


def test_eq212_lifted_backsub_c(D, N, p, kappa, nquad, f_fun, q_fun,
                               K=None, tol=5e-12, verbose=True):
  """
  Same test as multby_clenshaw_opt.py::test_eq212_lifted_backsub,
  except it uses the C backend for multbyq.
  """
  if K is None:
    K = N + p

  c_fN, (X, W, V_N, *_) = project_to_coeffs(D, N, kappa, nquad, f_fun)
  q_cp, _ = project_to_coeffs(D, p, kappa, nquad, q_fun)

  # C backend multiplication
  y_N = multbyq_apply_clenshaw_c(D, N, p, kappa, c_fN, q_cp, K=K)

  def qf_fun(Xin):
    return q_fun(Xin) * f_fun(Xin)

  c_qfN, _ = project_to_coeffs(D, N, kappa, nquad, qf_fun)

  rel_coef = np.linalg.norm(y_N - c_qfN) / max(1e-300, np.linalg.norm(c_qfN))

  qf_approx = V_N @ y_N
  qf_base   = V_N @ c_qfN
  rel_L2 = np.sqrt(
    np.sum(W * (qf_approx - qf_base)**2) /
    max(1e-300, np.sum(W * (qf_base**2)))
  )

  if verbose:
    print(f"[eq212 backsub C] D={D} N={N} p={p} K={K} nquad={nquad} "
          f"rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e}")

  ok = (rel_coef < tol) and (rel_L2 < tol)
  return ok, {"rel_coef": rel_coef, "rel_L2": rel_L2}


def run_smoke():
  # Mirror the __main__ sequence from multby_clenshaw_opt.py (but without scaling/timing)

  # D=1
  D = 1
  kappa = np.array([0.8, 1.7], dtype=np.float64)
  p = 10
  N = 5
  K = N + p
  nquad = K + p + 1
  f_fun = lambda X: X[:, 0]**5
  q_fun = lambda X: np.sin(X[:, 0])
  ok, info = test_eq212_lifted_backsub_c(D, N, p, kappa, nquad, f_fun, q_fun, K=K)
  assert ok, f"D=1 failed: {info}"

  # D=2
  D = 2
  kappa = np.array([0.8, 1.7, 2.3], dtype=np.float64)
  p = 10
  N = 5
  K = N + p
  nquad = K + p + 1
  f_fun = lambda X: X[:, 0]**5
  q_fun = lambda X: np.sin(X[:, 0] + X[:, 1])
  ok, info = test_eq212_lifted_backsub_c(D, N, p, kappa, nquad, f_fun, q_fun, K=K)
  assert ok, f"D=2 failed: {info}"

  # D=3 smoke
  D = 3
  kappa = np.array([0.8, 1.7, 2.3, 1.1], dtype=np.float64)
  p, N = 10, 8
  K = N + p
  nquad = K + p + 1
  f_fun = lambda X: X[:, 0]**4
  q_fun = lambda X: np.sin(X[:, 0] + X[:, 1] + 0.5*X[:, 2])
  ok, info = test_eq212_lifted_backsub_c(D, N, p, kappa, nquad, f_fun, q_fun, K=K)
  assert ok, f"D=3 failed: {info}"

  print("\nAll C-backend eq212 backsub smoke tests passed.")


if __name__ == "__main__":
  run_smoke()

