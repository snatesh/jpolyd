from __future__ import annotations

import argparse
import math

import numpy as np
import sympy as sp

from jelliptic import EllipticPlan
from jprecomp import RefSimplexPrecomp, dimPi


def parse_D_list(text):
  return [int(part.strip()) for part in text.split(",") if part.strip()]


def eval_sympy(expr, symbols, P):
  fun = sp.lambdify(symbols, expr, modules="numpy")
  args = [P[:, i] for i in range(P.shape[1])]
  values = np.asarray(fun(*args), dtype=np.float64)
  if values.ndim == 0:
    values = np.full(P.shape[0], float(values), dtype=np.float64)
  return np.broadcast_to(values, (P.shape[0],)).astype(np.float64, copy=False)


def affine_geometry(D):
  B = np.zeros((D, D), dtype=np.float64)
  for i in range(D):
    B[i, i] = 0.85 + 0.11 * i
    for j in range(D):
      if i != j:
        B[i, j] = 0.025 * (i + 1) / (j + 2)
  v0 = np.array([-0.17 + 0.09 * i for i in range(D)], dtype=np.float64)
  BinvT = np.asfortranarray(np.linalg.inv(B).T)
  detBabs = abs(float(np.linalg.det(B)))
  return v0, B, BinvT, detBabs


def map_to_physical(v0, B, Xhat):
  return np.ascontiguousarray(v0[None, :] + Xhat @ B.T)


def build_symbolic_problem(D, n):
  x = sp.symbols(f"x0:{D}", real=True)

  u = sp.Integer(1)
  for i in range(D):
    xi = x[i]
    u += sp.Rational(2 + i, 11) * xi
    u += sp.Rational(3 + i, 17) * xi**2
    u += sp.Rational((-1)**i * (i + 1), 23) * xi**3
    if n >= 4:
      u += sp.Rational(i + 2, 31) * xi**4
  for i in range(D):
    for j in range(i + 1, D):
      u += sp.Rational(i + j + 2, 29) * x[i] * x[j]
      if n >= 4:
        u += sp.Rational(i + 2, 37) * x[i]**2 * x[j]

  A = [[sp.Integer(0) for _ in range(D)] for _ in range(D)]
  for r in range(D):
    A[r][r] = (
      sp.Integer(1)
      + sp.Rational(r + 1, 13) * x[r]
      + sp.Rational(r + 2, 41) * x[r]**2
    )
  for r in range(D):
    for s in range(r + 1, D):
      value = (
        sp.Rational(r + 1, 47)
        + sp.Rational(s + 1, 53) * x[r]
        + sp.Rational(r + 2, 59) * x[s]
      )
      A[r][s] = value
      A[s][r] = value

  b = []
  for r in range(D):
    expr = sp.Rational((-1)**r * (r + 1), 19) + sp.Rational(r + 2, 43) * x[r]
    if D > 1:
      expr += sp.Rational(r + 1, 61) * x[r] * x[(r + 1) % D]
    else:
      expr += sp.Rational(1, 61) * x[r]**2
    b.append(expr)

  c = sp.Rational(3, 17)
  for r in range(D):
    c += sp.Rational(r + 1, 71) * x[r]

  principal = sp.Integer(0)
  for r in range(D):
    for s in range(D):
      principal += A[r][s] * sp.diff(u, x[r], x[s])

  first = sum(b[r] * sp.diff(u, x[r]) for r in range(D))
  zero = c * u
  full = sp.expand(principal + first + zero)
  return x, sp.expand(u), A, b, sp.expand(c), sp.expand(principal), sp.expand(first), sp.expand(zero), full


def project_expression(expr, symbols, P, W, V):
  values = eval_sympy(expr, symbols, P)
  return V.T @ (W * values)


def coefficient_arrays(pc, symbols, A_expr, b_expr, c_expr, v0, B, p2, p1, p0):
  Xhat, W, V = pc.residual_quad_basis()
  P = map_to_physical(v0, B, Xhat)

  Mp2 = dimPi(pc.D, p2)
  Mp1 = dimPi(pc.D, p1)
  Mp0 = dimPi(pc.D, p0)

  A = np.empty((pc.D, pc.D, Mp2), dtype=np.float64, order="C")
  for r in range(pc.D):
    for s in range(pc.D):
      A[r, s, :] = project_expression(
        A_expr[r][s], symbols, P, W, V[:, :Mp2]
      )

  b = np.empty((pc.D, Mp1), dtype=np.float64, order="C")
  for r in range(pc.D):
    b[r, :] = project_expression(
      b_expr[r], symbols, P, W, V[:, :Mp1]
    )

  c = project_expression(c_expr, symbols, P, W, V[:, :Mp0])
  return A, b, np.ascontiguousarray(c)


def projected_solution_coeffs(pc, symbols, u_expr, v0, B):
  Xhat, W = pc.volume_quad()
  V = pc.volume_basis()
  P = map_to_physical(v0, B, Xhat)
  return project_expression(u_expr, symbols, P, W, V)


def projected_residual(pc, symbols, expr, v0, B, detBabs):
  Xhat, W = pc.residual_quad()
  V = pc.residual_basis()[:, :pc.M]
  P = map_to_physical(v0, B, Xhat)
  return detBabs * project_expression(expr, symbols, P, W, V)


def relative_error(a, b):
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-300))


def run_dimension(D, n, q_vol, tol):
  kappa = np.array([0.71 + 0.17 * i for i in range(D + 1)], dtype=np.float64)
  pc = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=2,
    q_vol=q_vol,
    q_face=1 if D == 1 else q_vol,
  )

  p2, p1, p0 = 2, 2, 1
  plan = EllipticPlan(pc, p2=p2, p1=p1, p0=p0)
  workspace = plan.create_workspace()

  v0, B, BinvT, detBabs = affine_geometry(D)
  (
    symbols,
    u_expr,
    A_expr,
    b_expr,
    c_expr,
    principal_expr,
    first_expr,
    zero_expr,
    full_expr,
  ) = build_symbolic_problem(D, n)

  A, b, c = coefficient_arrays(
    pc,
    symbols,
    A_expr,
    b_expr,
    c_expr,
    v0,
    B,
    p2,
    p1,
    p0,
  )
  u_coeff = projected_solution_coeffs(pc, symbols, u_expr, v0, B)

  A0 = np.zeros_like(A)
  b0 = np.zeros_like(b)
  c0 = np.zeros_like(c)

  cases = [
    ("principal", A, b0, c0, principal_expr),
    ("first", A0, b, c0, first_expr),
    ("zero", A0, b0, c, zero_expr),
    ("full", A, b, c, full_expr),
  ]

  errors = {}
  for name, Ac, bc, cc, exact_expr in cases:
    L = plan.assemble_L_int(
      BinvT,
      detBabs,
      A=Ac,
      b=bc,
      c=cc,
      workspace=workspace,
    )
    actual = L @ u_coeff
    expected = projected_residual(
      pc,
      symbols,
      exact_expr,
      v0,
      B,
      detBabs,
    )
    err = relative_error(actual, expected)
    errors[name] = err
    assert err < tol, (D, name, err, actual, expected)

  # Constant identity principal coefficient. This exercises p2=0 and checks
  # the weighted-reference-volume normalization of the constant modal vector.
  identity_plan = EllipticPlan(pc, p2=0, p1=-1, p0=-1)
  identity_workspace = identity_plan.create_workspace()
  Xres, Wres, Vres = pc.residual_quad_basis()
  q_one = Vres[:, :1].T @ Wres
  A_identity = np.zeros((D, D, 1), dtype=np.float64, order="C")
  for r in range(D):
    A_identity[r, r, 0] = q_one[0]

  lap_expr = sum(sp.diff(u_expr, symbols[r], 2) for r in range(D))
  L_identity = identity_plan.assemble_L_int(
    BinvT,
    detBabs,
    A=A_identity,
    workspace=identity_workspace,
  )
  identity_actual = L_identity @ u_coeff
  identity_expected = projected_residual(
    pc, symbols, lap_expr, v0, B, detBabs
  )
  errors["identity"] = relative_error(identity_actual, identity_expected)
  assert errors["identity"] < tol, (
    D, "identity", errors["identity"], identity_actual, identity_expected
  )

  print(
    f"D={D} n={n} M={pc.M} m2={pc.m_int} "
    f"identity={errors['identity']:.3e} "
    f"principal={errors['principal']:.3e} "
    f"first={errors['first']:.3e} "
    f"zero={errors['zero']:.3e} "
    f"full={errors['full']:.3e}"
  )

  identity_workspace.close()
  identity_plan.close()
  workspace.close()
  plan.close()


def main():
  parser = argparse.ArgumentParser(
    description="Independent SymPy forward-operator test for jelliptic."
  )
  parser.add_argument("--D", default="1,2,3")
  parser.add_argument("--n", type=int, default=4)
  parser.add_argument("--q-vol", type=int, default=10)
  parser.add_argument("--tol", type=float, default=2.0e-10)
  args = parser.parse_args()

  if args.n < 4:
    raise ValueError("the default manufactured solution uses degree four; use n>=4")

  for D in parse_D_list(args.D):
    if D < 1 or D > 5:
      raise ValueError("D must be in 1..5")
    run_dimension(D, args.n, args.q_vol, args.tol)

  print("all SymPy elliptic forward-operator tests passed")


if __name__ == "__main__":
  main()
