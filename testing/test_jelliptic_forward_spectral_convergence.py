from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from jelliptic import EllipticPlan
from jprecomp import RefSimplexPrecomp, dimPi


ScalarFun = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SymbolicProblem:
  symbols: tuple[sp.Symbol, ...]
  u_expr: sp.Expr
  A_expr: tuple[tuple[sp.Expr, ...], ...]
  b_expr: tuple[sp.Expr, ...]
  c_expr: sp.Expr
  Lu_expr: sp.Expr
  u_fun: ScalarFun
  A_fun: tuple[tuple[ScalarFun, ...], ...]
  b_fun: tuple[ScalarFun, ...]
  c_fun: ScalarFun
  Lu_fun: ScalarFun


@dataclass(frozen=True)
class ConvergenceRow:
  D: int
  n: int
  M: int
  m2: int
  p: int
  q_proj: int
  q_eval: int
  rel_error: float
  abs_error: float
  exact_norm: float


def parse_D_list(text: str) -> list[int]:
  out = [int(part.strip()) for part in text.split(",") if part.strip()]
  if not out or any(D < 1 or D > 5 for D in out):
    raise ValueError("--D must list dimensions in 1..5")
  return out


def eval_lambdified(fun: Callable, P: np.ndarray) -> np.ndarray:
  """Evaluate a scalar lambdified SymPy function at row-major points P."""
  P = np.asarray(P, dtype=np.float64)
  args = [P[:, i] for i in range(P.shape[1])]
  values = np.asarray(fun(*args), dtype=np.float64)
  if values.ndim == 0:
    values = np.full(P.shape[0], float(values), dtype=np.float64)
  return np.broadcast_to(values, (P.shape[0],)).astype(np.float64, copy=False)


def make_scalar_fun(expr: sp.Expr, symbols: tuple[sp.Symbol, ...]) -> ScalarFun:
  raw = sp.lambdify(symbols, expr, modules="numpy")

  def wrapped(P: np.ndarray) -> np.ndarray:
    return eval_lambdified(raw, P)

  return wrapped


def build_symbolic_problem(D: int) -> SymbolicProblem:
  """Build a smooth, nonpolynomial, symmetric elliptic forward problem."""
  symbols = tuple(sp.symbols(f"x0:{D}", real=True))
  s = sum(symbols)
  weighted_s = sum((r + 1) * symbols[r] for r in range(D))

  # Smooth nonpolynomial trial function.
  u_expr = sp.exp(sp.sin(s)) + sp.Rational(1, 10) * sp.cos(weighted_s)

  # Symmetric, diagonally dominant principal coefficient field.
  A = [[sp.Integer(0) for _ in range(D)] for _ in range(D)]
  for r in range(D):
    A[r][r] = (
      sp.Rational(3, 2)
      + sp.Rational(1, 10) * sp.sin(s + symbols[r])
    )
  for r in range(D):
    for t in range(r + 1, D):
      value = sp.Rational(1, 50) * sp.cos(symbols[r] - symbols[t])
      A[r][t] = value
      A[t][r] = value

  b = tuple(
    sp.Rational(1, 10) * sp.sin(s + (r + 1) * symbols[r])
    for r in range(D)
  )
  c_expr = sp.Rational(2, 5) + sp.Rational(1, 10) * sp.cos(s)

  principal_expr = sum(
    A[r][t] * sp.diff(u_expr, symbols[r], symbols[t])
    for r in range(D)
    for t in range(D)
  )
  first_expr = sum(
    b[r] * sp.diff(u_expr, symbols[r])
    for r in range(D)
  )
  Lu_expr = principal_expr + first_expr + c_expr * u_expr

  A_expr = tuple(tuple(A[r][t] for t in range(D)) for r in range(D))
  A_fun = tuple(
    tuple(make_scalar_fun(A_expr[r][t], symbols) for t in range(D))
    for r in range(D)
  )
  b_fun = tuple(make_scalar_fun(expr, symbols) for expr in b)

  return SymbolicProblem(
    symbols=symbols,
    u_expr=u_expr,
    A_expr=A_expr,
    b_expr=b,
    c_expr=c_expr,
    Lu_expr=Lu_expr,
    u_fun=make_scalar_fun(u_expr, symbols),
    A_fun=A_fun,
    b_fun=b_fun,
    c_fun=make_scalar_fun(c_expr, symbols),
    Lu_fun=make_scalar_fun(Lu_expr, symbols),
  )


def affine_geometry(D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Return v0, B, B^{-T}, and |det B| for one fixed affine element."""
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


def map_to_physical(v0: np.ndarray, B: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(v0[None, :] + Xhat @ B.T)


def project_values(V: np.ndarray, W: np.ndarray, values: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(V.T @ (W * values))


def choose_quadrature_orders(
  n: int,
  q_factor: float,
  q_eval_extra: int,
) -> tuple[int, int]:
  # Keep orders modest: the mapped tensor-product quadrature and basis
  # evaluation eventually become roundoff-limited at large n and D.
  q_proj = max(n + 1, int(math.floor(q_factor * n + 0.5)))
  q_eval = max(q_proj, q_proj + q_eval_extra)
  return q_proj, q_eval


def project_solution(
  pc: RefSimplexPrecomp,
  problem: SymbolicProblem,
  v0: np.ndarray,
  B: np.ndarray,
) -> np.ndarray:
  Xhat, W = pc.volume_quad()
  V = pc.volume_basis()
  P = map_to_physical(v0, B, Xhat)
  return project_values(V, W, problem.u_fun(P))


def project_coefficients(
  pc: RefSimplexPrecomp,
  problem: SymbolicProblem,
  v0: np.ndarray,
  B: np.ndarray,
  p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  Xhat, W, V = pc.residual_quad_basis()
  P = map_to_physical(v0, B, Xhat)
  Mp = dimPi(pc.D, p)
  Vp = V[:, :Mp]

  A = np.empty((pc.D, pc.D, Mp), dtype=np.float64, order="C")
  for r in range(pc.D):
    for t in range(pc.D):
      A[r, t, :] = project_values(Vp, W, problem.A_fun[r][t](P))

  b = np.empty((pc.D, Mp), dtype=np.float64, order="C")
  for r in range(pc.D):
    b[r, :] = project_values(Vp, W, problem.b_fun[r](P))

  c = project_values(Vp, W, problem.c_fun(P))
  return A, b, c


def weighted_residual_error(
  pc_eval: RefSimplexPrecomp,
  problem: SymbolicProblem,
  v0: np.ndarray,
  B: np.ndarray,
  Lu_coeff: np.ndarray,
) -> tuple[float, float, float]:
  Xhat, W = pc_eval.residual_quad()
  V = pc_eval.residual_basis()[:, :pc_eval.m_int]
  P = map_to_physical(v0, B, Xhat)

  Lu_discrete = V @ Lu_coeff
  Lu_exact = problem.Lu_fun(P)
  diff = Lu_discrete - Lu_exact

  abs_error = math.sqrt(max(float(np.sum(W * diff * diff)), 0.0))
  exact_norm = math.sqrt(max(float(np.sum(W * Lu_exact * Lu_exact)), 0.0))
  rel_error = abs_error / max(exact_norm, 1.0e-300)
  return rel_error, abs_error, exact_norm


def run_degree(
  D: int,
  n: int,
  kappa: np.ndarray,
  problem: SymbolicProblem,
  q_factor: float,
  q_eval_extra: int,
) -> ConvergenceRow:
  q_proj, q_eval = choose_quadrature_orders(n, q_factor, q_eval_extra)
  v0, B, BinvT, detBabs = affine_geometry(D)

  pc = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=2,
    q_vol=q_proj,
    q_face=1 if D == 1 else q_proj,
  )
  pc_eval = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=2,
    q_vol=q_eval,
    q_face=1 if D == 1 else q_eval,
  )

  # Use the target solution degree for all three coefficient fields.
  p = n
  plan = EllipticPlan(pc, p2=p, p1=p, p0=p, assume_symmetric=True)
  workspace = plan.create_workspace()

  try:
    u_coeff = project_solution(pc, problem, v0, B)
    A, b, c = project_coefficients(pc, problem, v0, B, p)

    L_int = plan.assemble_L_int(
      BinvT,
      detBabs,
      A=A,
      b=b,
      c=c,
      workspace=workspace,
    )

    # L_int includes the affine volume factor used by the leaf residual.
    # Divide it out before reconstructing the pointwise forward operator.
    Lu_coeff = (L_int @ u_coeff) / detBabs
    rel_error, abs_error, exact_norm = weighted_residual_error(
      pc_eval,
      problem,
      v0,
      B,
      Lu_coeff,
    )

    row = ConvergenceRow(
      D=D,
      n=n,
      M=pc.M,
      m2=pc.m_int,
      p=p,
      q_proj=q_proj,
      q_eval=q_eval,
      rel_error=rel_error,
      abs_error=abs_error,
      exact_norm=exact_norm,
    )
  finally:
    workspace.close()
    plan.close()

  if not np.isfinite(row.rel_error):
    raise AssertionError(f"nonfinite forward error for D={D}, n={n}: {row}")

  print(
    f"D={D} n={n:2d} M={row.M:5d} m2={row.m2:5d} "
    f"p={p:2d} q_proj={q_proj:2d} q_eval={q_eval:2d} "
    f"rel_res={row.rel_error:.6e} abs_res={row.abs_error:.6e}"
  )
  return row


def write_csv(path: Path, rows: list[ConvergenceRow]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(ConvergenceRow.__dataclass_fields__))
    writer.writeheader()
    for row in rows:
      writer.writerow(row.__dict__)


def plot_convergence(path: Path, rows: list[ConvergenceRow], show: bool) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  plt.figure(figsize=(8.0, 5.5))

  for D in sorted({row.D for row in rows}):
    selected = sorted((row for row in rows if row.D == D), key=lambda row: row.M)
    modes = np.array([row.M for row in selected], dtype=np.int64)
    errors = np.array([row.rel_error for row in selected], dtype=np.float64)
    plt.semilogy(modes, errors, marker="o", label=f"D={D}")

  plt.xlabel(r"$M=\dim\Pi_n^D$")
  plt.ylabel(r"relative $L^2_{w_{\kappa_{\mathrm{res}}}}$ forward error")
  plt.title("Variable-coefficient elliptic forward-operator convergence")
  plt.grid(True, which="both", alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=180)
  print(f"saved convergence plot: {path}")

  if show:
    plt.show()
  else:
    plt.close()


def check_reduction(rows: list[ConvergenceRow], min_reduction: float) -> None:
  if min_reduction <= 0.0:
    return

  for D in sorted({row.D for row in rows}):
    selected = sorted((row for row in rows if row.D == D), key=lambda row: row.n)
    if len(selected) < 2:
      continue
    initial = max(row.rel_error for row in selected[: min(2, len(selected))])
    final = min(row.rel_error for row in selected[-min(2, len(selected)) :])
    reduction = initial / max(final, 1.0e-300)
    assert reduction >= min_reduction, (
      f"D={D}: forward error reduction {reduction:.3e} "
      f"is below requested {min_reduction:.3e}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Spectral convergence of the variable-coefficient elliptic forward "
      "operator against a symbolic nonpolynomial reference."
    )
  )
  parser.add_argument("--D", default="1,2,3")
  parser.add_argument("--n-min", type=int, default=2)
  parser.add_argument("--n-max", type=int, default=10)
  parser.add_argument("--n-step", type=int, default=1)
  parser.add_argument(
    "--q-factor",
    type=float,
    default=1.5,
    help="projection quadrature order is max(n+1, round(q_factor*n))",
  )
  parser.add_argument(
    "--q-eval-extra",
    type=int,
    default=1,
    help="evaluation quadrature order increment above q_proj",
  )
  parser.add_argument(
    "--min-reduction",
    type=float,
    default=0.0,
    help="optional required initial-to-final error reduction; 0 disables",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("build/jelliptic_forward_spectral_convergence.png"),
  )
  parser.add_argument(
    "--csv",
    type=Path,
    default=Path("build/jelliptic_forward_spectral_convergence.csv"),
  )
  parser.add_argument("--show", action="store_true")
  args = parser.parse_args()

  if args.n_min < 2:
    raise ValueError("use --n-min >= 2 for a second-order operator")
  if args.n_max < args.n_min:
    raise ValueError("--n-max must be at least --n-min")
  if args.n_step < 1:
    raise ValueError("--n-step must be positive")
  if args.q_factor < 1.0:
    raise ValueError("--q-factor should be at least 1")
  if args.q_eval_extra < 0:
    raise ValueError("--q-eval-extra must be nonnegative")

  rows: list[ConvergenceRow] = []
  for D in parse_D_list(args.D):
    print(f"\nBuilding symbolic problem for D={D}")
    problem = build_symbolic_problem(D)
    kappa = np.array(
      [0.71 + 0.17 * i for i in range(D + 1)],
      dtype=np.float64,
    )

    for n in range(args.n_min, args.n_max + 1, args.n_step):
      rows.append(
        run_degree(
          D,
          n,
          kappa,
          problem,
          args.q_factor,
          args.q_eval_extra,
        )
      )

  check_reduction(rows, args.min_reduction)
  write_csv(args.csv, rows)
  print(f"saved convergence data: {args.csv}")
  plot_convergence(args.output, rows, args.show)


if __name__ == "__main__":
  main()
