from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jmult import JMultClenshaw
from jprecomp import dimPi
from jquad_tprod import jquad_mapped_build_kappa
from thread_control import set_omp_threads, set_openblas_threads


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "build" / "jmult_clenshaw_symmetry_adjoint.csv"


@dataclass(frozen=True)
class AuditRow:
  D: int
  n: int
  p: int
  channel: str
  N_in: int
  N_out: int
  K: int
  M_in: int
  M_out: int
  M_K: int
  q_norm: float
  full_matrix_built: int
  reference_built: int
  full_symmetry_fro: float
  full_symmetry_max: float
  restricted_cross_symmetry_fro: float
  restricted_cross_symmetry_max: float
  random_adjoint_max: float
  shortcut_apply_max: float
  shortcut_matrix_rel: float
  quadrature_reference_rel: float
  required_metric: float
  passed: int


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
  values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError(f"{name} must contain at least one integer")
  if len(set(values)) != len(values):
    raise ValueError(f"{name} contains duplicate values")
  return values


def build_basis(D: int, degree: int, kappa: np.ndarray):
  alpha, tail, inv_h = jbasis_build_structures(D, degree, kappa)
  return (
    np.ascontiguousarray(alpha, dtype=np.int32),
    tail,
    inv_h,
  )


def make_q_coefficients(
  D: int,
  p: int,
  kappa: np.ndarray,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  alpha_p, _, _ = build_basis(D, p, kappa)
  degrees = np.sum(alpha_p, axis=1).astype(np.float64)

  # Mild degree scaling keeps every total-degree layer active without allowing
  # the largest layer to dominate solely through its mode count.
  q = rng.standard_normal(alpha_p.shape[0]) / np.sqrt(1.0 + degrees)
  q /= max(float(np.linalg.norm(q)), 1.0e-300)
  return alpha_p, np.ascontiguousarray(q, dtype=np.float64)


def assemble_full_operator(
  plan: JMultClenshaw,
  q: np.ndarray,
  M_K: int,
) -> np.ndarray:
  M = np.empty((M_K, M_K), dtype=np.float64, order="F")
  e = np.zeros(M_K, dtype=np.float64)
  y = np.empty(M_K, dtype=np.float64)

  for j in range(M_K):
    e[j] = 1.0
    plan.apply(q, e, out=y)
    M[:, j] = y
    e[j] = 0.0

  return M


def apply_restricted(
  plan: JMultClenshaw,
  q: np.ndarray,
  x: np.ndarray,
  M_K: int,
  M_out: int,
  work_in: np.ndarray,
  work_out: np.ndarray,
) -> np.ndarray:
  work_in.fill(0.0)
  work_in[:x.size] = x
  plan.apply(q, work_in, out=work_out)
  return work_out[:M_out].copy()


def relative_fro(A: np.ndarray, B: np.ndarray) -> float:
  return float(
    np.linalg.norm(A - B, ord="fro")
    / max(np.linalg.norm(A, ord="fro"), np.linalg.norm(B, ord="fro"), 1.0e-300)
  )


def relative_max(A: np.ndarray, B: np.ndarray) -> float:
  return float(
    np.max(np.abs(A - B))
    / max(np.max(np.abs(A)), np.max(np.abs(B)), 1.0e-300)
  )


def random_adjoint_error(
  plan: JMultClenshaw,
  q: np.ndarray,
  M_K: int,
  M_in: int,
  M_out: int,
  rng: np.random.Generator,
  trials: int,
) -> float:
  work_in = np.zeros(M_K, dtype=np.float64)
  work_out = np.empty(M_K, dtype=np.float64)
  max_error = 0.0

  for _ in range(trials):
    x = rng.standard_normal(M_in)
    z = rng.standard_normal(M_out)

    Ax = apply_restricted(
      plan,
      q,
      x,
      M_K,
      M_out,
      work_in,
      work_out,
    )
    candidate_ATz = apply_restricted(
      plan,
      q,
      z,
      M_K,
      M_in,
      work_in,
      work_out,
    )

    lhs = float(np.dot(z, Ax))
    rhs = float(np.dot(x, candidate_ATz))
    scale = (
      np.linalg.norm(z) * np.linalg.norm(Ax)
      + np.linalg.norm(x) * np.linalg.norm(candidate_ATz)
    )
    error = abs(lhs - rhs) / max(float(scale), 1.0e-300)
    max_error = max(max_error, error)

  return max_error


def shortcut_apply_error(
  plan_symmetric: JMultClenshaw,
  plan_general: JMultClenshaw,
  q: np.ndarray,
  M_K: int,
  rng: np.random.Generator,
  trials: int,
) -> float:
  x = np.empty(M_K, dtype=np.float64)
  y_sym = np.empty(M_K, dtype=np.float64)
  y_gen = np.empty(M_K, dtype=np.float64)
  max_error = 0.0

  for _ in range(trials):
    x[:] = rng.standard_normal(M_K)
    plan_symmetric.apply(q, x, out=y_sym)
    plan_general.apply(q, x, out=y_gen)
    error = float(
      np.linalg.norm(y_sym - y_gen)
      / max(np.linalg.norm(y_gen), 1.0e-300)
    )
    max_error = max(max_error, error)

  return max_error


def quadrature_reference(
  D: int,
  p: int,
  N_in: int,
  N_out: int,
  K: int,
  kappa: np.ndarray,
  q: np.ndarray,
  q_extra: int,
) -> np.ndarray:
  # The integrand degree is at most N_out + p + N_in.  The value below is
  # deliberately conservative for the mapped tensor-product Jacobi rule.
  nquad = max(K + p + q_extra, N_in + N_out + p + q_extra)

  X, W = jquad_mapped_build_kappa(D, nquad, kappa)

  alpha_p, tail_p, inv_h_p = build_basis(D, p, kappa)
  alpha_in, tail_in, inv_h_in = build_basis(D, N_in, kappa)
  alpha_out, tail_out, inv_h_out = build_basis(D, N_out, kappa)

  Vp = jbasis_eval_all(
    X, kappa, p, alpha_p, tail_p, inv_h_p, D
  )
  Vin = jbasis_eval_all(
    X, kappa, N_in, alpha_in, tail_in, inv_h_in, D
  )
  Vout = jbasis_eval_all(
    X, kappa, N_out, alpha_out, tail_out, inv_h_out, D
  )

  q_values = Vp @ q
  return np.asfortranarray(
    Vout.T @ ((W * q_values)[:, None] * Vin)
  )


def run_case(
  D: int,
  n: int,
  p: int,
  channel: str,
  N_in: int,
  N_out: int,
  kappa: np.ndarray,
  seed: int,
  random_trials: int,
  max_full_dim: int,
  reference_max_D: int,
  reference_max_dim: int,
  q_extra: int,
  tolerance: float,
) -> AuditRow:
  K = N_in + p
  M_in = dimPi(D, N_in)
  M_out = dimPi(D, N_out)
  M_K = dimPi(D, K)

  rng = np.random.default_rng(seed)
  alpha_p, q = make_q_coefficients(D, p, kappa, rng)

  plan_sym = JMultClenshaw(
    D,
    p,
    K,
    kappa,
    alpha_p,
    assume_symmetric=True,
  )
  plan_gen = JMultClenshaw(
    D,
    p,
    K,
    kappa,
    alpha_p,
    assume_symmetric=False,
  )

  M_sym = None
  M_gen = None
  full_sym_fro = math.nan
  full_sym_max = math.nan
  cross_fro = math.nan
  cross_max = math.nan
  shortcut_matrix_rel = math.nan
  reference_rel = math.nan

  try:
    adjoint_error = random_adjoint_error(
      plan_sym,
      q,
      M_K,
      M_in,
      M_out,
      rng,
      random_trials,
    )
    shortcut_error = shortcut_apply_error(
      plan_sym,
      plan_gen,
      q,
      M_K,
      rng,
      random_trials,
    )

    full_matrix_built = int(M_K <= max_full_dim)
    if full_matrix_built:
      M_sym = assemble_full_operator(plan_sym, q, M_K)
      M_gen = assemble_full_operator(plan_gen, q, M_K)

      full_sym_fro = relative_fro(M_sym, M_sym.T)
      full_sym_max = relative_max(M_sym, M_sym.T)
      shortcut_matrix_rel = relative_fro(M_sym, M_gen)

      # Forward restricted multiplication:
      #   A = R_out M E_in
      #
      # Candidate adjoint obtained with the same Clenshaw apply:
      #   A*_candidate = E_in^T M R_out^T.
      #
      # These are adjoints precisely when the relevant leading rectangular
      # cross-block of M is symmetric.
      A_forward = M_sym[:M_out, :M_in]
      A_candidate_T = M_sym[:M_in, :M_out]
      cross_fro = relative_fro(A_forward, A_candidate_T.T)
      cross_max = relative_max(A_forward, A_candidate_T.T)

    reference_built = int(
      D <= reference_max_D
      and M_in <= reference_max_dim
      and M_out <= reference_max_dim
    )
    if reference_built:
      M_ref = quadrature_reference(
        D,
        p,
        N_in,
        N_out,
        K,
        kappa,
        q,
        q_extra,
      )

      if M_sym is not None:
        M_restricted = M_sym[:M_out, :M_in]
      else:
        M_restricted = np.empty(
          (M_out, M_in),
          dtype=np.float64,
          order="F",
        )
        e = np.zeros(M_K, dtype=np.float64)
        y = np.empty(M_K, dtype=np.float64)
        for j in range(M_in):
          e[j] = 1.0
          plan_sym.apply(q, e, out=y)
          M_restricted[:, j] = y[:M_out]
          e[j] = 0.0

      reference_rel = relative_fro(M_restricted, M_ref)

  finally:
    plan_gen.close()
    plan_sym.close()

  required_values = [adjoint_error, shortcut_error]
  if np.isfinite(cross_fro):
    required_values.append(cross_fro)
  if np.isfinite(shortcut_matrix_rel):
    required_values.append(shortcut_matrix_rel)
  if np.isfinite(reference_rel):
    required_values.append(reference_rel)

  required_metric = max(required_values)
  passed = int(required_metric <= tolerance)

  return AuditRow(
    D=D,
    n=n,
    p=p,
    channel=channel,
    N_in=N_in,
    N_out=N_out,
    K=K,
    M_in=M_in,
    M_out=M_out,
    M_K=M_K,
    q_norm=float(np.linalg.norm(q)),
    full_matrix_built=full_matrix_built,
    reference_built=reference_built,
    full_symmetry_fro=full_sym_fro,
    full_symmetry_max=full_sym_max,
    restricted_cross_symmetry_fro=cross_fro,
    restricted_cross_symmetry_max=cross_max,
    random_adjoint_max=adjoint_error,
    shortcut_apply_max=shortcut_error,
    shortcut_matrix_rel=shortcut_matrix_rel,
    quadrature_reference_rel=reference_rel,
    required_metric=required_metric,
    passed=passed,
  )


def write_csv(path: Path, rows: list[AuditRow]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for row in rows:
      writer.writerow(asdict(row))


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Audit symmetry and adjoint reuse for the C-backed multivariate "
      "Clenshaw multiplication operator.  The test distinguishes full lifted "
      "symmetry from the restricted cross-block symmetry actually needed by "
      "the elliptic LSMR forward/transpose actions."
    )
  )
  parser.add_argument("--D", default="3")
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--degrees", default="0,1,2,3,4,5,6")
  parser.add_argument("--random-trials", type=int, default=8)
  parser.add_argument(
    "--max-full-dim",
    type=int,
    default=600,
    help="assemble the full lifted M_q only when dim(Pi_K^D) is at most this",
  )
  parser.add_argument(
    "--reference-max-D",
    type=int,
    default=3,
    help="build a quadrature Galerkin reference only through this dimension",
  )
  parser.add_argument(
    "--reference-max-dim",
    type=int,
    default=500,
    help="build the quadrature reference only when both restricted dimensions fit",
  )
  parser.add_argument(
    "--q-extra",
    type=int,
    default=2,
    help="extra conservative quadrature order used by the Galerkin reference",
  )
  parser.add_argument("--tol", type=float, default=2.0e-10)
  parser.add_argument("--seed", type=int, default=731000)
  parser.add_argument("--strict", action="store_true")
  parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
  args = parser.parse_args()

  dimensions = parse_int_list(args.D, "--D")
  degrees = parse_int_list(args.degrees, "--degrees")

  if args.n < 2:
    raise ValueError("--n must be at least 2")
  if any(D < 1 or D > 5 for D in dimensions):
    raise ValueError("--D values must lie in 1..5")
  if any(p < 0 for p in degrees):
    raise ValueError("coefficient degrees must be nonnegative")
  if args.random_trials < 1:
    raise ValueError("--random-trials must be positive")
  if args.max_full_dim < 1:
    raise ValueError("--max-full-dim must be positive")
  if args.reference_max_D < 0:
    raise ValueError("--reference-max-D must be nonnegative")
  if args.reference_max_dim < 1:
    raise ValueError("--reference-max-dim must be positive")
  if args.q_extra < 1:
    raise ValueError("--q-extra must be positive")
  if not np.isfinite(args.tol) or args.tol <= 0.0:
    raise ValueError("--tol must be finite and positive")

  set_openblas_threads(1)
  set_omp_threads(10)
  print("thread control: OpenBLAS=1, OpenMP=1")

  rows: list[AuditRow] = []

  for D in dimensions:
    kappa = np.asarray(
      [0.71 + 0.17 * i for i in range(D + 1)],
      dtype=np.float64,
    )

    channels = (
      ("principal", args.n - 2, args.n - 2),
      ("first", args.n - 1, args.n - 2),
      ("zero", args.n, args.n - 2),
    )

    print(f"\nD={D}, solution degree n={args.n}")

    for p in degrees:
      for channel_index, (channel, N_in, N_out) in enumerate(channels):
        row = run_case(
          D=D,
          n=args.n,
          p=p,
          channel=channel,
          N_in=N_in,
          N_out=N_out,
          kappa=kappa,
          seed=args.seed + 10000 * D + 100 * p + channel_index,
          random_trials=args.random_trials,
          max_full_dim=args.max_full_dim,
          reference_max_D=args.reference_max_D,
          reference_max_dim=args.reference_max_dim,
          q_extra=args.q_extra,
          tolerance=args.tol,
        )
        rows.append(row)

        print(
          f"  p={p:2d} {channel:9s} "
          f"Nin/Nout/K={row.N_in}/{row.N_out}/{row.K} "
          f"Min/Mout/MK={row.M_in}/{row.M_out}/{row.M_K} "
          f"full_sym={row.full_symmetry_fro:.3e} "
          f"cross_sym={row.restricted_cross_symmetry_fro:.3e} "
          f"adj={row.random_adjoint_max:.3e} "
          f"shortcut={row.shortcut_apply_max:.3e} "
          f"ref={row.quadrature_reference_rel:.3e} "
          f"{'PASS' if row.passed else 'FAIL'}"
        )

  write_csv(args.output, rows)
  failures = [row for row in rows if not row.passed]

  print(f"\nresults CSV: {args.output}")
  print(
    "Interpretation: full_symmetry_fro is diagnostic for the entire lifted "
    "operator. restricted_cross_symmetry_fro and random_adjoint_max are the "
    "quantities needed to reuse the same Clenshaw apply in the elliptic "
    "transpose path."
  )

  if failures:
    print(f"{len(failures)} cases exceeded tol={args.tol:.3e}")
    if args.strict:
      details = ", ".join(
        f"D={row.D},p={row.p},{row.channel}:{row.required_metric:.3e}"
        for row in failures[:12]
      )
      raise AssertionError(f"Clenshaw symmetry/adjoint audit failed: {details}")
  else:
    print(f"all {len(rows)} required checks passed tol={args.tol:.3e}")


if __name__ == "__main__":
  main()
