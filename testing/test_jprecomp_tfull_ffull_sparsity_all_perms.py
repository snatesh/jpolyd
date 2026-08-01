#!/usr/bin/env python3
"""
Audit current C-backed RefSimplexPrecomp trace/flux sparsity for every face
permutation in arbitrary simplex dimension.

This test intentionally does NOT import or use the legacy Python-side basis,
quadrature, derivative, trace, or flux assemblers.  All operator blocks come
from the current C/C++ RefSimplexPrecomp backend through:

  T_ref      : (kf, M, nsigma, nface)
  Fgrad_ref  : (kf, M, D, nsigma, nface)
  Mface_ref  : (kf, kf, nface)
  face_ref_scale

The current backend's common-face convention is exercised automatically:

  kappa_face = kappa_volume[:D]

for every face and every local-to-canonical face permutation.

For each D, face, and permutation, the test:

1. measures numerical density of T_ref and every directional Fgrad_ref block;
2. forms a generic all-directions flux contraction
       F_eta = sum_a eta_a Fgrad_ref[a]
   to measure fill after affine-style directional mixing;
3. stacks all faces into T_full and F_full for each permutation index;
4. checks the total-degree-forbidden entries are at roundoff;
5. prunes at a configurable relative Frobenius threshold, converts to CSC,
   and checks forward and transpose applications against the dense blocks;
6. verifies Mface_ref / face_ref_scale is the identity to quadrature accuracy.

The per-permutation full matrices use the same permutation index on every face.
That is a diagnostic device for covering every stored permutation block; it is
not meant to represent every globally realizable combination of face sigmas
on one simplex.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.sparse

from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = (
  _PROJECT_ROOT / "build" / "jprecomp_tfull_ffull_sparsity_all_perms"
)
_DEFAULT_THRESHOLDS = (1.0e-14, 1.0e-12, 1.0e-10)


@dataclass(frozen=True)
class BlockAudit:
  D: int
  n: int
  operator: str
  face_id: int
  sigma_index: int
  sigma: str
  axis: int
  rows: int
  cols: int
  entries: int
  matrix_fro: float
  numerical_nnz: int
  numerical_density: float
  forbidden_degree_ratio: float
  csc_nnz: int
  csc_density: float
  prune_relative_fro: float
  forward_relative_error: float
  transpose_relative_error: float


@dataclass(frozen=True)
class FullAudit:
  D: int
  n: int
  operator: str
  sigma_index: int
  sigma: str
  rows: int
  cols: int
  entries: int
  numerical_nnz: int
  numerical_density: float
  csc_nnz: int
  csc_density: float
  prune_relative_fro: float
  forward_relative_error: float
  transpose_relative_error: float


def parse_D_list(text: str) -> tuple[int, ...]:
  if text.strip().lower() == "all":
    return (1, 2, 3, 4, 5, 6)
  values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError("--D must contain at least one dimension")
  if any(D < 1 or D > 6 for D in values):
    raise ValueError("--D values must lie in 1..6")
  if len(set(values)) != len(values):
    raise ValueError("--D contains duplicate dimensions")
  return values


def parse_thresholds(text: str) -> tuple[float, ...]:
  values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError("--thresholds must contain at least one value")
  if any(not np.isfinite(value) or value <= 0.0 for value in values):
    raise ValueError("all thresholds must be finite and positive")
  return tuple(sorted(set(values)))


def modal_total_degrees(D: int, n: int) -> np.ndarray:
  """Total degree for cumulative-degree modal ordering."""
  if D == 0:
    return np.zeros(1, dtype=np.int64)

  degree = np.empty(dimPi(D, n), dtype=np.int64)
  start = 0
  for m in range(n + 1):
    stop = dimPi(D, m)
    degree[start:stop] = m
    start = stop
  return degree


def sigma_strings(D: int) -> tuple[str, ...]:
  # C++ jperms uses lexicographic Lehmer ordering, matching itertools.
  return tuple(
    "(" + ",".join(str(value) for value in perm) + ")"
    for perm in itertools.permutations(range(D))
  )


def matrix_scale(A: np.ndarray) -> float:
  return max(float(np.linalg.norm(A, ord="fro")), 1.0e-300)


def numerical_nnz(A: np.ndarray, relative_tolerance: float) -> int:
  scale = matrix_scale(A)
  return int(np.count_nonzero(np.abs(A) > relative_tolerance * scale))


def forbidden_ratio(
  A: np.ndarray,
  face_degree: np.ndarray,
  volume_degree: np.ndarray,
  derivative_order: int,
) -> float:
  forbidden = (
    face_degree[:, None]
    >
    volume_degree[None, :] - int(derivative_order)
  )
  return float(np.linalg.norm(A[forbidden]) / matrix_scale(A))


def prune_to_csc(
  A: np.ndarray,
  relative_tolerance: float,
) -> tuple[scipy.sparse.csc_matrix, float]:
  scale = matrix_scale(A)
  kept = np.abs(A) > relative_tolerance * scale
  pruned = np.where(kept, A, 0.0)
  csc = scipy.sparse.csc_matrix(pruned)
  error = float(np.linalg.norm(A - pruned, ord="fro") / scale)
  return csc, error


def sparse_apply_errors(
  A: np.ndarray,
  A_csc: scipy.sparse.csc_matrix,
  rng: np.random.Generator,
  trials: int,
) -> tuple[float, float]:
  norm_A = matrix_scale(A)
  max_forward = 0.0
  max_transpose = 0.0

  for _ in range(trials):
    x = rng.standard_normal(A.shape[1])
    y_dense = A @ x
    y_sparse = A_csc @ x
    forward = float(
      np.linalg.norm(y_dense - y_sparse)
      / max(norm_A * np.linalg.norm(x), 1.0e-300)
    )
    max_forward = max(max_forward, forward)

    z = rng.standard_normal(A.shape[0])
    x_dense = A.T @ z
    x_sparse = A_csc.T @ z
    transpose = float(
      np.linalg.norm(x_dense - x_sparse)
      / max(norm_A * np.linalg.norm(z), 1.0e-300)
    )
    max_transpose = max(max_transpose, transpose)

  return max_forward, max_transpose


def generic_eta(D: int, face_id: int) -> np.ndarray:
  """Deterministic nonzero directional contraction for Fgrad_ref."""
  a = np.arange(1, D + 1, dtype=np.float64)
  eta = (
    0.37 * a
    + np.cos((face_id + 1.0) * (a + 0.25))
    + 0.11 * (face_id + 1.0)
  )
  # Avoid an accidental tiny component.
  tiny = np.abs(eta) < 0.15
  eta[tiny] += np.where(eta[tiny] >= 0.0, 0.31, -0.31)
  eta /= np.linalg.norm(eta)
  return eta


def estimated_backend_bytes(D: int, n: int) -> int:
  M = dimPi(D, n)
  kf = 1 if D == 1 else dimPi(D - 1, n)
  nface = D + 1
  nsigma = math.factorial(D)

  T = kf * M * nsigma * nface
  Fgrad = kf * M * D * nsigma * nface
  Mface = kf * kf * nface
  return 8 * (T + Fgrad + Mface)


def audit_block(
  *,
  D: int,
  n: int,
  operator: str,
  face_id: int,
  sigma_index: int,
  sigma: str,
  axis: int,
  A: np.ndarray,
  face_degree: np.ndarray,
  volume_degree: np.ndarray,
  derivative_order: int,
  density_tolerance: float,
  prune_tolerance: float,
  rng: np.random.Generator,
  trials: int,
) -> BlockAudit:
  A = np.asarray(A, dtype=np.float64)
  entries = int(A.size)
  nnz = numerical_nnz(A, density_tolerance)
  csc, prune_error = prune_to_csc(A, prune_tolerance)
  forward_error, transpose_error = sparse_apply_errors(
    A,
    csc,
    rng,
    trials,
  )

  return BlockAudit(
    D=D,
    n=n,
    operator=operator,
    face_id=face_id,
    sigma_index=sigma_index,
    sigma=sigma,
    axis=axis,
    rows=int(A.shape[0]),
    cols=int(A.shape[1]),
    entries=entries,
    matrix_fro=float(np.linalg.norm(A, ord="fro")),
    numerical_nnz=nnz,
    numerical_density=nnz / max(entries, 1),
    forbidden_degree_ratio=forbidden_ratio(
      A,
      face_degree,
      volume_degree,
      derivative_order,
    ),
    csc_nnz=int(csc.nnz),
    csc_density=int(csc.nnz) / max(entries, 1),
    prune_relative_fro=prune_error,
    forward_relative_error=forward_error,
    transpose_relative_error=transpose_error,
  )


def audit_full(
  *,
  D: int,
  n: int,
  operator: str,
  sigma_index: int,
  sigma: str,
  A: np.ndarray,
  density_tolerance: float,
  prune_tolerance: float,
  rng: np.random.Generator,
  trials: int,
) -> FullAudit:
  A = np.asarray(A, dtype=np.float64)
  entries = int(A.size)
  nnz = numerical_nnz(A, density_tolerance)
  csc, prune_error = prune_to_csc(A, prune_tolerance)
  forward_error, transpose_error = sparse_apply_errors(
    A,
    csc,
    rng,
    trials,
  )

  return FullAudit(
    D=D,
    n=n,
    operator=operator,
    sigma_index=sigma_index,
    sigma=sigma,
    rows=int(A.shape[0]),
    cols=int(A.shape[1]),
    entries=entries,
    numerical_nnz=nnz,
    numerical_density=nnz / max(entries, 1),
    csc_nnz=int(csc.nnz),
    csc_density=int(csc.nnz) / max(entries, 1),
    prune_relative_fro=prune_error,
    forward_relative_error=forward_error,
    transpose_relative_error=transpose_error,
  )


def summarize(
  rows: list[BlockAudit] | list[FullAudit],
  operator: str,
) -> tuple[float, float, float]:
  selected = [
    row.numerical_density
    for row in rows
    if row.operator == operator
  ]
  if not selected:
    return math.nan, math.nan, math.nan
  values = np.asarray(selected, dtype=np.float64)
  return (
    float(np.min(values)),
    float(np.mean(values)),
    float(np.max(values)),
  )


def write_dataclass_csv(path: Path, rows) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(asdict(rows[0]).keys())
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(asdict(row))


def run_dimension(
  D: int,
  n: int,
  q_pad: int,
  density_tolerance: float,
  prune_tolerance: float,
  trials: int,
  seed: int,
  structural_tolerance: float,
  sparse_error_tolerance: float,
  mass_tolerance: float,
) -> tuple[list[BlockAudit], list[FullAudit], dict[str, float | int]]:
  kappa = np.asarray(
    [0.71 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
  )
  q_vol = n + q_pad
  q_face = 1 if D == 1 else n + q_pad

  pc = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=q_pad,
    q_vol=q_vol,
    q_face=q_face,
  )

  expected_nsigma = math.factorial(D)
  if pc.nsigma != expected_nsigma:
    raise AssertionError(
      f"D={D}: backend nsigma={pc.nsigma}, expected {expected_nsigma}"
    )

  T_ref = pc.T_ref()
  Fgrad_ref = pc.Fgrad_ref()
  Mface_ref = pc.Mface_ref()
  face_ref_scale = pc.face_ref_scale()

  expected_T_shape = (pc.kf, pc.M, pc.nsigma, pc.nface)
  expected_F_shape = (pc.kf, pc.M, D, pc.nsigma, pc.nface)
  expected_Mf_shape = (pc.kf, pc.kf, pc.nface)
  if T_ref.shape != expected_T_shape:
    raise AssertionError((T_ref.shape, expected_T_shape))
  if Fgrad_ref.shape != expected_F_shape:
    raise AssertionError((Fgrad_ref.shape, expected_F_shape))
  if Mface_ref.shape != expected_Mf_shape:
    raise AssertionError((Mface_ref.shape, expected_Mf_shape))

  # This is the current C++ convention in RefSimplexPrecomp.
  common_face_kappa = kappa[:D]

  face_degree = modal_total_degrees(D - 1, n)
  volume_degree = modal_total_degrees(D, n)
  sigmas = sigma_strings(D)
  rng = np.random.default_rng(seed + 1000 * D + 17 * n)

  mass_identity_errors = []
  mass_density = []
  for face_id in range(pc.nface):
    scale = float(face_ref_scale[face_id])
    target = scale * np.eye(pc.kf)
    block = Mface_ref[:, :, face_id]
    error = float(
      np.linalg.norm(block - target, ord="fro")
      / max(np.linalg.norm(target, ord="fro"), 1.0e-300)
    )
    mass_identity_errors.append(error)
    mass_density.append(
      numerical_nnz(block, density_tolerance) / max(block.size, 1)
    )

  block_rows: list[BlockAudit] = []
  full_rows: list[FullAudit] = []

  for sigma_index, sigma in enumerate(sigmas):
    T_blocks = []
    F_blocks = []

    for face_id in range(pc.nface):
      T = T_ref[:, :, sigma_index, face_id]
      block_rows.append(
        audit_block(
          D=D,
          n=n,
          operator="T_ref",
          face_id=face_id,
          sigma_index=sigma_index,
          sigma=sigma,
          axis=-1,
          A=T,
          face_degree=face_degree,
          volume_degree=volume_degree,
          derivative_order=0,
          density_tolerance=density_tolerance,
          prune_tolerance=prune_tolerance,
          rng=rng,
          trials=trials,
        )
      )
      T_blocks.append(T)

      eta = generic_eta(D, face_id)
      F_eta = np.zeros((pc.kf, pc.M), dtype=np.float64)
      for axis in range(D):
        F_axis = Fgrad_ref[:, :, axis, sigma_index, face_id]
        block_rows.append(
          audit_block(
            D=D,
            n=n,
            operator="Fgrad_ref",
            face_id=face_id,
            sigma_index=sigma_index,
            sigma=sigma,
            axis=axis,
            A=F_axis,
            face_degree=face_degree,
            volume_degree=volume_degree,
            derivative_order=1,
            density_tolerance=density_tolerance,
            prune_tolerance=prune_tolerance,
            rng=rng,
            trials=trials,
          )
        )
        F_eta += eta[axis] * F_axis

      block_rows.append(
        audit_block(
          D=D,
          n=n,
          operator="F_eta",
          face_id=face_id,
          sigma_index=sigma_index,
          sigma=sigma,
          axis=-1,
          A=F_eta,
          face_degree=face_degree,
          volume_degree=volume_degree,
          derivative_order=1,
          density_tolerance=density_tolerance,
          prune_tolerance=prune_tolerance,
          rng=rng,
          trials=trials,
        )
      )
      F_blocks.append(F_eta)

    T_full = np.asfortranarray(np.vstack(T_blocks))
    F_full = np.asfortranarray(np.vstack(F_blocks))

    full_rows.append(
      audit_full(
        D=D,
        n=n,
        operator="T_full",
        sigma_index=sigma_index,
        sigma=sigma,
        A=T_full,
        density_tolerance=density_tolerance,
        prune_tolerance=prune_tolerance,
        rng=rng,
        trials=trials,
      )
    )
    full_rows.append(
      audit_full(
        D=D,
        n=n,
        operator="F_full_eta",
        sigma_index=sigma_index,
        sigma=sigma,
        A=F_full,
        density_tolerance=density_tolerance,
        prune_tolerance=prune_tolerance,
        rng=rng,
        trials=trials,
      )
    )

  max_forbidden = max(
    (row.forbidden_degree_ratio for row in block_rows),
    default=0.0,
  )
  max_prune = max(
    [row.prune_relative_fro for row in block_rows]
    + [row.prune_relative_fro for row in full_rows],
    default=0.0,
  )
  max_forward = max(
    [row.forward_relative_error for row in block_rows]
    + [row.forward_relative_error for row in full_rows],
    default=0.0,
  )
  max_transpose = max(
    [row.transpose_relative_error for row in block_rows]
    + [row.transpose_relative_error for row in full_rows],
    default=0.0,
  )
  max_mass_identity = max(mass_identity_errors, default=0.0)

  if max_forbidden > structural_tolerance:
    raise AssertionError(
      f"D={D}: forbidden-degree ratio {max_forbidden:.3e} exceeds "
      f"{structural_tolerance:.3e}"
    )
  if max(max_prune, max_forward, max_transpose) > sparse_error_tolerance:
    raise AssertionError(
      f"D={D}: CSC pruning/apply error exceeds {sparse_error_tolerance:.3e}: "
      f"prune={max_prune:.3e}, forward={max_forward:.3e}, "
      f"transpose={max_transpose:.3e}"
    )
  if max_mass_identity > mass_tolerance:
    raise AssertionError(
      f"D={D}: normalized Mface identity error {max_mass_identity:.3e} "
      f"exceeds {mass_tolerance:.3e}"
    )

  # For D>=2, total-degree restrictions must create actual numerical zeros.
  if D >= 2:
    for operator in ("T_ref", "Fgrad_ref", "F_eta"):
      maximum_density = max(
        row.numerical_density
        for row in block_rows
        if row.operator == operator
      )
      if maximum_density >= 1.0:
        raise AssertionError(
          f"D={D}: {operator} unexpectedly has a fully dense block"
        )

  summary = {
    "D": D,
    "n": n,
    "M": pc.M,
    "kf": pc.kf,
    "nface": pc.nface,
    "nsigma": pc.nsigma,
    "q_vol": pc.q_vol,
    "q_face": pc.q_face,
    "estimated_dense_backend_bytes": estimated_backend_bytes(D, n),
    "common_face_kappa_first": float(common_face_kappa[0]),
    "common_face_kappa_last": float(common_face_kappa[-1]),
    "Mface_identity_error_max": max_mass_identity,
    "Mface_density_min": float(np.min(mass_density)),
    "Mface_density_mean": float(np.mean(mass_density)),
    "Mface_density_max": float(np.max(mass_density)),
    "forbidden_degree_ratio_max": max_forbidden,
    "prune_relative_fro_max": max_prune,
    "forward_relative_error_max": max_forward,
    "transpose_relative_error_max": max_transpose,
  }

  for operator in ("T_ref", "Fgrad_ref", "F_eta"):
    low, mean, high = summarize(block_rows, operator)
    summary[f"{operator}_density_min"] = low
    summary[f"{operator}_density_mean"] = mean
    summary[f"{operator}_density_max"] = high

  for operator in ("T_full", "F_full_eta"):
    low, mean, high = summarize(full_rows, operator)
    summary[f"{operator}_density_min"] = low
    summary[f"{operator}_density_mean"] = mean
    summary[f"{operator}_density_max"] = high

  print("=" * 88)
  print(
    f"D={D} n={n} M={pc.M} kf={pc.kf} "
    f"faces={pc.nface} permutations={pc.nsigma}"
  )
  print(f"common face kappa (current backend): {common_face_kappa}")
  print(
    "estimated dense T/Fgrad/Mface storage: "
    f"{estimated_backend_bytes(D, n) / 2**20:.2f} MiB"
  )
  print(
    f"Mface: identity error max={max_mass_identity:.3e}, "
    f"density min/mean/max="
    f"{summary['Mface_density_min']:.4f}/"
    f"{summary['Mface_density_mean']:.4f}/"
    f"{summary['Mface_density_max']:.4f}"
  )
  for operator in ("T_ref", "Fgrad_ref", "F_eta", "T_full", "F_full_eta"):
    print(
      f"{operator:11s} density min/mean/max="
      f"{summary[f'{operator}_density_min']:.4f}/"
      f"{summary[f'{operator}_density_mean']:.4f}/"
      f"{summary[f'{operator}_density_max']:.4f}"
    )
  print(
    f"max forbidden-degree ratio={max_forbidden:.3e}; "
    f"max prune Fro error={max_prune:.3e}; "
    f"max CSC apply/transpose="
    f"{max_forward:.3e}/{max_transpose:.3e}"
  )

  return block_rows, full_rows, summary


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Audit sparsity of current C-backed jprecomp T_ref, Fgrad_ref, and "
      "Mface_ref for every face permutation and arbitrary simplex dimension."
    )
  )
  parser.add_argument(
    "--D",
    default="1,2,3,4",
    help="comma-separated dimensions in 1..6, or 'all'",
  )
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=2)
  parser.add_argument(
    "--density-rel",
    type=float,
    default=1.0e-12,
    help="relative-Frobenius threshold used to report numerical density",
  )
  parser.add_argument(
    "--prune-rel",
    type=float,
    default=1.0e-12,
    help="relative-Frobenius threshold used before CSC conversion",
  )
  parser.add_argument(
    "--thresholds",
    default=",".join(f"{value:.0e}" for value in _DEFAULT_THRESHOLDS),
    help="reserved for compatible multi-threshold extensions",
  )
  parser.add_argument("--random-trials", type=int, default=4)
  parser.add_argument("--seed", type=int, default=742000)
  parser.add_argument("--structural-tol", type=float, default=5.0e-11)
  parser.add_argument("--sparse-error-tol", type=float, default=5.0e-10)
  parser.add_argument("--mass-tol", type=float, default=5.0e-11)
  parser.add_argument(
    "--max-backend-gib",
    type=float,
    default=3.0,
    help=(
      "skip a requested D when the estimated dense T/Fgrad/Mface storage "
      "alone exceeds this limit; use --force to override"
    ),
  )
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  dimensions = parse_D_list(args.D)
  _ = parse_thresholds(args.thresholds)

  if args.n < 2:
    raise ValueError("--n must be at least 2")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.density_rel <= 0.0 or args.prune_rel <= 0.0:
    raise ValueError("density/prune thresholds must be positive")
  if args.random_trials < 1:
    raise ValueError("--random-trials must be positive")
  if args.max_backend_gib <= 0.0:
    raise ValueError("--max-backend-gib must be positive")

  set_openblas_threads(1)
  set_omp_threads(1)
  print("thread control: OpenBLAS=1, OpenMP=1")
  print(
    "operator source: current C/C++ RefSimplexPrecomp only; "
    "no legacy Python operator assembly"
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)

  all_blocks: list[BlockAudit] = []
  all_full: list[FullAudit] = []
  summaries: list[dict[str, float | int]] = []

  max_bytes = args.max_backend_gib * (2**30)
  for D in dimensions:
    estimate = estimated_backend_bytes(D, args.n)
    if estimate > max_bytes and not args.force:
      print(
        f"\nSKIP D={D}, n={args.n}: estimated dense backend T/Fgrad/Mface "
        f"storage is {estimate / 2**30:.2f} GiB, above "
        f"--max-backend-gib={args.max_backend_gib:.2f}. "
        "Use a smaller --n or --force."
      )
      continue

    blocks, full, summary = run_dimension(
      D=D,
      n=args.n,
      q_pad=args.q_pad,
      density_tolerance=args.density_rel,
      prune_tolerance=args.prune_rel,
      trials=args.random_trials,
      seed=args.seed,
      structural_tolerance=args.structural_tol,
      sparse_error_tolerance=args.sparse_error_tol,
      mass_tolerance=args.mass_tol,
    )
    all_blocks.extend(blocks)
    all_full.extend(full)
    summaries.append(summary)

  if not summaries:
    raise RuntimeError("no dimensions were run")

  block_csv = args.output_dir / "jprecomp_tf_blocks_all_perms.csv"
  full_csv = args.output_dir / "jprecomp_tfull_ffull_all_perms.csv"
  summary_csv = args.output_dir / "jprecomp_tf_sparsity_summary.csv"

  write_dataclass_csv(block_csv, all_blocks)
  write_dataclass_csv(full_csv, all_full)

  with summary_csv.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

  print("\nall requested current-backend trace/flux sparsity audits passed")
  print(f"block CSV: {block_csv}")
  print(f"full CSV: {full_csv}")
  print(f"summary CSV: {summary_csv}")


if __name__ == "__main__":
  main()
