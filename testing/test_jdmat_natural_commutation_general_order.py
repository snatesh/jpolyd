#!/usr/bin/env python3
"""
Exhaustive parameter-aware commutation test for natural Jacobi derivatives.

The test uses only the current C-backed Python wrappers:

  * jdmat.dmat_build_tprod_natural_pruned_csc
  * jprecomp.RefSimplexPrecomp

For requested PDE orders q (default 2,3,4), it performs three checks.

1. Every reachable adjacent commuting square

   At every derivative state alpha with |alpha| <= q-2 and every i < j,

     D_j(kappa+s(alpha)+delta_i) D_i(kappa+s(alpha))
       =
     D_i(kappa+s(alpha)+delta_j) D_j(kappa+s(alpha)),

   where

     s(alpha) = (alpha_0,...,alpha_{D-1}, |alpha|),
     delta_i  = e_i + e_D.

   Both paths have exactly the same source and destination Jacobi families.

2. Full derivative-word path independence

   For every derivative multi-index alpha with 2 <= |alpha| <= q, all
   distinct permutations of its derivative word are explicitly composed and
   compared with a canonical path.  Forward and transpose random-vector
   actions are checked as well as the sparse Frobenius matrix discrepancy.

3. Current production jprecomp second-partial symmetry

   The existing promoted reference blocks are checked directly:

     Lij_ref[:, :, i, j] == Lij_ref[:, :, j, i].

The test also verifies the structural claim needed by the proposed batched
factor skeleton: for fixed (source degree, derivative axis), every reachable
natural-shift DMat has exactly the same CSC indptr and indices arrays.

Examples
--------
  python3 test_jdmat_natural_commutation_general_order.py

  python3 test_jdmat_natural_commutation_general_order.py \
    --D 1,2,3,4 --n 6 --orders 2,3,4

  python3 test_jdmat_natural_commutation_general_order.py \
    --D 5 --n 4 --orders 2,3,4 --force
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from jdmat import dmat_build_tprod_natural_pruned_csc
from jprecomp import RefSimplexPrecomp, dimPi

try:
  from thread_control import set_omp_threads, set_openblas_threads
except ImportError:
  set_omp_threads = None
  set_openblas_threads = None


@dataclass
class ErrorMaxima:
  matrix: float = 0.0
  forward: float = 0.0
  transpose: float = 0.0

  def update(self, other: "ErrorMaxima") -> None:
    self.matrix = max(self.matrix, other.matrix)
    self.forward = max(self.forward, other.forward)
    self.transpose = max(self.transpose, other.transpose)


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
  values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError(f"{name} must contain at least one integer")
  if len(set(values)) != len(values):
    raise ValueError(f"{name} contains duplicate values")
  return values


def weak_compositions(total: int, length: int):
  """Yield all length-tuples of nonnegative integers summing to total."""
  if length == 1:
    yield (total,)
    return

  for first in range(total + 1):
    for tail in weak_compositions(total - first, length - 1):
      yield (first,) + tail


def source_shift(alpha: tuple[int, ...]) -> np.ndarray:
  """Natural Jacobi-family shift after derivative multi-index alpha."""
  return np.asarray(alpha + (sum(alpha),), dtype=np.float64)


def increment_alpha(
  alpha: tuple[int, ...],
  axis: int,
) -> tuple[int, ...]:
  out = list(alpha)
  out[axis] += 1
  return tuple(out)


def word_from_alpha(alpha: tuple[int, ...]) -> tuple[int, ...]:
  return tuple(
    axis
    for axis, count in enumerate(alpha)
    for _ in range(count)
  )


def unique_words(alpha: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
  base = word_from_alpha(alpha)
  return tuple(sorted(set(itertools.permutations(base))))


def sparse_frobenius(A: sp.spmatrix) -> float:
  data = np.asarray(A.data, dtype=np.float64)
  return float(np.sqrt(np.dot(data, data)))


def dense_frobenius(A: np.ndarray) -> float:
  return float(np.linalg.norm(np.asarray(A, dtype=np.float64), ord="fro"))


def normalized_sparse_errors(
  A: sp.csc_matrix,
  B: sp.csc_matrix,
  rng: np.random.Generator,
  random_trials: int,
) -> ErrorMaxima:
  A = A.tocsc()
  B = B.tocsc()
  difference = (A - B).tocsc()
  difference.eliminate_zeros()

  norm_A = sparse_frobenius(A)
  norm_B = sparse_frobenius(B)
  scale = max(norm_A, norm_B, 1.0e-300)

  result = ErrorMaxima(
    matrix=sparse_frobenius(difference) / scale,
  )

  for _ in range(random_trials):
    x = rng.standard_normal(A.shape[1])
    Ax = np.asarray(A @ x).reshape(-1)
    Bx = np.asarray(B @ x).reshape(-1)
    result.forward = max(
      result.forward,
      float(
        np.linalg.norm(Ax - Bx)
        / max(scale * np.linalg.norm(x), 1.0e-300)
      ),
    )

    z = rng.standard_normal(A.shape[0])
    ATz = np.asarray(A.T @ z).reshape(-1)
    BTz = np.asarray(B.T @ z).reshape(-1)
    result.transpose = max(
      result.transpose,
      float(
        np.linalg.norm(ATz - BTz)
        / max(scale * np.linalg.norm(z), 1.0e-300)
      ),
    )

  return result


def normalized_dense_errors(
  A: np.ndarray,
  B: np.ndarray,
  rng: np.random.Generator,
  random_trials: int,
) -> ErrorMaxima:
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  scale = max(dense_frobenius(A), dense_frobenius(B), 1.0e-300)

  result = ErrorMaxima(
    matrix=float(np.linalg.norm(A - B, ord="fro") / scale),
  )

  for _ in range(random_trials):
    x = rng.standard_normal(A.shape[1])
    result.forward = max(
      result.forward,
      float(
        np.linalg.norm(A @ x - B @ x)
        / max(scale * np.linalg.norm(x), 1.0e-300)
      ),
    )

    z = rng.standard_normal(A.shape[0])
    result.transpose = max(
      result.transpose,
      float(
        np.linalg.norm(A.T @ z - B.T @ z)
        / max(scale * np.linalg.norm(z), 1.0e-300)
      ),
    )

  return result


def estimated_boundary_precomp_bytes(D: int, n: int) -> int:
  """
  Conservative estimate of the dominant existing dense T/Fgrad/Mface arrays.

  This is only a skip guard.  RefSimplexPrecomp contains additional arrays.
  """
  M = dimPi(D, n)
  kf = 1 if D == 1 else dimPi(D - 1, n)
  nface = D + 1
  nsigma = math.factorial(D)

  dense_entries = (
    kf * M * nsigma * nface
    + kf * M * D * nsigma * nface
    + kf * kf * nface
  )
  return 8 * dense_entries


class NaturalDerivativeAudit:
  def __init__(
    self,
    D: int,
    n: int,
    kappa: np.ndarray,
    quadrature_order: int,
    random_trials: int,
    seed: int,
  ):
    self.D = int(D)
    self.n = int(n)
    self.kappa = np.asarray(kappa, dtype=np.float64)
    self.quadrature_order = int(quadrature_order)
    self.random_trials = int(random_trials)
    self.rng = np.random.default_rng(seed)

    # Key: (source degree, source derivative multi-index, axis).
    self.factor_cache: dict[
      tuple[int, tuple[int, ...], int],
      sp.csc_matrix,
    ] = {}

    # Key: derivative word applied from the original source family.
    self.word_cache: dict[tuple[int, ...], sp.csc_matrix] = {}

  def factor(
    self,
    degree: int,
    alpha: tuple[int, ...],
    axis: int,
  ) -> sp.csc_matrix:
    if len(alpha) != self.D:
      raise ValueError("invalid derivative-state length")
    if degree != self.n - sum(alpha):
      raise ValueError(
        f"inconsistent degree/state: degree={degree}, alpha={alpha}, "
        f"expected degree {self.n - sum(alpha)}"
      )
    if degree < 1:
      raise ValueError("cannot differentiate a degree-zero source")

    key = (int(degree), tuple(alpha), int(axis))
    cached = self.factor_cache.get(key)
    if cached is not None:
      return cached

    kappa_src = self.kappa + source_shift(alpha)
    A = dmat_build_tprod_natural_pruned_csc(
      self.D,
      degree,
      self.quadrature_order,
      kappa_src,
      axis,
    ).tocsc()

    expected_shape = (
      dimPi(self.D, degree - 1),
      dimPi(self.D, degree),
    )
    if A.shape != expected_shape:
      raise AssertionError(
        f"DMat shape {A.shape}, expected {expected_shape} for "
        f"D={self.D}, degree={degree}, alpha={alpha}, axis={axis}"
      )

    if not A.has_sorted_indices:
      A.sort_indices()

    self.factor_cache[key] = A
    return A

  def compose_word(self, word: tuple[int, ...]) -> sp.csc_matrix:
    cached = self.word_cache.get(word)
    if cached is not None:
      return cached

    if len(word) == 0:
      A = sp.identity(
        dimPi(self.D, self.n),
        dtype=np.float64,
        format="csc",
      )
      self.word_cache[word] = A
      return A

    alpha = (0,) * self.D
    A: sp.csc_matrix | None = None
    degree = self.n

    for axis in word:
      Dfactor = self.factor(degree, alpha, axis)
      A = Dfactor if A is None else (Dfactor @ A).tocsc()
      alpha = increment_alpha(alpha, axis)
      degree -= 1

    assert A is not None
    A.eliminate_zeros()
    if not A.has_sorted_indices:
      A.sort_indices()

    self.word_cache[word] = A
    return A

  def check_stencil_invariance(
    self,
    max_order: int,
  ) -> tuple[int, int]:
    """
    Compare every reachable natural-shift structure at fixed degree and axis.

    Returns:
      (number of structural families, number of factor instances checked).
    """
    family_count = 0
    instance_count = 0

    for level in range(max_order):
      degree = self.n - level
      for axis in range(self.D):
        baseline: sp.csc_matrix | None = None
        baseline_alpha: tuple[int, ...] | None = None
        family_count += 1

        for alpha in weak_compositions(level, self.D):
          A = self.factor(degree, alpha, axis)
          instance_count += 1

          if baseline is None:
            baseline = A
            baseline_alpha = alpha
            continue

          same_indptr = np.array_equal(A.indptr, baseline.indptr)
          same_indices = np.array_equal(A.indices, baseline.indices)
          if not same_indptr or not same_indices:
            raise AssertionError(
              "natural DMat stencil changed across reachable shifts: "
              f"D={self.D}, degree={degree}, axis={axis}, "
              f"baseline_alpha={baseline_alpha}, alpha={alpha}, "
              f"same_indptr={same_indptr}, same_indices={same_indices}"
            )

    return family_count, instance_count

  def check_adjacent_squares(
    self,
    order: int,
  ) -> tuple[int, ErrorMaxima]:
    maximum = ErrorMaxima()
    count = 0

    for level in range(order - 1):
      degree = self.n - level
      for alpha in weak_compositions(level, self.D):
        for i in range(self.D):
          for j in range(i + 1, self.D):
            Di = self.factor(degree, alpha, i)
            alpha_i = increment_alpha(alpha, i)
            Dj_after_i = self.factor(degree - 1, alpha_i, j)
            Aij = (Dj_after_i @ Di).tocsc()

            Dj = self.factor(degree, alpha, j)
            alpha_j = increment_alpha(alpha, j)
            Di_after_j = self.factor(degree - 1, alpha_j, i)
            Aji = (Di_after_j @ Dj).tocsc()

            errors = normalized_sparse_errors(
              Aij,
              Aji,
              self.rng,
              self.random_trials,
            )
            maximum.update(errors)
            count += 1

    return count, maximum

  def check_full_word_path_independence(
    self,
    order: int,
  ) -> tuple[int, int, ErrorMaxima]:
    """
    Compare all derivative-word permutations through the requested order.

    Returns:
      (number of derivative multi-indices, number of noncanonical paths,
       maximum errors).
    """
    maximum = ErrorMaxima()
    multi_index_count = 0
    comparison_count = 0

    for derivative_order in range(2, order + 1):
      for alpha in weak_compositions(derivative_order, self.D):
        words = unique_words(alpha)
        canonical = word_from_alpha(alpha)
        Acanonical = self.compose_word(canonical)
        multi_index_count += 1

        for word in words:
          if word == canonical:
            continue

          Aword = self.compose_word(word)
          errors = normalized_sparse_errors(
            Aword,
            Acanonical,
            self.rng,
            self.random_trials,
          )
          maximum.update(errors)
          comparison_count += 1

    return multi_index_count, comparison_count, maximum


def check_jprecomp_second_partials(
  pc: RefSimplexPrecomp,
  rng: np.random.Generator,
  random_trials: int,
) -> tuple[int, ErrorMaxima]:
  Lij = pc.Lij_ref()
  expected_shape = (pc.M, pc.M, pc.D, pc.D)
  if Lij.shape != expected_shape:
    raise AssertionError(
      f"jprecomp Lij_ref shape {Lij.shape}, expected {expected_shape}"
    )

  maximum = ErrorMaxima()
  count = 0

  for i in range(pc.D):
    for j in range(i + 1, pc.D):
      errors = normalized_dense_errors(
        Lij[:, :, i, j],
        Lij[:, :, j, i],
        rng,
        random_trials,
      )
      maximum.update(errors)
      count += 1

  return count, maximum


def assert_errors(
  label: str,
  errors: ErrorMaxima,
  tolerance: float,
) -> None:
  worst = max(errors.matrix, errors.forward, errors.transpose)
  if not np.isfinite(worst) or worst > tolerance:
    raise AssertionError(
      f"{label} failed: matrix={errors.matrix:.3e}, "
      f"forward={errors.forward:.3e}, "
      f"transpose={errors.transpose:.3e}, "
      f"tolerance={tolerance:.3e}"
    )


def run_dimension(
  D: int,
  n: int,
  orders: tuple[int, ...],
  q_pad: int,
  random_trials: int,
  seed: int,
  tolerance: float,
) -> None:
  kappa = np.asarray(
    [0.19 + 0.13 * i + 0.025 * ((-1) ** i) for i in range(D + 1)],
    dtype=np.float64,
  )
  quadrature_order = n + q_pad

  print("=" * 96)
  print(
    f"D={D}, n={n}, orders={orders}, quadrature_order={quadrature_order}"
  )
  print(f"kappa={kappa}")

  pc = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=q_pad,
    q_vol=quadrature_order,
    q_face=1 if D == 1 else quadrature_order,
  )

  rng = np.random.default_rng(seed + 200003 * D + 101 * n)
  jprecomp_count, jprecomp_errors = check_jprecomp_second_partials(
    pc,
    rng,
    random_trials,
  )
  assert_errors(
    f"D={D} jprecomp promoted Lij/Lji",
    jprecomp_errors,
    tolerance,
  )
  print(
    "jprecomp promoted second partials: "
    f"pairs={jprecomp_count}, "
    f"matrix/forward/transpose="
    f"{jprecomp_errors.matrix:.3e}/"
    f"{jprecomp_errors.forward:.3e}/"
    f"{jprecomp_errors.transpose:.3e}"
  )

  audit = NaturalDerivativeAudit(
    D=D,
    n=n,
    kappa=kappa,
    quadrature_order=quadrature_order,
    random_trials=random_trials,
    seed=seed + 700001 * D + 103 * n,
  )

  max_order = max(orders)
  family_count, factor_count = audit.check_stencil_invariance(max_order)
  print(
    "natural stencil invariance: "
    f"families={family_count}, factor_instances={factor_count}, PASS"
  )

  for order in orders:
    adjacent_count, adjacent_errors = audit.check_adjacent_squares(order)
    assert_errors(
      f"D={D}, order={order} adjacent commuting squares",
      adjacent_errors,
      tolerance,
    )

    (
      multi_index_count,
      path_comparison_count,
      path_errors,
    ) = audit.check_full_word_path_independence(order)
    assert_errors(
      f"D={D}, order={order} full path independence",
      path_errors,
      tolerance,
    )

    print(
      f"order={order}: "
      f"adjacent_squares={adjacent_count}, "
      f"adj(matrix/fwd/trans)="
      f"{adjacent_errors.matrix:.3e}/"
      f"{adjacent_errors.forward:.3e}/"
      f"{adjacent_errors.transpose:.3e}; "
      f"multi_indices={multi_index_count}, "
      f"noncanonical_paths={path_comparison_count}, "
      f"path(matrix/fwd/trans)="
      f"{path_errors.matrix:.3e}/"
      f"{path_errors.forward:.3e}/"
      f"{path_errors.transpose:.3e}"
    )

  print(
    f"D={D}: PASS; unique C-backed DMat factors built="
    f"{len(audit.factor_cache)}, composed derivative words="
    f"{len(audit.word_cache)}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Exhaustively verify parameter-aware commutation and path independence "
      "of current C-backed natural DMat operators through orders 2,3,4."
    )
  )
  parser.add_argument(
    "--D",
    default="1,2,3,4",
    help="comma-separated simplex dimensions in 1..6",
  )
  parser.add_argument(
    "--orders",
    default="2,3,4",
    help="comma-separated derivative/PDE orders",
  )
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=2)
  parser.add_argument("--random-trials", type=int, default=4)
  parser.add_argument("--seed", type=int, default=731942)
  parser.add_argument("--tol", type=float, default=5.0e-11)
  parser.add_argument(
    "--max-precomp-gib",
    type=float,
    default=2.0,
    help=(
      "skip a dimension when the dominant dense jprecomp boundary arrays "
      "alone exceed this estimate; use --force to override"
    ),
  )
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()

  dimensions = parse_int_list(args.D, "--D")
  orders = tuple(sorted(parse_int_list(args.orders, "--orders")))

  if any(D < 1 or D > 6 for D in dimensions):
    raise ValueError("--D values must lie in 1..6")
  if any(order < 2 for order in orders):
    raise ValueError("--orders values must be at least 2")
  if args.n < max(orders):
    raise ValueError("--n must be at least max(--orders)")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.random_trials < 1:
    raise ValueError("--random-trials must be positive")
  if not np.isfinite(args.tol) or args.tol <= 0.0:
    raise ValueError("--tol must be finite and positive")
  if args.max_precomp_gib <= 0.0:
    raise ValueError("--max-precomp-gib must be positive")

  if set_openblas_threads is not None:
    set_openblas_threads(1)
  if set_omp_threads is not None:
    set_omp_threads(1)
  print("thread control: OpenBLAS=1, OpenMP=1 where available")
  print(
    "backend path: current C-backed jprecomp and jdmat only; "
    "orders tested:",
    orders,
  )

  ran = 0
  byte_limit = args.max_precomp_gib * (2**30)

  for D in dimensions:
    estimate = estimated_boundary_precomp_bytes(D, args.n)
    if estimate > byte_limit and not args.force:
      print(
        f"SKIP D={D}, n={args.n}: dominant dense boundary precompute "
        f"estimate={estimate / 2**30:.2f} GiB exceeds "
        f"--max-precomp-gib={args.max_precomp_gib:.2f}; "
        "reduce --n or use --force"
      )
      continue

    run_dimension(
      D=D,
      n=args.n,
      orders=orders,
      q_pad=args.q_pad,
      random_trials=args.random_trials,
      seed=args.seed,
      tolerance=args.tol,
    )
    ran += 1

  if ran == 0:
    raise RuntimeError("no dimensions were run")

  print("=" * 96)
  print("all requested natural-derivative commutation tests passed")


if __name__ == "__main__":
  main()
