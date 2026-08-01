#!/usr/bin/env python3
"""
Focused verification of the current C-backed one-parameter KMat stencil.

For promoted parameter r,

  K_r^(n)(kappa):
    Pi_n^D(kappa) -> Pi_n^D(kappa + e_r),

the current KMat implementation uses a degree-independent delta stencil with
two coupling families:

  Hom(j) <- Hom(j)      (same total degree)
  Hom(j) <- Hom(j + 1)  (target degree one lower than source degree)

This test verifies exactly the assumptions needed by a fixed-stencil values
entry point.

1. Kappa invariance at fixed degree
   For every promoted parameter and a broad collection of admissible source
   kappas, the raw compressed CSC structure is byte-for-byte identical.

2. Degree-independent stencil with boundary truncation
   A delta stencil is extracted once from the maximum requested degree. For
   every degree n >= D+1, the test regenerates the complete finite-n CSC
   pattern by truncating that same delta stencil at the Pi_n boundary and
   requires exact equality with the backend CSC pattern.

3. Special parameter values
   The exact same finite-n pattern test is repeated for generic, zero,
   repeated, alternating, and shifted source kappas. Explicit zero numerical
   values are retained as structural slots; no numerical pruning is used in
   these checks.

4. Cross-promoted-parameter report
   The test reports whether the stabilized delta stencils are identical for
   all promoted parameters r. This is reported rather than assumed.

Degrees below D+1 are intentionally excluded from structural assertions:
the current backend uses a dense-quadrature fallback there and then compresses
the numerical dense matrix. Roundoff-sized entries from that fallback are not
the structural stencil used by the production sparse path.

There is no PDE-order argument in this test. Promotion path commutation through
orders 2,3,4 is a separate algebraic question and is not needed to establish
KMat stencil invariance.

Example
-------
  python3 test_jkmat_promotion_stencil_v4.py \
    --D 1,2,3,4 --max-n 7 --shift-depth 4
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from jbasis import jbasis_build_structures
from libjpolyd_loader import libjpolyd

try:
  from thread_control import set_omp_threads, set_openblas_threads
except ImportError:
  set_omp_threads = None
  set_openblas_threads = None


TEST_VERSION = "v4-fixed-delta-truncation"

_INT_P = ctypes.POINTER(ctypes.c_int)
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)

try:
  _build_k_csc = libjpolyd.js_kmat_build_tprod_pruned_csc
  _free_k_csc = libjpolyd.js_kmat_csc_free
except AttributeError as exc:
  raise AttributeError(
    "The loaded libjpolyd does not export "
    "js_kmat_build_tprod_pruned_csc/js_kmat_csc_free."
  ) from exc

_build_k_csc.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_uint,
  _DOUBLE_P,
  _DOUBLE_P,
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(_INT_P),
  ctypes.POINTER(_INT_P),
  ctypes.POINTER(_DOUBLE_P),
]
_build_k_csc.restype = ctypes.c_int

_free_k_csc.argtypes = [_INT_P, _INT_P, _DOUBLE_P]
_free_k_csc.restype = None


def dim_pi(D: int, n: int) -> int:
  return math.comb(n + D, D)


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
  values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError(f"{name} must contain at least one integer")
  if len(set(values)) != len(values):
    raise ValueError(f"{name} contains duplicate values")
  return values


def weak_compositions(total: int, length: int):
  if length == 1:
    yield (total,)
    return
  for first in range(total + 1):
    for tail in weak_compositions(total - first, length - 1):
      yield (first,) + tail


def shifts_through_depth(length: int, depth: int):
  for total in range(depth + 1):
    yield from weak_compositions(total, length)


def generic_kappa(D: int) -> np.ndarray:
  return np.asarray(
    [0.215 + 0.08 * i + 0.035 * ((i + 1) % 2) for i in range(D + 1)],
    dtype=np.float64,
  )


def representative_sources(D: int) -> list[tuple[str, np.ndarray]]:
  base = generic_kappa(D)
  return [
    ("generic", base),
    ("zero", np.zeros(D + 1, dtype=np.float64)),
    ("repeated", np.full(D + 1, 0.25, dtype=np.float64)),
    (
      "alternating",
      np.asarray(
        [0.0 if i % 2 == 0 else 0.5 for i in range(D + 1)],
        dtype=np.float64,
      ),
    ),
    (
      "shifted_generic",
      base + np.arange(D + 1, dtype=np.float64),
    ),
  ]


def build_natural_k_csc(
  D: int,
  n: int,
  quadrature_order: int,
  kappa_src: np.ndarray,
  parameter: int,
) -> sp.csc_matrix:
  kappa_src = np.ascontiguousarray(kappa_src, dtype=np.float64)
  if kappa_src.shape != (D + 1,):
    raise ValueError("kappa_src has wrong shape")
  if parameter < 0 or parameter > D:
    raise ValueError("invalid promoted parameter")

  kappa_tgt = kappa_src.copy()
  kappa_tgt[parameter] += 1.0

  nrow_c = ctypes.c_int(0)
  ncol_c = ctypes.c_int(0)
  nnz_c = ctypes.c_int(0)
  colptr_p = _INT_P()
  rowind_p = _INT_P()
  values_p = _DOUBLE_P()

  rc = _build_k_csc(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(quadrature_order),
    kappa_src.ctypes.data_as(_DOUBLE_P),
    kappa_tgt.ctypes.data_as(_DOUBLE_P),
    ctypes.byref(nrow_c),
    ctypes.byref(ncol_c),
    ctypes.byref(nnz_c),
    ctypes.byref(colptr_p),
    ctypes.byref(rowind_p),
    ctypes.byref(values_p),
  )
  if rc != 0:
    raise RuntimeError(
      "js_kmat_build_tprod_pruned_csc failed with "
      f"code {rc}: D={D}, n={n}, parameter={parameter}"
    )

  nrow = int(nrow_c.value)
  ncol = int(ncol_c.value)
  nnz = int(nnz_c.value)
  expected = dim_pi(D, n)

  try:
    if (nrow, ncol) != (expected, expected):
      raise AssertionError(
        f"unexpected K shape {(nrow, ncol)}, "
        f"expected {(expected, expected)}"
      )

    indptr = np.ctypeslib.as_array(
      colptr_p,
      shape=(ncol + 1,),
    ).astype(np.int32, copy=True)

    if nnz:
      indices = np.ctypeslib.as_array(
        rowind_p,
        shape=(nnz,),
      ).astype(np.int32, copy=True)
      values = np.ctypeslib.as_array(
        values_p,
        shape=(nnz,),
      ).astype(np.float64, copy=True)
    else:
      indices = np.empty(0, dtype=np.int32)
      values = np.empty(0, dtype=np.float64)
  finally:
    _free_k_csc(colptr_p, rowind_p, values_p)

  A = sp.csc_matrix(
    (values, indices, indptr),
    shape=(nrow, ncol),
  )
  if not A.has_sorted_indices:
    A.sort_indices()
  return A


def alpha_table(D: int, n: int) -> np.ndarray:
  alpha, _, _ = jbasis_build_structures(
    D,
    n,
    generic_kappa(D),
  )
  alpha = np.asarray(alpha, dtype=np.int64)
  expected = (dim_pi(D, n), D)
  if alpha.shape != expected:
    raise AssertionError(
      f"alpha table shape {alpha.shape}, expected {expected}"
    )
  return alpha


def structural_entries(A: sp.csc_matrix) -> frozenset[tuple[int, int]]:
  """
  Return every stored CSC slot, including explicit numerical zeros.
  """
  entries: set[tuple[int, int]] = set()
  for col in range(A.shape[1]):
    for k in range(int(A.indptr[col]), int(A.indptr[col + 1])):
      entries.add((int(A.indices[k]), col))
  return frozenset(entries)


def same_raw_structure(A: sp.csc_matrix, B: sp.csc_matrix) -> bool:
  return (
    A.shape == B.shape
    and np.array_equal(A.indptr, B.indptr)
    and np.array_equal(A.indices, B.indices)
  )


def extract_delta_stencil(
  A: sp.csc_matrix,
  alpha: np.ndarray,
) -> tuple[
  frozenset[tuple[int, ...]],
  frozenset[tuple[int, ...]],
]:
  """
  Extract the two KMatStencil coupling families from raw CSC slots.

  delta0:
    target and source have equal total degree.

  deltam1:
    target degree is source degree minus one.
  """
  delta0: set[tuple[int, ...]] = set()
  deltam1: set[tuple[int, ...]] = set()
  unexpected: list[tuple[int, int, int, int]] = []

  for row, col in structural_entries(A):
    row_degree = int(np.sum(alpha[row]))
    col_degree = int(np.sum(alpha[col]))
    delta = tuple((alpha[row] - alpha[col]).tolist())

    if row_degree == col_degree:
      delta0.add(delta)
    elif row_degree == col_degree - 1:
      deltam1.add(delta)
    else:
      unexpected.append((row, col, row_degree, col_degree))

  if unexpected:
    raise AssertionError(
      "KMat contains structural couplings outside "
      "Hom(j)<-Hom(j) and Hom(j)<-Hom(j+1): "
      f"first={unexpected[:8]}"
    )

  return frozenset(delta0), frozenset(deltam1)


def expected_entries_from_deltas(
  alpha: np.ndarray,
  delta0: frozenset[tuple[int, ...]],
  deltam1: frozenset[tuple[int, ...]],
) -> frozenset[tuple[int, int]]:
  """
  Truncate one degree-independent delta stencil to the finite Pi_n space.
  """
  index_by_alpha = {
    tuple(a.tolist()): index
    for index, a in enumerate(alpha)
  }

  expected: set[tuple[int, int]] = set()

  for col, source in enumerate(alpha):
    source_degree = int(np.sum(source))

    for delta in delta0:
      target = tuple(
        int(source[d]) + int(delta[d])
        for d in range(source.size)
      )
      if any(value < 0 for value in target):
        continue
      if sum(target) != source_degree:
        continue
      row = index_by_alpha.get(target)
      if row is not None:
        expected.add((row, col))

    if source_degree >= 1:
      for delta in deltam1:
        target = tuple(
          int(source[d]) + int(delta[d])
          for d in range(source.size)
        )
        if any(value < 0 for value in target):
          continue
        if sum(target) != source_degree - 1:
          continue
        row = index_by_alpha.get(target)
        if row is not None:
          expected.add((row, col))

  return frozenset(expected)


def structure_difference(
  expected: frozenset[tuple[int, int]],
  actual: frozenset[tuple[int, int]],
) -> str:
  missing = sorted(expected - actual)
  extra = sorted(actual - expected)
  return (
    f"missing={missing[:12]}, extra={extra[:12]}, "
    f"expected_nnz={len(expected)}, actual_nnz={len(actual)}"
  )


def run_dimension(
  D: int,
  max_n: int,
  q_pad: int,
  shift_depth: int,
  verbose: bool,
) -> None:
  stencil_min = D + 1
  if max_n < stencil_min:
    raise ValueError(
      f"D={D}: --max-n={max_n} is below stencil_min=D+1={stencil_min}"
    )

  q = max_n + q_pad
  base = generic_kappa(D)
  representative = representative_sources(D)
  all_shifted_sources = [
    (
      "shift_" + "_".join(str(value) for value in shift),
      base + np.asarray(shift, dtype=np.float64),
    )
    for shift in shifts_through_depth(D + 1, shift_depth)
  ]

  print("=" * 96)
  print(
    f"D={D}, stencil degrees={stencil_min}..{max_n}, "
    f"quadrature order={q}, shift depth={shift_depth}"
  )
  print(f"base kappa={base}")
  print(
    "low-degree dense fallback excluded from structural assertions: "
    f"n < {stencil_min}"
  )

  parameter_stencils: list[
    tuple[
      frozenset[tuple[int, ...]],
      frozenset[tuple[int, ...]],
    ]
  ] = []

  for parameter in range(D + 1):
    reference_max = build_natural_k_csc(
      D,
      max_n,
      q,
      base,
      parameter,
    )
    alpha_max = alpha_table(D, max_n)
    delta0, deltam1 = extract_delta_stencil(
      reference_max,
      alpha_max,
    )
    parameter_stencils.append((delta0, deltam1))

    # Broad fixed-degree kappa invariance.
    fixed_degree_comparisons = 0
    for name, source in [
      *representative,
      *all_shifted_sources,
    ]:
      current = build_natural_k_csc(
        D,
        max_n,
        q,
        source,
        parameter,
      )
      fixed_degree_comparisons += 1
      if not same_raw_structure(reference_max, current):
        raise AssertionError(
          "raw KMat CSC structure depends on kappa at fixed degree: "
          f"D={D}, n={max_n}, parameter={parameter}, source={name}"
        )

    # Strong cross-degree test: one max-degree delta stencil must generate
    # every finite-n raw CSC pattern by boundary truncation.
    degree_comparisons = 0
    for n in range(stencil_min, max_n + 1):
      alpha_n = alpha_table(D, n)
      expected = expected_entries_from_deltas(
        alpha_n,
        delta0,
        deltam1,
      )

      for name, source in representative:
        current = build_natural_k_csc(
          D,
          n,
          q,
          source,
          parameter,
        )
        actual = structural_entries(current)
        degree_comparisons += 1

        if actual != expected:
          raise AssertionError(
            "finite-degree KMat pattern is not the truncation of one "
            "degree-independent delta stencil: "
            f"D={D}, n={n}, parameter={parameter}, source={name}; "
            + structure_difference(expected, actual)
          )

    print(
      f"parameter={parameter}: PASS; "
      f"same-degree deltas={len(delta0)}, "
      f"degree-down deltas={len(deltam1)}, "
      f"fixed-degree kappa comparisons={fixed_degree_comparisons}, "
      f"degree-truncation comparisons={degree_comparisons}"
    )
    if verbose:
      print("  delta0 :", sorted(delta0))
      print("  deltam1:", sorted(deltam1))

  all_identical = all(
    parameter_stencils[r] == parameter_stencils[0]
    for r in range(1, D + 1)
  )
  print(
    "cross-promoted-parameter delta stencils: "
    + ("IDENTICAL" if all_identical else "PARAMETER-SPECIFIC")
  )
  print(f"D={D}: PASS")


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Verify one degree-independent, kappa-independent delta stencil for "
      "current C-backed one-parameter KMat promotions."
    )
  )
  parser.add_argument(
    "--D",
    default="1,2,3,4",
    help="comma-separated dimensions in 1..6",
  )
  parser.add_argument(
    "--max-n",
    type=int,
    default=7,
    help="maximum degree used to extract and test the stencil",
  )
  parser.add_argument("--q-pad", type=int, default=2)
  parser.add_argument(
    "--shift-depth",
    type=int,
    default=4,
    help=(
      "test all nonnegative integer source-kappa shifts with total shift "
      "at most this value at the maximum degree"
    ),
  )
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  dimensions = parse_int_list(args.D, "--D")
  if any(D < 1 or D > 6 for D in dimensions):
    raise ValueError("--D values must lie in 1..6")
  if args.max_n < 1:
    raise ValueError("--max-n must be positive")
  if args.q_pad < 1:
    raise ValueError("--q-pad must be at least 1")
  if args.shift_depth < 0:
    raise ValueError("--shift-depth must be nonnegative")

  if set_openblas_threads is not None:
    set_openblas_threads(1)
  if set_omp_threads is not None:
    set_omp_threads(1)

  print(f"KMat stencil test version: {TEST_VERSION}")
  print("thread control: OpenBLAS=1, OpenMP=1 where available")
  print(
    "scope: stencil invariance only; no PDE-order/path-commutation sweep"
  )

  for D in dimensions:
    run_dimension(
      D=D,
      max_n=args.max_n,
      q_pad=args.q_pad,
      shift_depth=args.shift_depth,
      verbose=args.verbose,
    )

  print("=" * 96)
  print("all requested KMat fixed-stencil tests passed")


if __name__ == "__main__":
  main()
