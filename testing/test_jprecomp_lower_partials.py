from __future__ import annotations

import argparse
import math

import numpy as np

from jprecomp import RefSimplexPrecomp, dimPi


def multiindices(D: int, n: int):
  alpha = [0] * D

  def rec(pos: int, remaining: int):
    if pos == D - 1:
      alpha[pos] = remaining
      yield tuple(alpha)
      return
    for value in range(remaining + 1):
      alpha[pos] = value
      yield from rec(pos + 1, remaining - value)

  for total in range(n + 1):
    yield from rec(0, total)


def make_dense_polynomial(D: int, n: int, seed: int):
  rng = np.random.default_rng(seed)
  terms = []
  for alpha in multiindices(D, n):
    # Keep every monomial active while avoiding large coefficient growth.
    coeff = float(rng.uniform(-0.5, 0.5)) / (1.0 + sum(alpha))
    terms.append((alpha, coeff))

  def value(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros(X.shape[0], dtype=np.float64)
    for alpha, coeff in terms:
      term = np.full(X.shape[0], coeff, dtype=np.float64)
      for axis, power in enumerate(alpha):
        if power:
          term *= X[:, axis] ** power
      out += term
    return out

  def partial(X: np.ndarray, axis: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros(X.shape[0], dtype=np.float64)
    for alpha, coeff in terms:
      power_axis = alpha[axis]
      if power_axis == 0:
        continue
      term = np.full(
        X.shape[0], coeff * power_axis, dtype=np.float64
      )
      for j, power in enumerate(alpha):
        derivative_power = power - 1 if j == axis else power
        if derivative_power:
          term *= X[:, j] ** derivative_power
      out += term
    return out

  return value, partial


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
  return float(
    np.linalg.norm(actual - expected)
    / max(np.linalg.norm(expected), 1.0e-300)
  )


def run_case(D: int, n: int, q_pad: int, tol: float) -> None:
  kappa = np.array(
    [0.41 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
  )
  q_vol = n + q_pad
  q_face = 1 if D == 1 else q_vol
  pc = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=q_pad,
    q_vol=q_vol,
    q_face=q_face,
  )

  np.testing.assert_allclose(
    pc.kappa_res,
    kappa + 2.0,
    rtol=0.0,
    atol=0.0,
  )

  value, partial = make_dense_polynomial(D, n, seed=7300 + D)

  X_src, W_src = pc.volume_quad()
  V_src = pc.volume_basis()
  c_src = V_src.T @ (W_src * value(X_src))

  X_res, W_res = pc.residual_quad()
  V_res = pc.residual_basis()

  Li = pc.Li_ref()
  L0 = pc.L0_ref()

  c0_actual = L0 @ c_src
  c0_expected = V_res.T @ (W_res * value(X_res))
  err0 = relative_error(c0_actual, c0_expected)

  m_first = dimPi(D, n - 1)
  max_first = 0.0
  max_tail = 0.0
  for axis in range(D):
    ci_actual = Li[:, :, axis] @ c_src
    ci_expected = V_res.T @ (W_res * partial(X_res, axis))
    err = relative_error(ci_actual, ci_expected)
    tail = float(np.linalg.norm(ci_actual[m_first:]))
    max_first = max(max_first, err)
    max_tail = max(max_tail, tail)

  assert err0 < tol, (D, "L0", err0)
  assert max_first < tol, (D, "Li", max_first)
  assert max_tail < 10.0 * tol, (D, "Li tail", max_tail)

  print(
    f"D={D} n={n} M={pc.M} m_first={m_first} "
    f"L0_rel={err0:.3e} Li_rel={max_first:.3e} "
    f"Li_tail={max_tail:.3e}"
  )


def parse_D_list(text: str) -> list[int]:
  if text.strip().lower() == "all":
    return [1, 2, 3, 4, 5]
  values = [int(part.strip()) for part in text.split(",") if part.strip()]
  if not values or any(D < 1 or D > 5 for D in values):
    raise ValueError("--D must list dimensions in 1..5 or be 'all'")
  return values


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Validate promoted first- and zero-order reference operators "
      "against direct projection in the kappa+2 residual basis."
    )
  )
  parser.add_argument("--D", default="all")
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=3)
  parser.add_argument("--tol", type=float, default=5.0e-11)
  args = parser.parse_args()

  if args.n < 2:
    raise ValueError("n must be at least 2")
  if args.q_pad < 1:
    raise ValueError("q-pad must be positive")

  for D in parse_D_list(args.D):
    run_case(D, args.n, args.q_pad, args.tol)

  print("all promoted lower-order precompute tests passed")


if __name__ == "__main__":
  main()
