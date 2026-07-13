import numpy as np
from jlsmr import lsmr_dense_solve


def build_dense_case(m=80, n=30):
  # Build directly as Fortran order because the C wrapper expects column-major.
  A = np.empty((m, n), dtype=np.float64, order="F")
  for j in range(n):
    jj = j + 1
    for i in range(m):
      ii = i + 1
      A[i, j] = np.sin(0.13 * ii * jj) + 0.25 * np.cos(0.07 * (ii + 2 * jj))
      if ii == jj:
        A[i, j] += 2.0

  xtrue = np.empty(n, dtype=np.float64)
  for j in range(n):
    jj = j + 1
    xtrue[j] = ((-1.0) ** jj) * jj / n

  b = A @ xtrue
  return A, b, xtrue


def test_dense_lsmr():
  A, b, xtrue = build_dense_case()
  x, info = lsmr_dense_solve(A, b, nout=0)

  relerr = np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue)
  print("LSMR info:", info)
  print(f"relative solution error = {relerr:.3e}")

  assert info["ret"] == 0
  assert info["stat"] == 0
  assert info["istop"] > 0
  assert relerr < 1e-10


if __name__ == "__main__":
  test_dense_lsmr()
  print("jlsmr dense smoke test passed.")
