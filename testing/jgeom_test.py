import numpy as np

from jgeom import invert_colmajor, affine_from_verts


def print_matrix(name, A):
  print(f"{name} shape={A.shape}")
  print(A)
  print()


def run_inverse_demo_d4():
  print("=== D=4 column-major inverse test ===")

  A = np.array(
    [
      [2.0, 0.5, -0.25, 1.0],
      [0.1, 3.0, 0.75, -0.5],
      [1.2, -0.4, 2.5, 0.25],
      [-0.3, 0.8, 0.6, 1.75],
    ],
    dtype=np.float64,
    order="F",
  )

  Ainv, detA = invert_colmajor(A)

  print_matrix("A", A)
  print_matrix("Ainv from C++", Ainv)
  print(f"det(A) from C++  = {detA:.16e}")
  print(f"det(A) from numpy = {np.linalg.det(A):.16e}")
  print()

  print_matrix("A @ Ainv", A @ Ainv)
  print_matrix("Ainv @ A", Ainv @ A)

  np.testing.assert_allclose(A @ Ainv, np.eye(4), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(Ainv @ A, np.eye(4), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(detA, np.linalg.det(A), rtol=1e-12, atol=1e-12)


def run_affine_demo_d4():
  print("=== D=4 affine simplex geometry test ===")

  # V is D x (D+1), column-major.
  # Column j is vertex v_j.
  V = np.array(
    [
      [1.0, 3.0, 1.5, 0.5, 2.0],
      [-2.0, -1.0, 1.0, -0.5, 0.75],
      [0.25, 0.0, 0.75, 2.0, -0.5],
      [1.5, 2.5, 1.25, 0.0, 3.0],
    ],
    dtype=np.float64,
    order="F",
  )

  geom = affine_from_verts(V)

  B_expected = np.asfortranarray(V[:, 1:] - V[:, [0]])
  BinvT_expected = np.linalg.inv(B_expected).T

  print_matrix("Vertices V, columns are simplex vertices", V)
  print_matrix("B from C++", geom["B"])
  print_matrix("B expected = [v1-v0, ..., v4-v0]", B_expected)

  print_matrix("BinvT from C++", geom["BinvT"])
  print_matrix("BinvT expected = inv(B).T", BinvT_expected)

  print(f"detB from C++    = {geom['detB']:.16e}")
  print(f"detB from numpy  = {np.linalg.det(B_expected):.16e}")
  print(f"abs(detB) C++    = {geom['detBabs']:.16e}")
  print()

  np.testing.assert_allclose(geom["B"], B_expected, rtol=0.0, atol=0.0)
  np.testing.assert_allclose(geom["BinvT"], BinvT_expected, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(geom["detB"], np.linalg.det(B_expected), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(geom["detBabs"], abs(np.linalg.det(B_expected)), rtol=1e-12, atol=1e-12)


def run_degenerate_demo_d4():
  print("=== D=4 degenerate simplex rejection test ===")

  V_bad = np.zeros((4, 5), dtype=np.float64, order="F")

  try:
    affine_from_verts(V_bad)
  except np.linalg.LinAlgError as exc:
    print("Correctly rejected degenerate 4-simplex:")
    print(f"  {exc}")
    print()
    return

  raise AssertionError("Expected np.linalg.LinAlgError for degenerate 4-simplex")


if __name__ == "__main__":
  run_inverse_demo_d4()
  run_affine_demo_d4()
  run_degenerate_demo_d4()

  print("Verbose D=4 jgeom test completed successfully.")
