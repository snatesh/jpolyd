import itertools
import math
import numpy as np

from jperms import (
  face_vertices,
  face_sigma_array,
  face_sigma_index,
  perm_to_lehmer_index,
  face_values_local_to_canonical,
)


def inverse_perm(p):
  p = list(map(int, p))
  inv = [0] * len(p)
  for i, pi in enumerate(p):
    inv[pi] = i
  return inv


def print_lex_table(d):
  print(f"=== Lehmer / lexicographic permutation table for length d={d} ===")
  print("index : permutation tuple")

  for idx, p in enumerate(itertools.permutations(range(d))):
    p = list(p)
    idx_from_c = perm_to_lehmer_index(p)
    assert idx_from_c == idx
    print(f"{idx:4d}  : {p}")

  print()


def run_d4_face_demo():
  D = 4
  d = D

  # A 4-simplex has local vertices 0..4.
  # Each face has 4 vertices.
  local_to_global = np.array([42, 7, 100, 13, 55], dtype=np.int32)

  print("=== Face permutation convention for one 4-simplex ===")
  print(f"local simplex vertices 0..{D} have global IDs: {local_to_global.tolist()}")
  print("face_id is the opposite local vertex")
  print("canonical face order = increasing global vertex ID")
  print("sigma_local_to_canonical[local_face_pos] = canonical_face_pos")
  print("sigma_canonical_to_local[canonical_face_pos] = local_face_pos")
  print()

  local_values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

  for face_id in range(D + 1):
    fv = face_vertices(D, face_id)
    face_global_local_order = local_to_global[fv]
    canonical_global = sorted(face_global_local_order.tolist())

    sigma_l2c = face_sigma_array(local_to_global, face_id)
    sigma_c2l = inverse_perm(sigma_l2c)

    idx = face_sigma_index(local_to_global, face_id)
    idx_l2c = perm_to_lehmer_index(sigma_l2c)
    idx_c2l = perm_to_lehmer_index(sigma_c2l)

    canonical_values = face_values_local_to_canonical(local_values, sigma_l2c)

    print(f"face_id={face_id} opposite local vertex {face_id}")
    print(f"  face local vertex numbers                  = {fv.tolist()}")
    print(f"  face global vertex IDs, local order         = {face_global_local_order.tolist()}")
    print(f"  face global vertex IDs, canonical order     = {canonical_global}")
    print(f"  sigma_local_to_canonical                    = {sigma_l2c.tolist()}")
    print(f"  sigma_canonical_to_local                    = {sigma_c2l}")
    print(f"  face_sigma_index                            = {idx}")
    print(f"  perm_to_lehmer_index(local_to_canonical)    = {idx_l2c}")
    print(f"  perm_to_lehmer_index(canonical_to_local)    = {idx_c2l}")
    print(f"  local values {local_values.tolist()} -> canonical {canonical_values.tolist()}")
    print()

    assert idx == idx_l2c

    # Check canonical values manually:
    # local_to_canonical says where each local value lands.
    manual = np.empty_like(local_values)
    for local_pos, canonical_pos in enumerate(sigma_l2c):
      manual[canonical_pos] = local_values[local_pos]

    np.testing.assert_allclose(canonical_values, manual)


def run_specific_lehmer_example():
  sigma = [2, 0, 3, 1]
  idx = perm_to_lehmer_index(sigma)

  print("=== Specific d=4 Lehmer example ===")
  print(f"sigma = {sigma}")
  print("Lehmer digits:")
  unused = list(range(len(sigma)))
  digits = []

  for i, s in enumerate(sigma):
    digit = sum(1 for u in unused if u < s)
    digits.append(digit)
    unused.remove(s)

  print(f"  digits = {digits}")

  manual_idx = 0
  d = len(sigma)
  for i, digit in enumerate(digits):
    weight = math.factorial(d - 1 - i)
    manual_idx += digit * weight
    print(f"  position {i}: {digit} * {d - 1 - i}! = {digit * weight}")

  print(f"  manual index = {manual_idx}")
  print(f"  C index      = {idx}")
  assert manual_idx == idx
  print()


if __name__ == "__main__":
  print_lex_table(4)
  run_specific_lehmer_example()
  run_d4_face_demo()
  print("Verbose d=4 jperms test completed successfully.")
