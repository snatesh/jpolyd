import ctypes
import os
import sys
import numpy as np

# Ensure we can import python/jweight.py
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python"))

import jweight  # this loads libjweight

print("Testing jweight_w_kappa for κ ≡ 1/2:")

for D in (1, 2, 3):
    kappa = np.array([0.5] * (D + 1), dtype=np.float64)
    kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result = jweight.libjweight.jweight_w_kappa(kappa_ptr, D)
    expected = np.prod(np.arange(1, D + 1))  # factorial using NumPy only
    print(f"D = {D}, w_kappa = {result:.12f} (expected {expected})")
