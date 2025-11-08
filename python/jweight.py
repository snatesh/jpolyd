import ctypes
import numpy as np
import os

# Load shared library (adjust path as needed)
libjweight = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "install", "lib", "libjpolyd.so"))

libjweight.jweight_w_kappa.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                ctypes.c_int]
libjweight.jweight_w_kappa.restype = ctypes.c_double




