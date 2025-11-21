import ctypes
import os
from libjpolyd_loader import libjpolyd


# Signature: void jpoly_set_omp_threads(int n)
libjpolyd.jpoly_set_omp_threads.argtypes = [ctypes.c_int]
libjpolyd.jpoly_set_omp_threads.restype  = None

def set_omp_threads(n: int):
  """
  Set the number of OpenMP threads used by underlying C++ code
  (e.g. jbasis::eval_all).
  
  Equivalent to calling omp_set_dynamic(0) and omp_set_num_threads(n).
  """
  if n <= 0:
      raise ValueError("n must be a positive integer")
  libjpolyd.jpoly_set_omp_threads(int(n))

