import math
from ctypes import (
  c_int,
  c_double,
  POINTER,
)
import numpy as np
from jevd import jevd_serial
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad
)
from jmat import jmat_build
from libjpolyd_loader import libjpolyd


# -----------------------------------------------------------------------------
# C prototype for jquad_optimize
# -----------------------------------------------------------------------------

libjpolyd.jquad_optimize.restype = c_int
libjpolyd.jquad_optimize.argtypes = [
  c_int,                    # D
  c_int,                    # node_deg
  c_int,                    # m_basis
  POINTER(c_double),        # kappa
  POINTER(c_double),        # z_io
  POINTER(c_double),        # V_opt (nullable)
  c_int,                    # max_nlopt_eval
  c_int,                    # max_gn_iter
  c_double,                 # gn_step
  c_double,                 # tol
  c_double,                 # tol_up
  c_int,                    # verbose
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def dim_Pi(D: int, n: int) -> int:
  """
  Dimension of total-degree ≤ n polynomials on R^D:
    dim Pi_n^D = C(n + D, D).
  """
  if n < 0 or D < 1:
    return 0
  return math.comb(n + D, D)

def max_offdiag(J):
    """
    Compute max absolute off-diagonal entry across all matrices J[:,:,ell].

    Parameters
    ----------
    J : ndarray, shape (N, N, D)

    Returns
    -------
    float
        Maximum |J[i,j,ell]| for i!=j.
    """
    N, N2, D = J.shape
    assert N == N2
    mask = ~np.eye(N, dtype=bool)
    # max over each matrix, then over ell
    return np.max(np.abs(J[mask, :]))


def compute_joint_spectrum(D, n, kappa,
                           tol=1e-10,
                           max_sweeps=100):
  """
  Build D Jacobi matrices for total degree n, jointly diagonalize them,
  and return the matched eigenvalues as an array of shape (N, D).

  Parameters
  ----------
  D : int
      Dimension of the simplex.
  n : int
      Maximum total degree for the Jacobi basis (Pi_n^D).
  kappa : array_like, shape (D+1,)
      Jacobi parameters on the simplex.
  tol : float
      Threshold on |s| for JEV D (Cardoso–Souloumiac).
  max_sweeps : int
      Maximum sweeps in the JEV D algorithm.

  Returns
  -------
  eigs : ndarray, shape (N, D)
      Joint eigenvalue tuples (lambda_1, ..., lambda_D) per basis vector.
      Each row corresponds to one joint eigenvector (i.e., one column of V).
  J_diag : ndarray, shape (N, N, D)
      The approximately diagonalized Jacobi matrices.
  V : ndarray, shape (N, N)
      Joint diagonalizer (columns are joint eigenvectors).
  info : int
      JEV D info flag: 0 = converged, 1 = not converged, <0 = input error.
  sweeps : int
      Number of sweeps actually performed.
  max_offdiag : float
      Last max |s| reported by the algorithm.
  """

  kappa = np.asarray(kappa, dtype=np.float64)
  if kappa.shape[0] != D + 1:
    raise ValueError("kappa must have length D+1")

  # 1) Build Jacobi matrices for each coordinate x_i, i=1..D.
  #    Assume jmat_build returns an array J_all with shape (D, N, N),
  #    where N = dim Pi_n^D and each J_all[i,:,:] is the Jacobi
  #    matrix for coordinate x_{i+1}.
  J_all = jmat_build(D, n, kappa)   # shape (D, N, N)
  if J_all.ndim != 3 or J_all.shape[0] != D:
    raise RuntimeError("jmat_build returned unexpected shape")

  D_, N, N2 = J_all.shape
  if N != N2:
    raise RuntimeError("Jacobi matrices must be square")
  if D_ != D:
    raise RuntimeError("First dimension of J_all must be D")

  # 2) Pack into shape (N, N, D) (AoS stack of matrices) for jevd_serial.
  #    Each slice J_stack[:, :, ell] is J_{ell}, and we ensure Fortran
  #    layout so it matches the C layout J[i + m*(j + nn*m)].
  J_stack = np.asfortranarray(J_all.transpose(1, 2, 0))  # (N, N, D), F-order

  # 3) Run serial JEV D to jointly diagonalize (N x N x D) stack.
  J_diag, V, info, sweeps, max_offdiag_jevd = jevd_serial(
    J_stack,
    tol=tol,
    max_sweeps=max_sweeps,
    accumulate_V=True
  )
  # J_diag has shape (N, N, D), V is N x N (both Fortran).

  # 4) Extract eigenvalues per matrix (the diagonals).
  #    For each dimension ell = 0..D-1, take diag(J_diag[:,:,ell]).
  eig_list = []
  for ell in range(D):
    diag_ell = np.diag(J_diag[:, :, ell])
    eig_list.append(diag_ell)

  # Stack columns → shape (N, D)
  eigs = np.stack(eig_list, axis=1)

  # Optional: sort the joint eigenvalues for convenience
  # (global ordering is otherwise arbitrary but consistent across dims).
  # Here we sort lexicographically by the tuple (lambda_1, ..., lambda_D).
  sort_idx = np.lexsort(eigs.T[::-1])  # last key is first column
  eigs_sorted = eigs[sort_idx, :]
  # 5) Compute final off-diagonal residual
  final_offdiag = max_offdiag(J_diag)


  return eigs_sorted, J_diag, V, info, sweeps, max_offdiag_jevd, final_offdiag


def build_initial_guess(D, n_basis, kappa, tol=1e-12, max_sweeps=100):
  """
  Use joint spectrum as node seeds and a least-squares system
  to get an initial weight guess.

  Returns
  -------
  X0 : ndarray, shape (N, D)
  w0 : ndarray, shape (N,)
  z0 : ndarray, shape (N*(D+1),)
  """
  # Joint spectrum from Jacobi matrices
  eigs, J_diag, V_joint, info, sweeps, max_off, final_offdiag = compute_joint_spectrum(
    D, n_basis, kappa, tol=tol, max_sweeps=max_sweeps
  )

  X0 = eigs  # N x D nodes
  N = X0.shape[0]

  # Build basis structures
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n_basis, kappa)
  M = alpha_table.shape[0]

  # Vandermonde at these nodes
  V = jbasis_eval_all(X0, kappa, n_basis, alpha_table, tail_deg, inv_h, D)

  # Desired integrals = e1
  e1 = np.zeros(M)
  e1[0] = 1.0

  # Least-squares for initial weights
  # V^T w ≈ e1  =>  minimize ||V^T w - e1||^2
  # Use normal eqs or lstsq on V.
  w0, *_ = np.linalg.lstsq(V.T, e1, rcond=None)

  # Pack z0
  z0 = np.concatenate([X0.reshape(-1), w0])
  return z0, alpha_table, tail_deg, inv_h, e1

# -----------------------------------------------------------------------------
# High-level Python wrapper
# -----------------------------------------------------------------------------
def optimize_quadrature(
  D: int,
  node_deg: int,
  m_basis: int,
  kappa,
  z0: np.ndarray | None = None,
  want_V_opt: bool = False,
  max_nlopt_eval: int = 4000,
  max_gn_iter: int = 2000,
  gn_step: float = 1.0,
  tol: float = 1e-15,
  tol_up: float = 1e3,
  verbose: bool = False,
):
  """
  High-level wrapper for jquad_optimize using *JEVD-based initial guess*.

  For D ≥ 2:
      If z0 is None → build_initial_guess(D, node_deg, kappa)

  For D = 1:
      Gauss–Jacobi rule is exact, and z0 is ignored by the C backend,
      but a dummy z array must be allocated.
  """

  if D < 1:
    raise ValueError("D must be >= 1")

  kappa_arr = np.asarray(kappa, dtype=np.float64)
  if kappa_arr.shape[0] != D + 1:
    raise ValueError(f"kappa must have length {D+1}")

  # Compute N and M
  if D == 1:
    N = int(node_deg)               # D=1: node_deg is #points
  else:
    N = dim_Pi(D, node_deg)

  M = dim_Pi(D, m_basis) if m_basis >= 0 else 0

  # -----------------------------------------------------------------------------
  # Build initial guess z0
  # -----------------------------------------------------------------------------
  if z0 is None:
    if D == 1:
      # Dummy initial guess: the C++ code overwrites it immediately
      z_flat = np.zeros((2 * N,), dtype=np.float64)
    else:
      # JEVD-based initial guess
      z_flat, *_ = build_initial_guess(D, node_deg, kappa_arr)
      # z_flat is already length N*(D+1)
  else:
    # User-provided initial guess
    z_arr = np.asarray(z0, dtype=np.float64)
    if z_arr.ndim == 2:
      if z_arr.shape != (N, D + 1):
        raise ValueError(f"z0 has shape {z_arr.shape}, expected {(N, D+1)}")
      z_flat = z_arr.ravel(order="C").copy()
    elif z_arr.ndim == 1:
      expected = 2 * N if D == 1 else (D + 1) * N
      if z_arr.size != expected:
        raise ValueError(f"z0 length {z_arr.size}, expected {expected}")
      z_flat = z_arr.copy()
    else:
      raise ValueError("z0 must be 1D or 2D array-like")

  # -----------------------------------------------------------------------------
  # Prepare V_opt buffer
  # -----------------------------------------------------------------------------
  if want_V_opt and M > 0:
    V_buf = np.empty((N * M,), dtype=np.float64)
    V_ptr = V_buf.ctypes.data_as(POINTER(c_double))
  else:
    V_buf = None
    V_ptr = None

  # -----------------------------------------------------------------------------
  # Call C API
  # -----------------------------------------------------------------------------

  kappa_ptr = kappa_arr.ctypes.data_as(POINTER(c_double))
  z_ptr = z_flat.ctypes.data_as(POINTER(c_double))

  status = libjpolyd.jquad_optimize(
    int(D),
    int(node_deg),
    int(m_basis),
    kappa_ptr,
    z_ptr,
    V_ptr,
    int(max_nlopt_eval),
    int(max_gn_iter),
    float(gn_step),
    float(tol),
    float(tol_up),
    int(bool(verbose)),
  )

  # -----------------------------------------------------------------------------
  # Unpack final X and w
  # -----------------------------------------------------------------------------

  if D == 1:
    X = z_flat[:N].reshape(N, 1)
    w = z_flat[N:2 * N]
  else:
    X = z_flat[:N * D].reshape(N, D)
    w = z_flat[N * D:].copy()

  # -----------------------------------------------------------------------------
  # Unpack V_opt
  # -----------------------------------------------------------------------------
  if want_V_opt and (V_buf is not None) and (M > 0):
    V_opt = V_buf.reshape((N, M), order="F")
  else:
    V_opt = None

  return X, w, V_opt, int(status)
