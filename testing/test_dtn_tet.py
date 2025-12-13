import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod

from jquad_tprod import jquad_mapped_build_kappa
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)


def dim_Pi(D, n):
  from math import comb
  return comb(n + D, D)


def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def kappa_face_from_kappa_src(kappa_src, face_id):
  if face_id == 0:  # x=0 -> drop k1
    return np.array([kappa_src[1], kappa_src[2], kappa_src[3]], dtype=np.float64)
  if face_id == 1:  # y=0 -> drop k2
    return np.array([kappa_src[0], kappa_src[2], kappa_src[3]], dtype=np.float64)
  if face_id == 2:  # z=0 -> drop k3
    return np.array([kappa_src[0], kappa_src[1], kappa_src[3]], dtype=np.float64)
  if face_id == 3:  # x+y+z=1 -> drop k4
    return np.array([kappa_src[0], kappa_src[1], kappa_src[2]], dtype=np.float64)
  raise ValueError("bad face_id")


def face_map_and_geom(face_id, u, v):
  if face_id == 0:
    Xf = np.stack([0*u, u, v], axis=1)
    area_scale = 0.5
    nvec = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
  elif face_id == 1:
    Xf = np.stack([u, 0*u, v], axis=1)
    area_scale = 0.5
    nvec = np.array([0.0, -1.0, 0.0], dtype=np.float64)
  elif face_id == 2:
    Xf = np.stack([u, v, 0*u], axis=1)
    area_scale = 0.5
    nvec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  elif face_id == 3:
    Xf = np.stack([u, v, 1.0 - u - v], axis=1)
    area_scale = np.sqrt(3.0) / 2.0
    nvec = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
  else:
    raise ValueError("bad face_id")
  return Xf, area_scale, nvec


def build_laplacian_promoted(D, n, q_vol, kappa_src):
  M = dim_Pi(D, n)
  kappa_lap = kappa_src + 2.0 * np.ones(D + 1, dtype=np.float64)

  A = np.zeros((M, M), dtype=np.float64)

  for axis in range(D):
    D1 = dmat_build_tprod_natural_pruned(D, n, q_vol, kappa_src, axis)
    kappa1 = kappa_src + dk_natural(D, axis)

    D2 = dmat_build_tprod_natural_pruned(D, n, q_vol, kappa1, axis)
    kappa2 = kappa1 + dk_natural(D, axis)

    Dxx = D2 @ D1

    K = kmat_build_tprod(D, n, q_vol, kappa2, kappa_lap)
    A += K @ Dxx

  return A, kappa_lap


def project_rhs_to_kappa(D, n, q_vol, kappa, f_fun, sign=+1.0):
  X, w = jquad_mapped_build_kappa(D, q_vol, kappa)
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha, tail, invh, D)

  fv = f_fun(X[:, 0], X[:, 1], X[:, 2])
  b = V.T @ (w * (sign * fv))
  return b


def assemble_TF_and_g_full(D, n, q_face, kappa_src, g_dirichlet):
  if D != 3:
    raise ValueError("assemble_TF_and_g_full assumes D=3")

  alpha_src, tail_src, invh_src = jbasis_build_structures(D, n, kappa_src)
  M = alpha_src.shape[0]

  T_blocks = []
  F_blocks = []
  g_blocks = []

  for face_id in range(4):
    kappa_tri = kappa_face_from_kappa_src(kappa_src, face_id)

    Xt, wt = jquad_mapped_build_kappa(2, q_face, kappa_tri)
    u = Xt[:, 0]
    v = Xt[:, 1]

    Xf, area_scale, nvec = face_map_and_geom(face_id, u, v)
    wf = wt * area_scale

    alpha_tri, tail_tri, invh_tri = jbasis_build_structures(2, n, kappa_tri)
    Vt = jbasis_eval_all(Xt, kappa_tri, n, alpha_tri, tail_tri, invh_tri, 2)

    Vv, dVv = jbasis_eval_all_with_grad(Xf, kappa_src, n, alpha_src, tail_src, invh_src, D)

    Tf = Vt.T @ (wf[:, None] * Vv)

    ndot = (nvec[0] * dVv[:, :, 0] +
            nvec[1] * dVv[:, :, 1] +
            nvec[2] * dVv[:, :, 2])
    Ff = Vt.T @ (wf[:, None] * ndot)

    gv = g_dirichlet(Xf[:, 0], Xf[:, 1], Xf[:, 2])
    gf = Vt.T @ (wf * gv)

    T_blocks.append(Tf)
    F_blocks.append(Ff)
    g_blocks.append(gf)

  T_full = np.vstack(T_blocks)
  F_full = np.vstack(F_blocks)
  g_full = np.concatenate(g_blocks)

  return T_full, F_full, g_full


def row_reduce_trace_space(T_full, F_full, g_full, k):
  # Normalize rows of T_full for pivot selection
  r = np.linalg.norm(T_full, axis=1)
  r[r == 0] = 1.0
  Tn = (T_full.T / r).T

  # Pivot on normalized T
  _, _, piv = scipy.linalg.qr(Tn.T, pivoting=True, mode="economic")
  sel = np.array(piv[:k], dtype=np.int64)

  # Apply selection to the original (un-normalized) operators/data
  T = T_full[sel, :]
  F = F_full[sel, :]
  g = g_full[sel]
  return T, F, g, sel

def equilibrate(A, b):
  # Row scale
  r = np.linalg.norm(A, axis=1)
  r[r == 0] = 1.0
  A1 = (A.T / r).T
  b1 = b / r

  # Column scale
  c = np.linalg.norm(A1, axis=0)
  c[c == 0] = 1.0
  A2 = A1 / c

  return A2, b1, r, c


def solve_equilibrated_square(A, b):
  Aeq, beq, r, c = equilibrate(A, b)

  # QR-based solve (square)
  x, *_ = np.linalg.lstsq(Aeq, beq, rcond=None)

  # Undo column scaling
  x = x / c
  return x


def row_scale(A, b=None):
  r = np.linalg.norm(A, axis=1)
  r[r == 0] = 1.0
  A2 = (A.T / r).T
  if b is None:
    return A2, r
  return A2, b / r, r


def reduce_trace_space_svd(T_full, F_full, g_full, k):
  # Thin SVD of T_full (Nbf x M)
  U, s, Vt = np.linalg.svd(T_full, full_matrices=False)

  # Orthonormal basis for the (numerical) row space (dimension k)
  Uk = U[:, :k]

  # Reduced operators/data in that orthonormal trace basis
  T = Uk.T @ T_full          # (k x M)
  F = Uk.T @ F_full          # (k x M)  (optional; not needed for Dirichlet solve)
  g = Uk.T @ g_full          # (k,)

  return T, F, g, s

def dtn_leaf_solve_dirichlet_nullspace(L_int, f_int, T_full, g_full, k, tol=1e-14):
  """
  Solve L_int c = -f_int with Dirichlet T_full c = g_full by:
    c = c_p + N y
    (T_full N) y = g_full - T_full c_p  (square via pivot row selection)
  """

  M_int, M = L_int.shape

  # --- Particular solution c_p via pivoted QR on L_int (column pivots)
  _, _, piv = scipy.linalg.qr(L_int, pivoting=True, mode="economic")
  piv = np.array(piv, dtype=np.int64)

  I = piv[:M_int]  # square block
  A = L_int[:, I]
  c_p = np.zeros(M, dtype=np.float64)
  c_p[I] = scipy.linalg.solve(A, -f_int, assume_a="gen")

  # --- Nullspace basis N via SVD (right singular vectors)
  # L_int = U S Vt, nullspace columns are last k columns of V (i.e. Vt.T[:, -k:])
  U, s, Vt = np.linalg.svd(L_int, full_matrices=True)
  N = Vt.T[:, -k:]   # (M, k)

  # --- Reduced boundary system
  Ared = T_full @ N                 # (Nb, k)
  bred = g_full - (T_full @ c_p)    # (Nb,)

  # Select k independent rows using pivoted QR on Ared^T
  _, _, pivr = scipy.linalg.qr(Ared.T, pivoting=True, mode="economic")
  sel = np.array(pivr[:k], dtype=np.int64)

  A_sq = Ared[sel, :]     # (k, k)
  b_sq = bred[sel]        # (k,)

  y = scipy.linalg.solve(A_sq, b_sq, assume_a="gen")

  c = c_p + N @ y
  return c


def dtn_leaf_solve_dirichlet_stacked(D, n, q_vol, q_face, kappa_src, f_rhs, g_dirichlet,
                                     build_dtn=False):
  """
  Solve -Δu=f on reference tet with Dirichlet u=g using the square stacked system:
    [ T ] c = [ g_red ]
    [ L ]     [ -f_int ]

  Returns:
    c (M,) volume coefficients in kappa_src basis.

  If build_dtn=True, also returns (S, q_f) in the reduced boundary space (size k).
  """
  if D != 3:
    raise ValueError("dtn_leaf_solve_dirichlet_stacked assumes D=3")
  if n < 2:
    raise ValueError("require n>=2 for Poisson leaf (nonempty interior operator)")

  M = dim_Pi(D, n)
  M_int = dim_Pi(D, n - 2)
  k = (n + 1) * (n + 1)

  # Laplacian (promoted) and interior block
  A_lap, kappa_lap = build_laplacian_promoted(D, n, q_vol, kappa_src)
  # We solve -Δu = f => interior equation: -L_int c = f_int
  # Our A_lap maps c -> (Δu)_coeffs in kappa_lap, so take L_int = A_lap[:M_int,:]
  L_int = A_lap[:M_int, :]
  f_lap = project_rhs_to_kappa(D, n, q_vol, kappa_lap, f_rhs, sign=+1.0)
  f_int = f_lap[:M_int]

  # Trace/flux maps and boundary moments
  T_full, F_full, g_full = assemble_TF_and_g_full(D, n, q_face, kappa_src, g_dirichlet)

  # Reduce boundary space to trace dimension k
  #T, F, g_red, _ = row_reduce_trace_space(T_full, F_full, g_full, k)
  T, F, g_red, s = reduce_trace_space_svd(T_full, F_full, g_full, k)



  # Build square stacked system
  alpha = np.linalg.norm(L_int) / (np.linalg.norm(T) + 1e-300)
  C = np.vstack([alpha*T, L_int])
  rhs = np.concatenate([alpha*g_red, -f_int])
  c = solve_equilibrated_square(C, rhs)

  ## Row-scale for stability
  #Ceq, rhseq, _ = row_scale(C, rhs)

  ## Square solve via QR (stable). This is not "least squares" formulation; it's a square solve.
  #c, *_ = np.linalg.lstsq(Ceq, rhseq, rcond=None)

  if not build_dtn:
    return c

  # Build reduced DtN operator and forcing contribution in reduced boundary coordinates:
  # q = S u + q_f, where u is reduced boundary DOF vector (size k) in the selected row-basis.
  # 1) q_f: solve with u=0
  rhs_f = np.concatenate([np.zeros(k, dtype=np.float64), -f_int])
  Ceq, rhsf_eq, rown = row_scale(C, rhs_f)
  cf, *_ = np.linalg.lstsq(Ceq, rhsf_eq, rcond=None)
  q_f = F @ cf

  # 2) S: columns from unit boundary vectors
  S = np.zeros((k, k), dtype=np.float64)
  Ceq, rown = row_scale(C)
  for j in range(k):
    ej = np.zeros(k, dtype=np.float64)
    ej[j] = 1.0
    rhs_j = np.concatenate([ej, np.zeros(M_int, dtype=np.float64)])
    cj, *_ = np.linalg.lstsq(Ceq, rhs_j / rown, rcond=None)
    S[:, j] = F @ cj

  return c, S, q_f


def build_monomial_list(m):
  mons = []
  for a in range(m + 1):
    for b in range(m + 1 - a):
      for c in range(m + 1 - a - b):
        deg = a + b + c
        coeff = 1.0 / (1.0 + deg)
        mons.append((a, b, c, coeff))
  return mons


def poly_u_and_lap(m):
  mons = build_monomial_list(m)

  def u(x, y, z):
    out = np.zeros_like(x, dtype=np.float64)
    for a, b, c, coeff in mons:
      out += coeff * (x**a) * (y**b) * (z**c)
    return out

  def lap_u(x, y, z):
    out = np.zeros_like(x, dtype=np.float64)
    for a, b, c, coeff in mons:
      if a >= 2:
        out += coeff * (a * (a - 1)) * (x**(a - 2)) * (y**b) * (z**c)
      if b >= 2:
        out += coeff * (b * (b - 1)) * (x**a) * (y**(b - 2)) * (z**c)
      if c >= 2:
        out += coeff * (c * (c - 1)) * (x**a) * (y**b) * (z**(c - 2))
    return out

  return u, lap_u


def weighted_L2_error_src(D, n, c_src, q_err, kappa_src, u_exact):
  X, w = jquad_mapped_build_kappa(D, q_err, kappa_src)

  alpha, tail, invh = jbasis_build_structures(D, n, kappa_src)
  V = jbasis_eval_all(X, kappa_src, n, alpha, tail, invh, D)

  uh = V @ c_src
  ue = u_exact(X[:, 0], X[:, 1], X[:, 2])

  diff = uh - ue
  num = np.sum(w * diff * diff)
  den = np.sum(w * ue * ue) + 1e-300
  return float(np.sqrt(num / den))


def run_dtn_poly_convergence(m_max=12, n_max=20):
  D = 3
  kappa_src = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

  u_exact, lap_u = poly_u_and_lap(m_max)

  def f_rhs(x, y, z):
    return -lap_u(x, y, z)

  def g_dirichlet(x, y, z):
    return u_exact(x, y, z)

  ns = np.arange(2, n_max + 1)
  dofs = np.array([dim_Pi(D, int(n)) for n in ns], dtype=np.int64)
  errs = np.zeros_like(ns, dtype=np.float64)

  for idx, n in enumerate(ns):
    n = int(n)

    q_face = n + 2
    q_vol = n + 2
    q_err = n+2#max(3 * m_max + 10, 40)
    # --- Dimensions ---
    M     = dim_Pi(D, n)
    M_int = dim_Pi(D, n - 2)
    k     = (n + 1) * (n + 1)
    
    # --- Build promoted Laplacian ---
    A_lap, kappa_lap = build_laplacian_promoted(D, n, q_vol, kappa_src)
    
    # --- RHS projection: -Δu = f ---
    # A_lap maps c -> (Δu)_coeffs, so we solve L_int c = -f_int
    f_lap = project_rhs_to_kappa(
      D, n, q_vol, kappa_lap, f_rhs, sign=+1.0
    )
    
    # --- Interior rows (degree <= n-2) ---
    L_int = A_lap[:M_int, :]
    f_int = f_lap[:M_int]
    
    # --- Full boundary trace + data (NO reduction) ---
    T_full, _, g_full = assemble_TF_and_g_full(
      D, n, q_face, kappa_src, g_dirichlet
    )
    
    # --- DtN leaf solve via nullspace completion ---
    c = dtn_leaf_solve_dirichlet_nullspace(
      L_int,
      f_int,
      T_full,
      g_full,
      k
    )

    #c = dtn_leaf_solve_dirichlet_stacked(
    #  D=D,
    #  n=n,
    #  q_vol=q_vol,
    #  q_face=q_face,
    #  kappa_src=kappa_src,
    #  f_rhs=f_rhs,
    #  g_dirichlet=g_dirichlet,
    #  build_dtn=False,
    #)

    errs[idx] = weighted_L2_error_src(D, n, c, q_err, kappa_src, u_exact)

    print(f"m={m_max:2d}  n={n:2d}  dofs={dofs[idx]:4d}  q_vol={q_vol:2d}  q_face={q_face:2d}  relL2={errs[idx]:.3e}")

  plt.figure()
  plt.semilogy(ns, errs, marker="o")
  plt.axvline(m_max, linestyle="--")
  plt.xlabel("n (basis total degree)")
  plt.ylabel("Relative weighted L2 error (kappa_src)")
  plt.title(f"DtN leaf (stacked square system) on reference tet (poly deg m={m_max})")
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.tight_layout()
  plt.show()

  plt.figure()
  plt.semilogy(dofs, errs, marker="o")
  plt.axvline(dim_Pi(D, m_max), linestyle="--")
  plt.xlabel("DOFs = dimPi(n,3)")
  plt.ylabel("Relative weighted L2 error (kappa_src)")
  plt.title(f"DtN leaf (stacked square system) on reference tet (poly deg m={m_max})")
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.tight_layout()
  plt.show()

  return ns, dofs, errs


if __name__ == "__main__":
  run_dtn_poly_convergence(m_max=8, n_max=16)

