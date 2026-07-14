#!/usr/bin/env python3
"""
Non-polynomial Poisson--Robin convergence sweep on affine D-simplices, D=1..4.

This is a Python-level integration/convergence driver using the same low-level
C-backed operators tested individually:

  L_int   from jlaplace
  T_full  from jtrace
  F_full  from jflux

It builds a smooth non-polynomial manufactured solution in D physical variables,
forms f = -Delta u and grad u with SymPy, assembles the stacked Poisson--Robin
least-squares system, solves it, and plots rel_L2 / rel_resid versus polynomial
order n.

Run examples:

  python testing/jpoisson_robin_nonpoly_convergence.py --D all --n-min 2 --n-max 7
  python testing/jpoisson_robin_nonpoly_convergence.py --D 4 --n-min 2 --n-max 6 --q-pad 2
  python testing/jpoisson_robin_nonpoly_convergence.py --D 3 --solver lstsq

If your LSMR shim is still verbose with nout=0, either patch the shim to route
nout<=0 to /dev/null, or run with --solver lstsq while inspecting convergence.
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)
from jquad_tprod import jquad_mapped_build_kappa
from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod
from jgeom import affine_from_verts
from jperms import (
  face_vertices,
  face_sigma_array,
  face_sigma_index,
  perm_to_lehmer_index,
  common_face_kappa,
)
from jtrace import assemble_T_full_common
from jflux import assemble_F_full_common
from jlaplace import assemble_L_int

try:
  from jlsmr import lsmr_dense_solve
except Exception:  # useful before LSMR install
  lsmr_dense_solve = None


def _dense(A):
  if hasattr(A, "toarray"):
    return A.toarray()
  return np.asarray(A, dtype=np.float64)


def build_basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def dk_natural(D, axis):
  """Jacobi parameter shift used by the existing tprod differentiation path."""
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def build_reference_second_partials(D, n, q_vol, kappa_src, kappa_lap):
  """Build promoted reference second-partial blocks Lij_ref[:,:,i,j]."""
  _, _, _, M = build_basis_structs(D, n, kappa_src)
  Lij = np.zeros((M, M, D, D), dtype=np.float64, order="F")

  D1 = []
  k1 = []
  for i in range(D):
    Di = dmat_build_tprod_natural_pruned(D, n, q_vol, kappa_src, i)
    D1.append(Di)
    k1.append(kappa_src + dk_natural(D, i))

  for i in range(D):
    for j in range(D):
      Dj = dmat_build_tprod_natural_pruned(D, n, q_vol, k1[i], j)
      k2 = k1[i] + dk_natural(D, j)
      D_ij_raw = Dj @ D1[i]
      K = kmat_build_tprod(D, n, q_vol, k2, kappa_lap)
      Lij[:, :, i, j] = _dense(K @ D_ij_raw)

  return Lij


def make_affine_simplex_vertices(D):
  """Return V with shape (D,D+1), columns are physical simplex vertices."""
  v0 = np.array([0.07 * ((-1.0) ** i) + 0.03 * i for i in range(D)], dtype=np.float64)
  B = np.eye(D, dtype=np.float64)
  for i in range(D):
    B[i, i] = 1.12 + 0.13 * i
  for i in range(D):
    for j in range(D):
      if i != j:
        B[i, j] = 0.035 * ((-1.0) ** (i + j)) * (i + 1.0) / (j + 2.0)

  V = np.zeros((D, D + 1), dtype=np.float64, order="F")
  V[:, 0] = v0
  for j in range(D):
    V[:, j + 1] = v0 + B[:, j]
  return np.asfortranarray(V)


def make_global_vids(D):
  base = np.array([42, 7, 100, 13, 55], dtype=np.int32)
  if D + 1 <= base.size:
    return base[:D + 1].copy()
  return np.arange(11, 11 + D + 1, dtype=np.int32)


def make_nonpoly_u_f_grad_D(D, simplify=False):
  """Build numpy-callable u, f=-Delta u, grad u in D physical variables."""
  xs = sp.symbols(f"x0:{D}", real=True)

  lin = sum(sp.Rational(i + 2, 7) * xs[i] for i in range(D))
  quad = sum(sp.Rational(i + 1, 11) * xs[i] ** 2 for i in range(D))
  mix = 0
  for i in range(D):
    for j in range(i + 1, D):
      mix += sp.Rational(1, 13 + i + 2 * j) * xs[i] * xs[j]

  u_expr = (
    sp.exp(sp.Rational(1, 5) * sp.cos(lin + sp.Rational(1, 3) * quad))
    + sp.sin(lin + mix)
    + sp.Rational(1, 10) * sp.exp(-quad)
  )

  grad_expr = [sp.diff(u_expr, x) for x in xs]
  lap_expr = sum(sp.diff(u_expr, x, 2) for x in xs)
  f_expr = -lap_expr

  if simplify:
    grad_expr = [sp.simplify(g) for g in grad_expr]
    f_expr = sp.simplify(f_expr)

  u_fun = sp.lambdify(xs, u_expr, "numpy")
  f_fun = sp.lambdify(xs, f_expr, "numpy")
  grad_funs = [sp.lambdify(xs, g, "numpy") for g in grad_expr]

  def grad_fun(*args):
    return tuple(g(*args) for g in grad_funs)

  return u_fun, f_fun, grad_fun


def eval_scalar_D(fun, P):
  P = np.asarray(P, dtype=np.float64)
  D = P.shape[1]
  vals = fun(*[P[:, i] for i in range(D)])
  vals = np.asarray(vals, dtype=np.float64)
  if vals.shape == ():
    vals = np.full(P.shape[0], float(vals), dtype=np.float64)
  return np.ravel(vals).astype(np.float64, copy=False)


def eval_grad_D(grad_fun, P):
  P = np.asarray(P, dtype=np.float64)
  D = P.shape[1]
  parts = grad_fun(*[P[:, i] for i in range(D)])
  if len(parts) != D:
    raise ValueError(f"grad_fun returned {len(parts)} components, expected {D}")
  out = np.empty((P.shape[0], D), dtype=np.float64)
  for j, g in enumerate(parts):
    g = np.asarray(g, dtype=np.float64)
    if g.shape == ():
      out[:, j] = float(g)
    else:
      out[:, j] = np.ravel(g)
  return out


def affine_map_ref_to_phys(V_phys, Xhat):
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  Xhat = np.asarray(Xhat, dtype=np.float64)
  D = V_phys.shape[0]
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def canonical_face_bary(Y):
  Y = np.asarray(Y, dtype=np.float64)
  nq, d = Y.shape
  B = np.empty((nq, d + 1), dtype=np.float64)
  B[:, 0] = 1.0 - np.sum(Y, axis=1)
  B[:, 1:] = Y
  return B


def face_points_in_volume_ref(D, face_id, sigma_local_to_canonical, Y):
  sigma = np.asarray(sigma_local_to_canonical, dtype=np.int64)
  if sigma.shape != (D,):
    raise ValueError(f"sigma must have shape ({D},)")

  B_can = canonical_face_bary(Y)
  nq = B_can.shape[0]
  lam_vol = np.zeros((nq, D + 1), dtype=np.float64)
  fv = face_vertices(D, face_id).astype(np.int64)

  for i_local in range(D):
    i_can = int(sigma[i_local])
    lam_vol[:, fv[i_local]] = B_can[:, i_can]

  return np.asfortranarray(lam_vol[:, 1:]), lam_vol


def physical_face_geometry(D, V_phys, global_vids):
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  global_vids = np.asarray(global_vids, dtype=np.int32)
  nface = D + 1

  face_scale = np.empty(nface, dtype=np.float64)
  unit_normal = np.empty((nface, D), dtype=np.float64)
  normal_scaled = np.empty((nface, D), dtype=np.float64)

  if D == 1:
    x0 = float(V_phys[0, 0])
    x1 = float(V_phys[0, 1])
    sgn = 1.0 if x1 >= x0 else -1.0
    # face_id=0 is opposite vertex 0, i.e. endpoint vertex 1.
    unit_normal[0, 0] = sgn
    unit_normal[1, 0] = -sgn
    face_scale[:] = 1.0
    normal_scaled[:, :] = unit_normal
    return face_scale, unit_normal, np.ascontiguousarray(normal_scaled)

  for face_id in range(nface):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      i_can = int(sigma[i_local])
      P_can[:, i_can] = V_phys[:, fv[i_local]]

    E = P_can[:, 1:] - P_can[:, [0]]
    G = E.T @ E
    s = math.sqrt(float(np.linalg.det(G)))

    _, _, vh = np.linalg.svd(E.T, full_matrices=True)
    n = vh[-1, :].copy()
    n /= np.linalg.norm(n)

    p_opp = V_phys[:, face_id]
    to_opp = p_opp - P_can[:, 0]
    if float(np.dot(n, to_opp)) > 0.0:
      n *= -1.0

    face_scale[face_id] = s
    unit_normal[face_id, :] = n
    normal_scaled[face_id, :] = s * n

  return face_scale, unit_normal, np.ascontiguousarray(normal_scaled)


def build_common_face_basis_generic(D, n_face, kappa_vol, q_face):
  if D == 1:
    return {
      "kappa_face": np.empty(0, dtype=np.float64),
      "Y": np.empty((1, 0), dtype=np.float64),
      "W_face": np.ones(1, dtype=np.float64),
      "Vt_common": np.asfortranarray(np.ones((1, 1), dtype=np.float64)),
      "kf": 1,
    }

  kappa_face = common_face_kappa(kappa_vol)
  alpha_face, tail_face, invh_face, kf = build_basis_structs(D - 1, n_face, kappa_face)
  Y, W_face = jquad_mapped_build_kappa(D - 1, q_face, kappa_face)
  Vt_common = jbasis_eval_all(Y, kappa_face, n_face, alpha_face, tail_face, invh_face, D - 1)
  return {
    "kappa_face": kappa_face,
    "Y": Y,
    "W_face": np.ascontiguousarray(W_face),
    "Vt_common": np.asfortranarray(Vt_common),
    "kf": kf,
  }


def _normalize_grad_shape(dV, nq, M, D):
  dV = np.asarray(dV, dtype=np.float64)
  if dV.shape == (nq, M, D):
    return dV
  if dV.shape == (nq, D, M):
    return np.transpose(dV, (0, 2, 1))
  raise ValueError(f"unexpected gradient shape {dV.shape}; expected ({nq},{M},{D})")


def eval_basis_with_grad(X, kappa, n, alpha, tail, invh, D):
  V, dV = jbasis_eval_all_with_grad(X, kappa, n, alpha, tail, invh, D)
  V = np.asarray(V, dtype=np.float64)
  dV = _normalize_grad_shape(dV, V.shape[0], V.shape[1], D)
  return V, dV


def all_sigma_tuples(D):
  out = [None] * math.factorial(D)
  for p in itertools.permutations(range(D)):
    p = np.asarray(p, dtype=np.int32)
    idx = int(perm_to_lehmer_index(p))
    out[idx] = p
  if any(x is None for x in out):
    raise RuntimeError("failed to generate all sigma tuples")
  return out


def build_boundary_inputs(D, n, kappa_vol, q_face, V_phys, global_vids):
  face_basis = build_common_face_basis_generic(D, n, kappa_vol, q_face)
  Y = face_basis["Y"]
  W_face = face_basis["W_face"]
  Vt_common = face_basis["Vt_common"]
  kf = int(face_basis["kf"])

  alpha_vol, tail_vol, invh_vol, M = build_basis_structs(D, n, kappa_vol)
  nface = D + 1
  sigmas = all_sigma_tuples(D)
  nsigma = len(sigmas)
  nq = Y.shape[0]

  Vv_sigma_face = np.empty((nq, M, nsigma, nface), dtype=np.float64, order="F")
  dVv_hat_sigma_face = np.empty((nq, M, D, nsigma, nface), dtype=np.float64, order="F")

  for face_id in range(nface):
    for idx, sigma in enumerate(sigmas):
      Xf, _ = face_points_in_volume_ref(D, face_id, sigma, Y)
      Vv, dVv = eval_basis_with_grad(Xf, kappa_vol, n, alpha_vol, tail_vol, invh_vol, D)
      Vv_sigma_face[:, :, idx, face_id] = Vv
      dVv_hat_sigma_face[:, :, :, idx, face_id] = dVv

  face_sigma = np.empty(nface, dtype=np.int32)
  sigma_arrays = []
  for face_id in range(nface):
    sig = face_sigma_array(global_vids, face_id).astype(np.int32)
    idx = int(face_sigma_index(global_vids, face_id))
    idx2 = int(perm_to_lehmer_index(sig))
    if idx != idx2:
      raise AssertionError((face_id, idx, idx2, sig))
    face_sigma[face_id] = idx
    sigma_arrays.append(sig)

  face_scale, unit_normal, normal_scaled = physical_face_geometry(D, V_phys, global_vids)

  return {
    "Y": Y,
    "W_face": W_face,
    "Vt_common": Vt_common,
    "kf": kf,
    "M": M,
    "nq": nq,
    "face_sigma_index": face_sigma,
    "sigma_arrays": sigma_arrays,
    "Vv_sigma_face": Vv_sigma_face,
    "dVv_hat_sigma_face": dVv_hat_sigma_face,
    "face_scale": face_scale,
    "unit_normal": unit_normal,
    "normal_scaled": normal_scaled,
  }


def project_source_int(D, n_int, kappa, q_vol, V_phys, f_fun):
  alpha, tail, invh, _ = build_basis_structs(D, n_int, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n_int, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_D(f_fun, P)
  detBabs = float(affine_from_verts(V_phys)["detBabs"])
  return detBabs * (V.T @ (What * vals))


def project_solution_coeffs(D, n, kappa, q_vol, V_phys, u_fun):
  alpha, tail, invh, _ = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_D(u_fun, P)
  return V.T @ (What * vals)


def project_robin_bnd(D, n_face, kappa_vol, q_face, V_phys, global_vids,
                      alpha_robin, beta_robin, u_fun, grad_u_fun, face_data):
  alpha_robin = np.asarray(alpha_robin, dtype=np.float64)
  beta_robin = np.asarray(beta_robin, dtype=np.float64)
  nface = D + 1
  if alpha_robin.shape != (nface,) or beta_robin.shape != (nface,):
    raise ValueError(f"alpha_robin and beta_robin must have shape ({nface},)")

  Y = np.asarray(face_data["Y"], dtype=np.float64)
  W_face = np.asarray(face_data["W_face"], dtype=np.float64)
  Vt_common = np.asarray(face_data["Vt_common"], dtype=np.float64)
  kf = int(face_data["kf"])
  face_scale = np.asarray(face_data["face_scale"], dtype=np.float64)
  unit_normal = np.asarray(face_data["unit_normal"], dtype=np.float64)
  sigma_arrays = face_data["sigma_arrays"]

  g = np.empty(nface * kf, dtype=np.float64)

  for face_id in range(nface):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    Xhat, _ = face_points_in_volume_ref(D, face_id, sigma_arrays[face_id], Y)
    P = affine_map_ref_to_phys(V_phys, Xhat)

    u_vals = eval_scalar_D(u_fun, P)
    grad_vals = eval_grad_D(grad_u_fun, P)
    q_vals = grad_vals @ unit_normal[face_id, :]
    robin_vals = alpha_robin[face_id] * u_vals + beta_robin[face_id] * q_vals

    g[sl] = Vt_common.T @ ((W_face * face_scale[face_id]) * robin_vals)

  return g


def row_rms_scale(A):
  A = np.asarray(A, dtype=np.float64)
  return 1.0 / max(1.0e-300, np.linalg.norm(A, ord="fro") / math.sqrt(A.size))


def solve_system(A, b, solver="lsmr", nout=0, atol=1e-12, btol=1e-12, itnlim=10000):
  A = np.asfortranarray(A, dtype=np.float64)
  b = np.ascontiguousarray(b, dtype=np.float64)

  if solver == "lsmr":
    if lsmr_dense_solve is None:
      raise RuntimeError("jlsmr.lsmr_dense_solve is unavailable; use --solver lstsq")
    x, info = lsmr_dense_solve(
      A,
      b,
      damp=0.0,
      atol=atol,
      btol=btol,
      conlim=1.0e14,
      itnlim=itnlim,
      nout=nout,
      return_info=True,
    )
    info["solver"] = "lsmr"
    return x, info

  if solver == "lstsq":
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, {
      "solver": "lstsq",
      "istop": -1,
      "itn": -1,
      "normr": float(np.linalg.norm(A @ x - b)),
      "normA": float(np.linalg.norm(A)),
      "condA": float(np.linalg.cond(A)),
      "normAr": float(np.linalg.norm(A.T @ (A @ x - b))),
    }

  raise ValueError(f"unknown solver {solver!r}")


def run_one(D, n, q_vol, q_face, q_eval, solver, nout, atol, btol):
  if n < 2:
    raise ValueError("n must be at least 2")

  kappa = np.array([0.73 + 0.19 * i for i in range(D + 1)], dtype=np.float64)
  global_vids = make_global_vids(D)
  V_phys = make_affine_simplex_vertices(D)

  u_fun, f_fun, grad_u_fun = make_nonpoly_u_f_grad_D(D, simplify=False)

  geom = affine_from_verts(V_phys)
  BinvT = geom["BinvT"]
  G = np.asfortranarray(BinvT.T @ BinvT)
  detBabs = float(geom["detBabs"])

  _, _, _, M = build_basis_structs(D, n, kappa)
  _, _, _, m_int = build_basis_structs(D, n - 2, kappa)

  Lij_ref = build_reference_second_partials(D, n, q_vol, kappa, kappa)
  L_int = assemble_L_int(D, G, detBabs, Lij_ref, m_int)

  bnd = build_boundary_inputs(D, n, kappa, q_face, V_phys, global_vids)
  T_full = assemble_T_full_common(
    D, M, bnd["kf"], bnd["nq"], bnd["face_sigma_index"], bnd["face_scale"],
    bnd["Vt_common"], bnd["W_face"], bnd["Vv_sigma_face"]
  )
  F_full = assemble_F_full_common(
    D, M, bnd["kf"], bnd["nq"], bnd["face_sigma_index"], bnd["normal_scaled"],
    BinvT, bnd["Vt_common"], bnd["W_face"], bnd["dVv_hat_sigma_face"]
  )

  nface = D + 1
  alpha_robin = np.array([1.0 + 0.07 * f for f in range(nface)], dtype=np.float64)
  beta_robin = np.array([0.18 + 0.025 * ((2 * f + 1) % 5) for f in range(nface)], dtype=np.float64)

  R = np.empty_like(T_full)
  kf = bnd["kf"]
  for face_id in range(nface):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    R[sl, :] = alpha_robin[face_id] * T_full[sl, :] + beta_robin[face_id] * F_full[sl, :]

  f_int = project_source_int(D, n - 2, kappa, q_vol, V_phys, f_fun)
  g_bnd = project_robin_bnd(
    D, n, kappa, q_face, V_phys, global_vids,
    alpha_robin, beta_robin, u_fun, grad_u_fun, bnd,
  )

  sL = row_rms_scale(L_int)
  sR = row_rms_scale(R)
  A = np.asfortranarray(np.vstack((sL * L_int, sR * R)))
  b = np.concatenate((sL * (-f_int), sR * g_bnd))

  c_sol, info = solve_system(A, b, solver=solver, nout=nout, atol=atol, btol=btol)
  c_ref = project_solution_coeffs(D, n, kappa, q_eval, V_phys, u_fun)

  rel_coef = np.linalg.norm(c_sol - c_ref) / max(1.0e-300, np.linalg.norm(c_ref))
  rel_resid = np.linalg.norm(A @ c_sol - b) / max(1.0e-300, np.linalg.norm(b))

  alpha_n, tail_n, invh_n, _ = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_eval, kappa)
  V_eval = jbasis_eval_all(Xhat, kappa, n, alpha_n, tail_n, invh_n, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  u_true = eval_scalar_D(u_fun, P)
  u_num = V_eval @ c_sol
  rel_L2 = math.sqrt(
    float(np.sum(What * (u_num - u_true) ** 2)) /
    max(1.0e-300, float(np.sum(What * u_true ** 2)))
  )

  return {
    "D": D,
    "n": n,
    "q_vol": q_vol,
    "q_face": q_face,
    "q_eval": q_eval,
    "M": M,
    "m_int": m_int,
    "kf": kf,
    "rows": A.shape[0],
    "detBabs": detBabs,
    "rel_coef": float(rel_coef),
    "rel_L2": float(rel_L2),
    "rel_resid": float(rel_resid),
    "solver": info.get("solver", solver),
    "itn": int(info.get("itn", -1)),
    "istop": int(info.get("istop", -1)),
    "normr": float(info.get("normr", np.nan)),
    "normAr": float(info.get("normAr", np.nan)),
    "condA": float(info.get("condA", np.nan)),
  }


def parse_D_list(s):
  if str(s).lower() == "all":
    return [1, 2, 3, 4]
  out = []
  for part in str(s).split(","):
    part = part.strip()
    if part:
      out.append(int(part))
  if not out:
    raise ValueError("empty D list")
  return out


def save_csv(rows, path):
  keys = [
    "D", "n", "q_vol", "q_face", "q_eval", "M", "m_int", "kf", "rows",
    "rel_coef", "rel_L2", "rel_resid", "itn", "istop", "condA",
  ]
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
      f.write(",".join(str(r[k]) for k in keys) + "\n")

import math
def dimPi(D, n):
  return math.comb(n + D, D)

def save_plot(rows, path, metric):
  plt.figure(figsize=(7.4, 5.0))
  Ds = sorted(set(r["D"] for r in rows))

  for D in Ds:
    sub = [r for r in rows if r["D"] == D]
    sub.sort(key=lambda r: r["n"])

    ns = np.array([dimPi(D, r["n"]) for r in sub], dtype=np.int64)
    vals = np.array([r[metric] for r in sub], dtype=np.float64)

    plt.semilogy(ns, vals, marker="o", label=f"D={D}")

  plt.xlabel(r"$\dim(\Pi_n^D)$")
  plt.ylabel(metric)
  plt.title(f"Poisson--Robin non-polynomial convergence ({metric})")
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=200)


def main():
  parser = argparse.ArgumentParser(description="Non-polynomial Poisson--Robin convergence sweep for D=1..4.")
  parser.add_argument("--D", default="all", help="Dimension list, e.g. all or 1,2,3,4")
  parser.add_argument("--n-min", type=int, default=2)
  parser.add_argument("--n-max", type=int, default=7)
  parser.add_argument("--q-pad", type=int, default=2, help="Use q_vol=q_face=n+q_pad")
  parser.add_argument("--q-eval-pad", type=int, default=4, help="Use q_eval=n+q_eval_pad")
  parser.add_argument("--solver", choices=("lsmr", "lstsq"), default="lsmr")
  parser.add_argument("--nout", type=int, default=0, help="LSMR Fortran output unit; 0 requests quiet mode")
  parser.add_argument("--atol", type=float, default=1e-12)
  parser.add_argument("--btol", type=float, default=1e-12)
  parser.add_argument("--metric", choices=("rel_L2", "rel_coef", "rel_resid"), default="rel_L2")
  parser.add_argument("--out-prefix", default=None)
  parser.add_argument("--no-plot", action="store_true")
  args = parser.parse_args()

  Ds = parse_D_list(args.D)
  rows = []

  for D in Ds:
    for n in range(args.n_min, args.n_max + 1):
      q_vol = n + args.q_pad
      q_face = 1 if D == 1 else n + args.q_pad
      q_eval = n + args.q_eval_pad
      r = run_one(
        D=D,
        n=n,
        q_vol=q_vol,
        q_face=q_face,
        q_eval=q_eval,
        solver=args.solver,
        nout=args.nout,
        atol=args.atol,
        btol=args.btol,
      )
      rows.append(r)
      print(
        f"D={D} n={n:2d} qv={q_vol:2d} qf={q_face:2d} qe={q_eval:2d} "
        f"M={r['M']:5d} rows={r['rows']:5d} "
        f"rel_L2={r['rel_L2']:.3e} rel_coef={r['rel_coef']:.3e} "
        f"rel_resid={r['rel_resid']:.3e} "
        f"solver={r['solver']} itn={r['itn']} istop={r['istop']} condA={r['condA']:.3e}"
      )

  this_dir = os.path.dirname(os.path.abspath(__file__))
  out_prefix = args.out_prefix
  if out_prefix is None:
    out_prefix = os.path.join(this_dir, "jpoisson_robin_nonpoly_convergence")

  csv_path = out_prefix + ".csv"
  save_csv(rows, csv_path)
  print(f"\nSaved CSV to: {csv_path}")

  if not args.no_plot:
    png_path = out_prefix + f"_{args.metric}.png"
    save_plot(rows, png_path, args.metric)
    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
  main()
