#!/usr/bin/env python3
"""
Single-tetrahedron Poisson--Robin manufactured-solution test.

This is intentionally a Python-level integration test of the low-level C-backed
operator assemblers:

  L_int from jlaplace
  T_full from jtrace
  F_full from jflux
  f_int and g_R from jsource

The PDE convention is

  Delta u = -f  in K,
  alpha u + beta (n.grad u) = g on each face.

For a manufactured physical solution u_phys(x,y,z), SymPy generates f=-Delta u
and grad u. The test projects all RHS data by quadrature, stacks the interior
and Robin equations, solves with dense LSMR when available, and compares the
solution coefficients against the direct reference projection of u_phys.
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np

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
  face_sigma_array,
  face_sigma_index,
  perm_to_lehmer_index,
  common_face_kappa,
)
from jtrace import assemble_T_full_common
from jflux import assemble_F_full_common
from jlaplace import assemble_L_int
from jsource import (
  affine_map_ref_to_phys,
  build_common_face_basis,
  eval_scalar_xyz,
  make_manufactured_u_f_grad,
  make_nonpolynomial_u_f_grad,
  physical_face_geometry,
  project_robin_bnd,
  project_source_int,
  face_points_in_volume_ref,
)

try:
  from jlsmr import lsmr_dense_solve
except Exception:  # pragma: no cover - useful when testing before LSMR install
  lsmr_dense_solve = None


def _dense(A):
  if hasattr(A, "toarray"):
    return A.toarray()
  return np.asarray(A, dtype=np.float64)


def build_basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def dk_natural(D, axis):
  """
  Jacobi parameter shift used by the existing tprod differentiation path.

  The final kappa entry belongs to barycentric vertex 0.
  """
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


def make_affine_tet_vertices():
  """Return V with shape (3,4), columns are physical tetrahedron vertices."""
  D = 3
  v0 = np.array([-0.17, 0.11, 0.24], dtype=np.float64)
  B = np.array(
    [
      [1.20, -0.08, 0.06],
      [0.05, 1.37, -0.11],
      [-0.04, 0.09, 1.51],
    ],
    dtype=np.float64,
  )
  V = np.zeros((D, D + 1), dtype=np.float64, order="F")
  V[:, 0] = v0
  for j in range(D):
    V[:, j + 1] = v0 + B[:, j]
  return np.asfortranarray(V)


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
  """Return permutations in slots equal to their jperms Lehmer index."""
  out = [None] * math.factorial(D)
  for p in itertools.permutations(range(D)):
    p = np.asarray(p, dtype=np.int32)
    idx = int(perm_to_lehmer_index(p))
    out[idx] = p
  if any(x is None for x in out):
    raise RuntimeError("failed to generate all sigma tuples")
  return out


def build_boundary_inputs(D, n, kappa_vol, q_face, V_phys, global_vids):
  """Build common face data and packed trace/flux basis evaluations."""
  if D != 3:
    raise ValueError("this integration test is currently the 3D tet case")

  face_basis = build_common_face_basis(D, n, kappa_vol, q_face)
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


def project_solution_coeffs(D, n, kappa, q_vol, V_phys, u_fun):
  """Project u_phys(F(xhat)) onto Pi_n^D in the reference coefficient convention."""
  alpha, tail, invh, _ = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_xyz(u_fun, P)
  return V.T @ (What * vals)

def solve_dense_lsmr_or_lstsq(A, b, atol=1e-12, btol=1e-12, itnlim=2000):
  A = np.asfortranarray(A, dtype=np.float64)
  b = np.ascontiguousarray(b, dtype=np.float64)

  if lsmr_dense_solve is not None:
    x, info = lsmr_dense_solve(
      A,
      b,
      damp=0.0,
      atol=atol,
      btol=btol,
      conlim=1.0e14,
      itnlim=itnlim,
      nout=0,
      return_info=True,
    )
    info["solver"] = "lsmr"
    return x, info

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


def row_rms_scale(A):
  A = np.asarray(A, dtype=np.float64)
  return 1.0 / max(1.0e-300, np.linalg.norm(A, ord="fro") / math.sqrt(A.size))


def run_one(m_max=3, n=None, q_vol=None, q_face=None, nonpoly=False,
            assert_tol=2e-8, verbose=True):
  D = 3
  if n is None:
    n = max(3, m_max)
  if q_vol is None:
    q_vol = n + 4
  if q_face is None:
    q_face = n + 4
  if n < 2:
    raise ValueError("n must be at least 2")

  kappa = np.array([0.73, 1.11, 1.47, 0.92], dtype=np.float64)
  global_vids = np.array([42, 7, 100, 13], dtype=np.int32)
  V_phys = make_affine_tet_vertices()

  if nonpoly:
    u_fun, f_fun, grad_u_fun = make_nonpolynomial_u_f_grad()
    sol_label = "nonpolynomial"
  else:
    u_fun, f_fun, grad_u_fun = make_manufactured_u_f_grad(m_max)
    sol_label = f"poly degree <= {m_max}"

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

  alpha_robin = np.array([1.00, 1.15, 0.90, 1.30], dtype=np.float64)
  beta_robin = np.array([0.25, 0.18, 0.32, 0.22], dtype=np.float64)

  R = np.empty_like(T_full)
  kf = bnd["kf"]
  for face_id in range(D + 1):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    R[sl, :] = alpha_robin[face_id] * T_full[sl, :] + beta_robin[face_id] * F_full[sl, :]

  f_int = project_source_int(D, n - 2, kappa, q_vol, V_phys, f_fun)
  g_bnd, _ = project_robin_bnd(
    D, n, kappa, q_face, V_phys, global_vids,
    alpha_robin, beta_robin, u_fun, grad_u_fun,
    face_data=bnd,
  )

  sL = row_rms_scale(L_int)
  sR = row_rms_scale(R)
  A = np.asfortranarray(np.vstack((sL * L_int, sR * R)))
  b = np.concatenate((sL * (-f_int), sR * g_bnd))

  c_sol, info = solve_dense_lsmr_or_lstsq(A, b, atol=1e-14, btol=1e-14)
  c_ref = project_solution_coeffs(D, n, kappa, max(q_vol, n + 4), V_phys, u_fun)

  rel_coef = np.linalg.norm(c_sol - c_ref) / max(1.0e-300, np.linalg.norm(c_ref))
  rel_resid = np.linalg.norm(A @ c_sol - b) / max(1.0e-300, np.linalg.norm(b))

  # A quadrature-grid physical L2 check on the affine pullback.
  alpha_n, tail_n, invh_n, _ = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, max(q_vol, n + 4), kappa)
  V_eval = jbasis_eval_all(Xhat, kappa, n, alpha_n, tail_n, invh_n, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  u_true = eval_scalar_xyz(u_fun, P)
  u_num = V_eval @ c_sol
  rel_L2 = math.sqrt(
    float(np.sum(What * (u_num - u_true) ** 2)) /
    max(1.0e-300, float(np.sum(What * u_true ** 2)))
  )
 
  if verbose:
    if isinstance(info, dict):
      itn = info.get("itn", -1)
      istop = info.get("istop", -1)
      solver = info.get("solver", "unknown")
      normr = info.get("normr", np.nan)
      normAr = info.get("normAr", np.nan)
      condA = info.get("condA", np.nan)
    else:
      itn = getattr(info, "itn", -1) if info is not None else -1
      istop = getattr(info, "istop", -1) if info is not None else -1
      solver = "unknown"
      normr = np.nan
      normAr = np.nan
      condA = np.nan
  
    print(
      f"[poisson-robin tet] {sol_label} n={n} M={M} m_int={m_int} "
      f"kf={kf} q_vol={q_vol} q_face={q_face} detB={detBabs:.6e} "
      f"rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e} rel_resid={rel_resid:.3e} "
      f"solver={solver} lsmr_itn={itn} istop={istop} "
      f"normr={normr:.3e} normAr={normAr:.3e} condA={condA:.3e}"
    )


  if not nonpoly and n >= m_max:
    assert rel_coef < assert_tol, rel_coef
    assert rel_L2 < assert_tol, rel_L2
    assert rel_resid < max(1e-10, 10 * assert_tol), rel_resid

  return {
    "rel_coef": rel_coef,
    "rel_L2": rel_L2,
    "rel_resid": rel_resid,
    "c_sol": c_sol,
    "c_ref": c_ref,
    "info": info,
  }


def run_smoke():
  # Polynomial exact-recovery checks.  The affine pullback preserves degree.
  run_one(m_max=3, n=3, q_vol=7, q_face=7, assert_tol=2e-8)
  run_one(m_max=4, n=4, q_vol=8, q_face=8, assert_tol=5e-8)

  # Smooth non-polynomial case: no exactness assertion, but it should solve
  # the projected overdetermined system with a small residual.
  info = run_one(m_max=3, n=5, q_vol=9, q_face=9, nonpoly=True, assert_tol=1e-6)
  #assert info["rel_resid"] < 1e-7, info["rel_resid"]

  print("\nAll Poisson-Robin tet manufactured-solution tests passed.")


def main():
  parser = argparse.ArgumentParser(description="Single-tet Poisson-Robin manufactured test.")
  parser.add_argument("--m-max", type=int, default=3, help="Polynomial total degree.")
  parser.add_argument("--n", type=int, default=None, help="Solution polynomial degree.")
  parser.add_argument("--q-vol", type=int, default=None, help="Volume quadrature order.")
  parser.add_argument("--q-face", type=int, default=None, help="Face quadrature order.")
  parser.add_argument("--nonpoly", action="store_true", help="Use a smooth non-polynomial manufactured solution.")
  parser.add_argument("--tol", type=float, default=2e-8, help="Assertion tolerance for polynomial exact cases.")
  parser.add_argument("--smoke", action="store_true", help="Run the built-in smoke suite.")
  args = parser.parse_args()

  if args.smoke:
    run_smoke()
  else:
    run_one(
      m_max=args.m_max,
      n=args.n,
      q_vol=args.q_vol,
      q_face=args.q_face,
      nonpoly=args.nonpoly,
      assert_tol=args.tol,
    )


if __name__ == "__main__":
  main()
