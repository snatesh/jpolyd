#!/usr/bin/env python3
"""
Non-polynomial Poisson--Robin convergence sweep using the RefSimplexPrecomp path.

This driver intentionally does not import/use the old Python-side operator paths:

  jlaplace.assemble_L_int
  jtrace.assemble_T_full_common
  jflux.assemble_F_full_common
  jdmat / jkmat second-partial construction

Instead, it consumes C++ precomputed reference blocks from jprecomp:

  Lij_ref, T_ref, Fgrad_ref, Mface_ref, volume quadrature, volume basis

and applies affine geometry/permutation data in Python.  Boundary RHS projection
also uses face quadrature/basis exposed by jprecomp; this driver no longer
imports or uses the old jbasis/jquad/jdmat/jkmat/jlaplace/jtrace/jflux paths.

Run examples:

  python testing/jpoisson_robin_nonpoly_convergence_precomp.py --D all --n-min 2 --n-max 7
  python testing/jpoisson_robin_nonpoly_convergence_precomp.py --D 4 --n-min 2 --n-max 6 --q-pad 2
  python testing/jpoisson_robin_nonpoly_convergence_precomp.py --D 3 --solver lstsq
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from jgeom import affine_from_verts
from jperms import face_vertices, face_sigma_array, face_sigma_index
from jprecomp import RefSimplexPrecomp

try:
  from jlsmr import lsmr_dense_solve
except Exception:
  lsmr_dense_solve = None


def dimPi(D, n):
  return math.comb(n + D, D) if n >= 0 else 0


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
  """Physical face scales and outward normals, using C-side face/permutation helpers."""
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
    unit_normal[0, 0] = sgn
    unit_normal[1, 0] = -sgn
    face_scale[:] = 1.0
    normal_scaled[:, :] = unit_normal
    return face_scale, unit_normal, np.ascontiguousarray(normal_scaled)

  for face_id in range(nface):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    # Canonicalize the physical face ordering using the same local->canonical
    # sigma used by precomputed face blocks.
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


def get_face_quad_basis(pc):
  """Return canonical common-face quadrature and basis from RefSimplexPrecomp."""
  if hasattr(pc, "face_quad_basis"):
    Y, W_face, V_face = pc.face_quad_basis()
  else:
    Y, W_face = pc.face_quad()
    V_face = pc.face_basis()

  return (
    np.asarray(Y, dtype=np.float64),
    np.asarray(W_face, dtype=np.float64),
    np.asfortranarray(V_face, dtype=np.float64),
  )


def assemble_L_int_from_precomp(pc, G, detBabs):
  Lij_ref = pc.Lij_ref()
  L_full = np.zeros((pc.M, pc.M), dtype=np.float64, order="F")
  for i in range(pc.D):
    for j in range(pc.D):
      L_full += G[i, j] * Lij_ref[:, :, i, j]
  return np.asfortranarray(detBabs * L_full[:pc.m_int, :])


def assemble_boundary_from_precomp(pc, geom, V_phys, global_vids):
  D = pc.D
  M = pc.M
  kf = pc.kf
  nface = pc.nface

  T_ref = pc.T_ref()
  Fgrad_ref = pc.Fgrad_ref()
  face_ref_scale = pc.face_ref_scale()

  BinvT = np.asarray(geom["BinvT"], dtype=np.float64)
  face_scale, unit_normal, normal_scaled = physical_face_geometry(D, V_phys, global_vids)

  T_full = np.empty((nface * kf, M), dtype=np.float64, order="F")
  F_full = np.empty((nface * kf, M), dtype=np.float64, order="F")
  sigma_indices = np.empty(nface, dtype=np.int32)

  for face_id in range(nface):
    sigma_idx = int(face_sigma_index(global_vids, face_id))
    sigma_indices[face_id] = sigma_idx
    sl = slice(face_id * kf, (face_id + 1) * kf)

    ref_scale = float(face_ref_scale[face_id])
    scale_ratio = float(face_scale[face_id]) / ref_scale

    T_full[sl, :] = scale_ratio * T_ref[:, :, sigma_idx, face_id]

    # Fgrad_ref already contains reference-face scale.  Physical flux uses
    # normal_scaled = physical_face_scale * outward_unit_normal, hence the
    # coefficient is B^{-1}(normal_scaled / reference_face_scale).
    eta = BinvT.T @ (normal_scaled[face_id, :] / ref_scale)
    F_block = np.zeros((kf, M), dtype=np.float64)
    for a in range(D):
      F_block += eta[a] * Fgrad_ref[:, :, a, sigma_idx, face_id]
    F_full[sl, :] = F_block

  return {
    "T_full": np.asfortranarray(T_full),
    "F_full": np.asfortranarray(F_full),
    "face_scale": face_scale,
    "unit_normal": unit_normal,
    "normal_scaled": normal_scaled,
    "face_ref_scale": face_ref_scale,
    "face_sigma_index": sigma_indices,
  }


def project_source_int_from_precomp(pc, V_phys, f_fun, detBabs):
  Xhat, What = pc.residual_quad()
  V = pc.residual_basis()
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_D(f_fun, P)
  V_int = V[:, :pc.m_int]
  return detBabs * (V_int.T @ (What * vals))


def project_solution_coeffs_from_precomp(pc_eval, V_phys, u_fun):
  Xhat, What = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_D(u_fun, P)
  return V.T @ (What * vals)


def project_robin_bnd_from_precomp_face_data(
  pc,
  V_phys,
  global_vids,
  alpha_robin,
  beta_robin,
  u_fun,
  grad_u_fun,
  boundary_data,
):
  D = pc.D
  alpha_robin = np.asarray(alpha_robin, dtype=np.float64)
  beta_robin = np.asarray(beta_robin, dtype=np.float64)
  nface = D + 1
  if alpha_robin.shape != (nface,) or beta_robin.shape != (nface,):
    raise ValueError(f"alpha_robin and beta_robin must have shape ({nface},)")

  Y, W_face, V_face = get_face_quad_basis(pc)
  kf = V_face.shape[1]
  face_scale = np.asarray(boundary_data["face_scale"], dtype=np.float64)
  unit_normal = np.asarray(boundary_data["unit_normal"], dtype=np.float64)

  g = np.empty(nface * kf, dtype=np.float64)

  for face_id in range(nface):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int32)
    Xhat, _ = face_points_in_volume_ref(D, face_id, sigma, Y)
    P = affine_map_ref_to_phys(V_phys, Xhat)

    u_vals = eval_scalar_D(u_fun, P)
    grad_vals = eval_grad_D(grad_u_fun, P)
    q_vals = grad_vals @ unit_normal[face_id, :]
    robin_vals = alpha_robin[face_id] * u_vals + beta_robin[face_id] * q_vals

    g[sl] = V_face.T @ ((W_face * face_scale[face_id]) * robin_vals)

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

  pc = RefSimplexPrecomp(D, n, kappa, q_pad=2, q_vol=q_vol, q_face=q_face)
  L_int = assemble_L_int_from_precomp(pc, G, detBabs)
  bnd = assemble_boundary_from_precomp(pc, geom, V_phys, global_vids)
  T_full = bnd["T_full"]
  F_full = bnd["F_full"]

  nface = D + 1
  alpha_robin = np.array([1.0 + 0.07 * f for f in range(nface)], dtype=np.float64)
  beta_robin = np.array([0.18 + 0.025 * ((2 * f + 1) % 5) for f in range(nface)], dtype=np.float64)

  R = np.empty_like(T_full)
  for face_id in range(nface):
    sl = slice(face_id * pc.kf, (face_id + 1) * pc.kf)
    R[sl, :] = alpha_robin[face_id] * T_full[sl, :] + beta_robin[face_id] * F_full[sl, :]

  f_int = project_source_int_from_precomp(pc, V_phys, f_fun, detBabs)
  g_bnd = project_robin_bnd_from_precomp_face_data(
    pc,
    V_phys,
    global_vids,
    alpha_robin,
    beta_robin,
    u_fun,
    grad_u_fun,
    bnd,
  )

  sL = row_rms_scale(L_int)
  sR = row_rms_scale(R)
  A = np.asfortranarray(np.vstack((sL * L_int, sR * R)))
  b = np.concatenate((sL * (-f_int), sR * g_bnd))

  c_sol, info = solve_system(A, b, solver=solver, nout=nout, atol=atol, btol=btol)

  # Evaluation/projection quadrature is also obtained through jprecomp.  This
  # builds another reference precompute at q_eval; it avoids direct jquad/jbasis
  # calls in this driver for the volume projection/evaluation path.
  q_face_eval = 1 if D == 1 else q_eval
  pc_eval = RefSimplexPrecomp(D, n, kappa, q_pad=2, q_vol=q_eval, q_face=q_face_eval)
  c_ref = project_solution_coeffs_from_precomp(pc_eval, V_phys, u_fun)

  rel_coef = np.linalg.norm(c_sol - c_ref) / max(1.0e-300, np.linalg.norm(c_ref))
  rel_resid = np.linalg.norm(A @ c_sol - b) / max(1.0e-300, np.linalg.norm(b))

  Xhat, What = pc_eval.volume_quad()
  V_eval = pc_eval.volume_basis()
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
    "M": pc.M,
    "m_int": pc.m_int,
    "kf": pc.kf,
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
  plt.title(f"Poisson--Robin non-polynomial convergence, precomp path ({metric})")
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=200)


def main():
  parser = argparse.ArgumentParser(description="Precomp-path non-polynomial Poisson--Robin convergence sweep for D=1..4.")
  parser.add_argument("--D", default="all", help="Dimension list, e.g. all or 1,2,3,4")
  parser.add_argument("--n-min", type=int, default=2)
  parser.add_argument("--n-max", type=int, default=6)
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
    out_prefix = os.path.join(this_dir, "jpoisson_robin_nonpoly_convergence_precomp")

  csv_path = out_prefix + ".csv"
  save_csv(rows, csv_path)
  print(f"\nSaved CSV to: {csv_path}")

  if not args.no_plot:
    png_path = out_prefix + f"_{args.metric}.png"
    save_plot(rows, png_path, args.metric)
    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
  main()
