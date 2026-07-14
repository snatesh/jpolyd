"""
Source and boundary-data projection helpers for local simplex PDE tests.

This module intentionally stays Python-level for now. It assembles RHS vectors
using the already-tested jpolyd basis/quadrature/geometry/permutation wrappers.

Conventions
-----------
Volume source vector for Poisson:

    f_int[i] = |det B| int_{Khat} f_phys(F(xhat)) phi_i(xhat) w_kappa dxhat

where f = -Delta u for the PDE Delta u = -f.

Robin boundary vector:

    g_bnd[f*kf+j] = int_{S_f} psi_j (alpha_f u + beta_f n_f.grad u) w_f dS

using the common canonical face basis convention.
"""

import math
import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jquad_tprod import jquad_mapped_build_kappa
from jgeom import affine_from_verts
from jperms import face_vertices, face_sigma_array, common_face_kappa


def build_basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def affine_map_ref_to_phys(V_phys, Xhat):
  """Map reference coordinates xhat=(lambda_1,...,lambda_D) to physical points."""
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  Xhat = np.asarray(Xhat, dtype=np.float64)
  D = V_phys.shape[0]
  if V_phys.shape != (D, D + 1):
    raise ValueError("V_phys must have shape (D,D+1)")
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def eval_scalar_xyz(fun, P):
  """Evaluate a scalar callable fun(x,y,z) on P with shape (nq,3)."""
  P = np.asarray(P, dtype=np.float64)
  if P.ndim != 2 or P.shape[1] != 3:
    raise ValueError("eval_scalar_xyz expects P with shape (nq,3)")
  vals = fun(P[:, 0], P[:, 1], P[:, 2])
  vals = np.asarray(vals, dtype=np.float64)
  if vals.shape == ():
    vals = np.full(P.shape[0], float(vals), dtype=np.float64)
  return np.ravel(vals).astype(np.float64, copy=False)


def eval_grad_xyz(grad_fun, P):
  """Evaluate grad_fun(x,y,z)->(ux,uy,uz) on P and return shape (nq,3)."""
  P = np.asarray(P, dtype=np.float64)
  if P.ndim != 2 or P.shape[1] != 3:
    raise ValueError("eval_grad_xyz expects P with shape (nq,3)")
  gx, gy, gz = grad_fun(P[:, 0], P[:, 1], P[:, 2])
  out = np.empty((P.shape[0], 3), dtype=np.float64)
  for j, g in enumerate((gx, gy, gz)):
    g = np.asarray(g, dtype=np.float64)
    if g.shape == ():
      out[:, j] = float(g)
    else:
      out[:, j] = np.ravel(g)
  return out


def project_source_int(D, n_int, kappa, q_vol, V_phys, f_fun):
  """
  Project physical source f into the interior test space Pi_{n_int}^D.

  This returns physical moments:

      |det B| int_{Khat} f(F(xhat)) phi_i(xhat) w_kappa dxhat.

  For Poisson Delta u = -f, the interior RHS used with L_int is -f_int.
  """
  D = int(D)
  if D != 3:
    raise ValueError("project_source_int currently expects D=3 callables f(x,y,z)")

  alpha, tail, invh, _ = build_basis_structs(D, n_int, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n_int, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  vals = eval_scalar_xyz(f_fun, P)
  detBabs = float(affine_from_verts(V_phys)["detBabs"])
  return detBabs * (V.T @ (What * vals))


def canonical_face_bary(Y):
  """
  Convert intrinsic canonical face coordinates to face barycentric coordinates.

  For a codimension-one face of a D-dimensional volume simplex, Y has shape
  (nq,D-1). The returned array has shape (nq,D) in canonical face vertex order:

      [1 - sum_j Y_j, Y_1, ..., Y_{D-1}].
  """
  Y = np.asarray(Y, dtype=np.float64)
  nq, d = Y.shape
  B = np.empty((nq, d + 1), dtype=np.float64)
  B[:, 0] = 1.0 - np.sum(Y, axis=1)
  B[:, 1:] = Y
  return B


def face_points_in_volume_ref(D, face_id, sigma_local_to_canonical, Y):
  """
  Embed canonical face quadrature points into reference volume coordinates.

  The jperms convention used here is

      sigma[local_face_pos] = canonical_face_pos.

  The omitted/opposite volume vertex gets barycentric coordinate zero. The
  returned Xhat coordinates are (lambda_1,...,lambda_D).
  """
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
  """
  Compute physical face scales, outward unit normals, and scaled normals.

  normal_scaled[f,:] = face_scale[f] * outward_unit_normal[f].
  """
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  global_vids = np.asarray(global_vids, dtype=np.int32)
  if V_phys.shape != (D, D + 1):
    raise ValueError(f"V_phys must have shape ({D},{D+1})")
  if global_vids.shape != (D + 1,):
    raise ValueError(f"global_vids must have shape ({D+1},)")

  nface = D + 1
  face_scale = np.empty(nface, dtype=np.float64)
  unit_normal = np.empty((nface, D), dtype=np.float64)
  normal_scaled = np.empty((nface, D), dtype=np.float64)

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


def build_common_face_basis(D, n_face, kappa_vol, q_face):
  """Build common canonical face quadrature/basis data for D>=2."""
  kappa_face = common_face_kappa(kappa_vol)
  alpha_face, tail_face, invh_face, kf = build_basis_structs(D - 1, n_face, kappa_face)
  Y, W_face = jquad_mapped_build_kappa(D - 1, q_face, kappa_face)
  Vt_common = jbasis_eval_all(
    Y, kappa_face, n_face, alpha_face, tail_face, invh_face, D - 1
  )
  return {
    "kappa_face": kappa_face,
    "Y": Y,
    "W_face": np.ascontiguousarray(W_face),
    "Vt_common": np.asfortranarray(Vt_common),
    "kf": kf,
  }


def project_robin_bnd(D, n_face, kappa_vol, q_face, V_phys, global_vids,
                      alpha_robin, beta_robin, u_fun, grad_u_fun,
                      face_data=None):
  """
  Project Robin boundary data into the common canonical face basis.

  Returns g_bnd with stacked face layout, where face f occupies rows
  f*kf:(f+1)*kf:

      g_fj = int_{S_f} psi_j (alpha_f u + beta_f n_f.grad u) w_f dS.

  Parameters alpha_robin and beta_robin are facewise constants with shape (D+1,).
  """
  D = int(D)
  if D != 3:
    raise ValueError("project_robin_bnd currently expects D=3 callables u(x,y,z)")

  global_vids = np.asarray(global_vids, dtype=np.int32)
  alpha_robin = np.asarray(alpha_robin, dtype=np.float64)
  beta_robin = np.asarray(beta_robin, dtype=np.float64)
  nface = D + 1
  if alpha_robin.shape != (nface,) or beta_robin.shape != (nface,):
    raise ValueError(f"alpha_robin and beta_robin must have shape ({nface},)")

  if face_data is None:
    face_data = build_common_face_basis(D, n_face, kappa_vol, q_face)

  Y = np.asarray(face_data["Y"], dtype=np.float64)
  W_face = np.asarray(face_data["W_face"], dtype=np.float64)
  Vt_common = np.asarray(face_data["Vt_common"], dtype=np.float64)
  kf = int(face_data["kf"])

  if "face_scale" in face_data and "unit_normal" in face_data:
    face_scale = np.asarray(face_data["face_scale"], dtype=np.float64)
    unit_normal = np.asarray(face_data["unit_normal"], dtype=np.float64)
  else:
    face_scale, unit_normal, _ = physical_face_geometry(D, V_phys, global_vids)

  sigma_arrays = face_data.get("sigma_arrays", None)
  if sigma_arrays is None:
    sigma_arrays = [face_sigma_array(global_vids, f).astype(np.int32) for f in range(nface)]

  g = np.empty(nface * kf, dtype=np.float64)

  for face_id in range(nface):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    Xhat, _ = face_points_in_volume_ref(D, face_id, sigma_arrays[face_id], Y)
    P = affine_map_ref_to_phys(V_phys, Xhat)

    u_vals = eval_scalar_xyz(u_fun, P)
    grad_vals = eval_grad_xyz(grad_u_fun, P)
    q_vals = grad_vals @ unit_normal[face_id, :]
    robin_vals = alpha_robin[face_id] * u_vals + beta_robin[face_id] * q_vals

    g[sl] = Vt_common.T @ ((W_face * face_scale[face_id]) * robin_vals)

  return g, {
    "face_scale": face_scale,
    "unit_normal": unit_normal,
    "sigma_arrays": sigma_arrays,
    "Y": Y,
    "W_face": W_face,
    "Vt_common": Vt_common,
    "kf": kf,
  }


def make_sym_u_f_and_grad(u_expr, simplify=True):
  """
  Build numpy-callable (u, f=-Delta u, grad_u) from a SymPy expression u_expr(x,y,z).
  """
  import sympy as sp

  x, y, z = sp.symbols("x y z", real=True)
  u = u_expr

  ux = sp.diff(u, x)
  uy = sp.diff(u, y)
  uz = sp.diff(u, z)

  lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2)
  f = -lap_u

  if simplify:
    ux = sp.simplify(ux)
    uy = sp.simplify(uy)
    uz = sp.simplify(uz)
    f = sp.simplify(f)

  u_fun = sp.lambdify((x, y, z), u, "numpy")
  f_fun = sp.lambdify((x, y, z), f, "numpy")
  ux_fun = sp.lambdify((x, y, z), ux, "numpy")
  uy_fun = sp.lambdify((x, y, z), uy, "numpy")
  uz_fun = sp.lambdify((x, y, z), uz, "numpy")

  def grad_u_fun(xv, yv, zv):
    return ux_fun(xv, yv, zv), uy_fun(xv, yv, zv), uz_fun(xv, yv, zv)

  return u_fun, f_fun, grad_u_fun


def make_manufactured_u_f_grad(m_max):
  """Total-degree polynomial manufactured solution in x,y,z."""
  import sympy as sp

  x, y, z = sp.symbols("x y z", real=True)
  u_expr = 0
  for a in range(m_max + 1):
    for b in range(m_max + 1 - a):
      for c in range(m_max + 1 - a - b):
        coef = sp.Rational(1, 1 + a + b + c)
        u_expr += coef * (x ** a) * (y ** b) * (z ** c)

  return make_sym_u_f_and_grad(u_expr, simplify=True)


def make_nonpolynomial_u_f_grad():
  """Smooth non-polynomial manufactured solution for convergence experiments."""
  import sympy as sp

  x, y, z = sp.symbols("x y z", real=True)
  u_expr = sp.exp(sp.cos(0.35 * x ** 2 + 0.21 * y ** 2 + 0.17 * z ** 2 + 0.11 * x * y))
  return make_sym_u_f_and_grad(u_expr, simplify=True)
