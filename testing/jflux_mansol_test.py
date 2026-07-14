import itertools
import math
import numpy as np

from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)
from jquad_tprod import jquad_mapped_build_kappa
from jgeom import affine_from_verts
from jperms import (
  face_vertices,
  face_sigma_array,
  face_sigma_index,
  perm_to_lehmer_index,
  common_face_kappa,
)
from jflux import assemble_F_full_common


def build_basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def manufactured_u_phys(P):
  """
  Degree <= 3 non-symmetric manufactured function on physical coordinates.
  The affine pullback to the reference simplex is still degree <= 3.
  """
  P = np.asarray(P, dtype=np.float64)
  nq, D = P.shape
  u = 0.73 * np.ones(nq, dtype=np.float64)

  for a in range(D):
    x = P[:, a]
    u += (0.19 + 0.07 * (a + 1)) * x
    u += ((-1.0) ** a) * (0.041 + 0.013 * a) * x * x
    u += (0.004 + 0.002 * (a + 1)) * x * x * x

  for a in range(D):
    for b in range(a + 1, D):
      xa = P[:, a]
      xb = P[:, b]
      u += (0.018 + 0.006 * (a + 2 * b)) * xa * xb
      u += (0.003 + 0.0015 * b) * xa * xb * xb
      u -= (0.002 + 0.001 * a) * xa * xa * xb

  return u


def grad_manufactured_u_phys(P):
  """Analytic physical gradient of manufactured_u_phys."""
  P = np.asarray(P, dtype=np.float64)
  nq, D = P.shape
  G = np.zeros((nq, D), dtype=np.float64)

  for a in range(D):
    x = P[:, a]
    G[:, a] += (0.19 + 0.07 * (a + 1))
    G[:, a] += 2.0 * ((-1.0) ** a) * (0.041 + 0.013 * a) * x
    G[:, a] += 3.0 * (0.004 + 0.002 * (a + 1)) * x * x

  for a in range(D):
    for b in range(a + 1, D):
      xa = P[:, a]
      xb = P[:, b]
      cab = 0.018 + 0.006 * (a + 2 * b)
      db = 0.003 + 0.0015 * b
      ea = 0.002 + 0.001 * a

      # cab * xa * xb
      G[:, a] += cab * xb
      G[:, b] += cab * xa

      # db * xa * xb^2
      G[:, a] += db * xb * xb
      G[:, b] += 2.0 * db * xa * xb

      # -ea * xa^2 * xb
      G[:, a] -= 2.0 * ea * xa * xb
      G[:, b] -= ea * xa * xa

  return G


def affine_map_ref_to_phys(V_phys, Xhat):
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  Xhat = np.asarray(Xhat, dtype=np.float64)
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def project_physical_function_to_ref_coeffs(D, n, kappa, q_vol, V_phys, u_phys):
  """
  Project u_phys onto the element basis through its affine pullback.

  We assemble the physical mass matrix and physical RHS explicitly:
      M_phys = |det B| V^T W V,
      b_phys = |det B| V^T W u_phys(F(xhat)),
  then solve for coefficients. For an affine element, |det B| cancels, but
  keeping it here makes the physical scaling explicit in the test.
  """
  alpha, tail, invh, M = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  uhat = u_phys(P)

  geom = affine_from_verts(V_phys)
  detBabs = float(geom["detBabs"])

  WV = (detBabs * What)[:, None] * V
  M_phys = V.T @ WV
  b_phys = V.T @ ((detBabs * What) * uhat)
  c = np.linalg.solve(M_phys, b_phys)

  return c, {
    "Xhat": Xhat,
    "What": What,
    "P": P,
    "V": V,
    "M": M,
    "detBabs": detBabs,
    "M_phys": M_phys,
  }


def canonical_face_bary(Y):
  """
  Convert intrinsic canonical face coordinates to face barycentric coords.

  For a codimension-one face of a D-dimensional volume simplex, Y has shape
  (nq,D-1). The returned array has shape (nq,D) and is ordered by canonical
  face vertex order:
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
  Embed canonical face quadrature points into the reference volume simplex.

  Current jperms convention used here:
      sigma[i_local_face_pos] = i_canonical_face_pos.

  The omitted/opposite volume vertex gets barycentric coordinate zero. The
  returned reference coordinates are xhat=(lambda_1,...,lambda_D).
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


def physical_face_geometry(D, V_phys, global_vids):
  """
  Compute physical face scales, outward unit normals, and scaled normals.

  normal_scaled[f,:] = face_scale[f] * n_out[f].

  This is the vector that appears directly in the flux integral relative to
  canonical face coordinates:
      (n_out · grad u) dS_phys = (normal_scaled · grad u) dY.
  """
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  if V_phys.shape != (D, D + 1):
    raise ValueError(f"V_phys must have shape ({D},{D+1})")

  nface = D + 1
  face_scale = np.empty(nface, dtype=np.float64)
  unit_normal = np.empty((nface, D), dtype=np.float64)
  normal_scaled = np.empty((nface, D), dtype=np.float64)

  for face_id in range(nface):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    # P_can[:,i] is the physical vertex in canonical face position i.
    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      i_can = int(sigma[i_local])
      P_can[:, i_can] = V_phys[:, fv[i_local]]

    E = P_can[:, 1:] - P_can[:, [0]]
    G = E.T @ E
    s = math.sqrt(float(np.linalg.det(G)))

    # Null vector of E.T gives a unit normal to the face.
    _, _, vh = np.linalg.svd(E.T, full_matrices=True)
    n = vh[-1, :].copy()
    n /= np.linalg.norm(n)

    # Orient outward: away from the opposite vertex.
    p_opp = V_phys[:, face_id]
    to_opp = p_opp - P_can[:, 0]
    if float(np.dot(n, to_opp)) > 0.0:
      n *= -1.0

    face_scale[face_id] = s
    unit_normal[face_id, :] = n
    normal_scaled[face_id, :] = s * n

  return face_scale, unit_normal, np.ascontiguousarray(normal_scaled)


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


def build_flux_inputs(D, n, kappa_vol, q_face, global_vids):
  """Build real jpolyd arrays consumed by the C-backed jflux assembler."""
  kappa_face = common_face_kappa(kappa_vol)

  alpha_vol, tail_vol, invh_vol, M = build_basis_structs(D, n, kappa_vol)
  alpha_face, tail_face, invh_face, kf = build_basis_structs(D - 1, n, kappa_face)

  Y, W_face = jquad_mapped_build_kappa(D - 1, q_face, kappa_face)
  Vt_common = jbasis_eval_all(
    Y, kappa_face, n, alpha_face, tail_face, invh_face, D - 1
  )
  Vt_common = np.asfortranarray(Vt_common)
  W_face = np.ascontiguousarray(W_face)

  sigmas = all_sigma_tuples(D)
  nsigma = len(sigmas)
  nface = D + 1
  nq = Y.shape[0]

  dVv_hat_sigma_face = np.empty((nq, M, D, nsigma, nface), dtype=np.float64, order="F")
  Vv_sigma_face = np.empty((nq, M, nsigma, nface), dtype=np.float64, order="F")

  for face_id in range(nface):
    for idx, sigma in enumerate(sigmas):
      Xhat_f, _ = face_points_in_volume_ref(D, face_id, sigma, Y)
      Vv, dVv = eval_basis_with_grad(
        Xhat_f, kappa_vol, n, alpha_vol, tail_vol, invh_vol, D
      )
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

  return {
    "kappa_face": kappa_face,
    "Y": Y,
    "W_face": W_face,
    "Vt_common": Vt_common,
    "Vv_sigma_face": Vv_sigma_face,
    "dVv_hat_sigma_face": dVv_hat_sigma_face,
    "face_sigma_index": face_sigma,
    "sigma_arrays": sigma_arrays,
    "M": M,
    "kf": kf,
    "nq": nq,
  }


def direct_physical_flux_moments(D, V_phys, flux_data, normal_scaled, grad_u_phys):
  """
  Independent physical-space reference for F_full*c.

  For each face f:
      mu_f[j] = int psi_j(Y) * (n_out · grad u_phys)(x_phys(Y))
                      * w_face(Y) dS_phys
              = Vt.T @ (W_face * (normal_scaled[f] · grad u_phys(x_phys(Y))))

  This does not use the assembled F matrix or modal coefficient vector.
  """
  Y = flux_data["Y"]
  W_face = flux_data["W_face"]
  Vt = flux_data["Vt_common"]
  kf = flux_data["kf"]

  mu_ref = np.empty(((D + 1) * kf,), dtype=np.float64)
  for face_id in range(D + 1):
    sigma = flux_data["sigma_arrays"][face_id]
    Xhat_f, _ = face_points_in_volume_ref(D, face_id, sigma, Y)
    P_face = affine_map_ref_to_phys(V_phys, Xhat_f)
    grad = grad_u_phys(P_face)
    ndS = grad @ normal_scaled[face_id, :]

    sl = slice(face_id * kf, (face_id + 1) * kf)
    mu_ref[sl] = Vt.T @ (W_face * ndS)

  return mu_ref


def make_physical_simplex(D):
  """Construct a non-reference affine simplex with nontrivial face geometry."""
  V_phys = np.zeros((D, D + 1), dtype=np.float64, order="F")
  V_phys[:, 0] = np.linspace(0.15, -0.20, D)
  for j in range(1, D + 1):
    V_phys[:, j] = V_phys[:, 0]
    V_phys[j - 1, j] += 1.0 + 0.10 * j
    V_phys[:, j] += 0.025 * j * np.arange(1, D + 1)
  return V_phys


def run_one(D, n, q_vol, q_face, verbose=True):
  if D < 2:
    raise ValueError("this flux manufactured test starts at D=2")

  # Cyclic storage [kappa_1,...,kappa_D,kappa_0].
  base = np.array([0.8, 1.7, 2.3, 1.1, 0.6, 1.4], dtype=np.float64)
  kappa_vol = np.ascontiguousarray(base[:D + 1])

  # Deliberately unsorted global IDs so face orientations are nontrivial.
  gids_base = np.array([42, 7, 100, 13, 55, 31], dtype=np.int32)
  global_vids = np.ascontiguousarray(gids_base[:D + 1])

  V_phys = make_physical_simplex(D)
  geom = affine_from_verts(V_phys)
  BinvT = np.asfortranarray(geom["BinvT"])
  face_scale, unit_normal, normal_scaled = physical_face_geometry(D, V_phys, global_vids)

  # Project physical u onto the reference modal expansion via pullback.
  c_u, vol_data = project_physical_function_to_ref_coeffs(
    D, n, kappa_vol, q_vol, V_phys, manufactured_u_phys
  )

  flux_data = build_flux_inputs(D, n, kappa_vol, q_face, global_vids)
  assert flux_data["M"] == vol_data["M"]

  F = assemble_F_full_common(
    D,
    flux_data["M"],
    flux_data["kf"],
    flux_data["nq"],
    flux_data["face_sigma_index"],
    normal_scaled,
    BinvT,
    flux_data["Vt_common"],
    flux_data["W_face"],
    flux_data["dVv_hat_sigma_face"],
  )

  mu_from_F = F @ c_u
  mu_ref = direct_physical_flux_moments(
    D, V_phys, flux_data, normal_scaled, grad_manufactured_u_phys
  )

  rel_mom = np.linalg.norm(mu_from_F - mu_ref) / max(1e-300, np.linalg.norm(mu_ref))

  # Also reconstruct the scalar normal derivative on each face by dividing the
  # physical flux moments by the constant face scale.
  Y = flux_data["Y"]
  W = flux_data["W_face"]
  Vt = flux_data["Vt_common"]
  kf = flux_data["kf"]
  max_rel_face_L2 = 0.0
  for face_id in range(D + 1):
    sl = slice(face_id * kf, (face_id + 1) * kf)
    q_coeff = mu_from_F[sl] / face_scale[face_id]
    q_h = Vt @ q_coeff

    sigma = flux_data["sigma_arrays"][face_id]
    Xhat_f, _ = face_points_in_volume_ref(D, face_id, sigma, Y)
    P_face = affine_map_ref_to_phys(V_phys, Xhat_f)
    q_exact = grad_manufactured_u_phys(P_face) @ unit_normal[face_id, :]

    num = np.sum(W * (q_h - q_exact) ** 2)
    den = np.sum(W * q_exact ** 2)
    rel = math.sqrt(num / max(1e-300, den))
    max_rel_face_L2 = max(max_rel_face_L2, rel)

  if verbose:
    print(
      f"[manufactured flux] D={D} n={n} M={flux_data['M']} "
      f"kf={flux_data['kf']} nq_face={flux_data['nq']} "
      f"detB={geom['detB']:.6e} rel_mom={rel_mom:.3e} "
      f"max_rel_face_L2={max_rel_face_L2:.3e}"
    )
    print(f"  global_vids        = {global_vids.tolist()}")
    print(f"  face_sigma_index   = {flux_data['face_sigma_index'].tolist()}")
    print(f"  face_scale         = {face_scale.tolist()}")
    print(f"  unit_normal[0]     = {unit_normal[0, :].tolist()}")

  if rel_mom > 5e-10 or max_rel_face_L2 > 5e-10:
    raise AssertionError(
      f"manufactured flux failed: D={D}, rel_mom={rel_mom}, "
      f"max_rel_face_L2={max_rel_face_L2}"
    )

  return rel_mom, max_rel_face_L2


def run_smoke():
  # u_phys has degree 3; n >= 4 gives comfortable room for derivative and
  # projection checks. q values are padded to make the weighted products exact
  # to roundoff in the current mapped tensor-product quadrature.
  run_one(D=2, n=5, q_vol=10, q_face=10)
  run_one(D=3, n=5, q_vol=9, q_face=9)
  run_one(D=4, n=4, q_vol=8, q_face=8)
  print("\nAll manufactured jflux tests passed.")


if __name__ == "__main__":
  run_smoke()
