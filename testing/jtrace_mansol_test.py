import itertools
import math
import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jquad_tprod import jquad_mapped_build_kappa
from jgeom import affine_from_verts
from jperms import (
  face_vertices,
  face_sigma_array,
  face_sigma_index,
  perm_to_lehmer_index,
  common_face_kappa,
)
from jtrace import assemble_T_full_common


def build_basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def manufactured_u_phys(P):
  """
  Degree <= 3 manufactured function on physical coordinates.

  Since the element map x_phys = v0 + B*xhat is affine, the pullback
  u_hat(xhat) = u_phys(v0 + B*xhat) is also degree <= 3 on the reference
  simplex. Therefore Pi_n contains it exactly for n >= 3.
  """
  P = np.asarray(P, dtype=np.float64)
  nq, D = P.shape
  u = 0.85 * np.ones(nq, dtype=np.float64)

  for a in range(D):
    pa = P[:, a]
    u += (0.21 + 0.11 * (a + 1)) * pa
    u += ((-1.0) ** a) * (0.045 + 0.018 * a) * pa * pa
    u += (0.006 + 0.003 * (a + 1)) * pa * pa * pa

  for a in range(D):
    for b in range(a + 1, D):
      u += (0.025 + 0.007 * (a + 2 * b)) * P[:, a] * P[:, b]
      u += (0.004 + 0.002 * b) * P[:, a] * P[:, b] * P[:, b]

  return u


def affine_map_ref_to_phys(V_phys, Xhat):
  """
  Map reference coordinates xhat=(lambda_1,...,lambda_D) to physical points.

  V_phys has columns [v0,v1,...,vD].  The reference coordinates are
  xhat_i = lambda_i for i=1,...,D, so

      x_phys = v0 + sum_i xhat_i * (v_i - v0).
  """
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  Xhat = np.asarray(Xhat, dtype=np.float64)
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def project_physical_function_to_ref_coeffs(D, n, kappa, q_vol, V_phys, u_phys):
  """
  Project a physical-space function onto the reference element basis.

  The physical weighted moments are

      m_i^phys = int_K u_phys(x) phi_i(F^{-1}x) w_kappa(F^{-1}x) dx
               = |det B| int_Khat u_phys(F(xhat)) phi_i(xhat) w_kappa dxhat.

  The modal expansion coefficients are m_i^phys / |det B|, because the
  physical mass matrix is also scaled by |det B| for an affine element.
  Equivalently, coefficients are the reference projection of the pullback.
  This function computes both, returns the expansion coefficients, and leaves
  the det/physical-moment scaling visible for diagnostics.
  """
  alpha, tail, invh, M = build_basis_structs(D, n, kappa)
  Xhat, What = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(Xhat, kappa, n, alpha, tail, invh, D)
  P = affine_map_ref_to_phys(V_phys, Xhat)
  uhat = u_phys(P)

  c_ref = V.T @ (What * uhat)
  geom = affine_from_verts(V_phys)
  detBabs = float(geom["detBabs"])
  physical_moments = detBabs * c_ref
  c_from_phys_moments = physical_moments / detBabs

  np.testing.assert_allclose(c_from_phys_moments, c_ref, rtol=1e-13, atol=1e-13)

  return c_ref, {
    "Xhat": Xhat,
    "What": What,
    "P": P,
    "V": V,
    "alpha": alpha,
    "tail": tail,
    "invh": invh,
    "M": M,
    "detBabs": detBabs,
    "physical_moments": physical_moments,
  }


def canonical_face_bary(Y):
  """
  Convert intrinsic canonical face coordinates to face barycentric coords.

  For a codimension-one face of a D-dimensional volume simplex, Y has shape
  (nq,D-1). The returned array has shape (nq,D) and is ordered by canonical
  face vertex order:

      [1 - sum_j Y_j, Y_1, ..., Y_{D-1}].

  These are barycentric coordinates on the face, not yet on the full volume
  simplex.
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

  Therefore, if B_can contains canonical face barycentric coordinates, the
  barycentric coordinate attached to local face position i is

      B_can[:, sigma[i]].

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

  return np.asfortranarray(lam_vol[:, 1:])


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


def compute_physical_face_scales(D, V_phys, global_vids):
  """
  Compute surface Jacobians for all physical faces.

  If y is the canonical reference-face coordinate, then

      dS_phys = face_scale[f] dS_hat.

  This scale is not canceled in T_full because T_full returns physical-face
  trace moments.
  """
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  if V_phys.shape != (D, D + 1):
    raise ValueError(f"V_phys must have shape ({D},{D+1})")

  scales = np.empty(D + 1, dtype=np.float64)
  for face_id in range(D + 1):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    # P_can[:,i] is the physical vertex in canonical face position i.
    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      i_can = int(sigma[i_local])
      P_can[:, i_can] = V_phys[:, fv[i_local]]

    E = P_can[:, 1:] - P_can[:, [0]]
    G = E.T @ E
    scales[face_id] = math.sqrt(float(np.linalg.det(G)))

  return scales


def build_trace_inputs(D, n, kappa_vol, q_face, global_vids):
  """Build real jpolyd arrays consumed by the C-backed jtrace assembler."""
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

  Vv_sigma_face = np.empty((nq, M, nsigma, nface), dtype=np.float64, order="F")
  for face_id in range(nface):
    for idx, sigma in enumerate(sigmas):
      Xf = face_points_in_volume_ref(D, face_id, sigma, Y)
      Vv_sigma_face[:, :, idx, face_id] = jbasis_eval_all(
        Xf, kappa_vol, n, alpha_vol, tail_vol, invh_vol, D
      )

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
    "face_sigma_index": face_sigma,
    "sigma_arrays": sigma_arrays,
    "M": M,
    "kf": kf,
    "nq": nq,
  }


def direct_physical_trace_moments(D, V_phys, trace_data, face_scale, u_phys):
  """
  Independent physical-space reference for T_full*c.

  For each face f:

      lambda_f[j] = int_{S_f^phys} psi_j(yhat) u_phys(x_phys(yhat))
                    w_face(yhat) dS_phys

                  = Vt.T @ (W_face * face_scale[f] * u_phys(F_f(yhat))).

  This does not use the assembled T matrix or the volume coefficient vector.
  """
  Y = trace_data["Y"]
  W_face = trace_data["W_face"]
  Vt = trace_data["Vt_common"]
  kf = trace_data["kf"]

  lam_ref = np.empty(((D + 1) * kf,), dtype=np.float64)
  for face_id in range(D + 1):
    sigma = trace_data["sigma_arrays"][face_id]
    Xhat_f = face_points_in_volume_ref(D, face_id, sigma, Y)
    P_face = affine_map_ref_to_phys(V_phys, Xhat_f)
    u_face = u_phys(P_face)

    sl = slice(face_id * kf, (face_id + 1) * kf)
    lam_ref[sl] = Vt.T @ ((W_face * face_scale[face_id]) * u_face)

  return lam_ref


def make_physical_simplex(D):
  """Construct a non-reference affine simplex with nontrivial face scales."""
  V_phys = np.zeros((D, D + 1), dtype=np.float64, order="F")
  V_phys[:, 0] = np.linspace(0.15, -0.20, D)
  for j in range(1, D + 1):
    V_phys[:, j] = V_phys[:, 0]
    V_phys[j - 1, j] += 1.0 + 0.10 * j
    V_phys[:, j] += 0.03 * j * np.arange(1, D + 1)
  return V_phys


def run_one(D, n, q_vol, q_face, verbose=True):
  if D < 2:
    raise ValueError("this trace test starts at D=2 because faces have positive dimension")

  # Cyclic storage [kappa_1,...,kappa_D,kappa_0].
  base = np.array([0.8, 1.7, 2.3, 1.1, 0.6, 1.4], dtype=np.float64)
  kappa_vol = np.ascontiguousarray(base[:D + 1])

  # Deliberately unsorted global IDs so face orientations are nontrivial.
  gids_base = np.array([42, 7, 100, 13, 55, 31], dtype=np.int32)
  global_vids = np.ascontiguousarray(gids_base[:D + 1])

  V_phys = make_physical_simplex(D)
  geom = affine_from_verts(V_phys)
  face_scale = compute_physical_face_scales(D, V_phys, global_vids)

  # Project physical u onto the reference modal expansion via pullback.
  c_u, vol_data = project_physical_function_to_ref_coeffs(
    D, n, kappa_vol, q_vol, V_phys, manufactured_u_phys
  )
  M = vol_data["M"]

  trace_data = build_trace_inputs(D, n, kappa_vol, q_face, global_vids)
  assert trace_data["M"] == M

  T = assemble_T_full_common(
    D,
    trace_data["M"],
    trace_data["kf"],
    trace_data["nq"],
    trace_data["face_sigma_index"],
    face_scale,
    trace_data["Vt_common"],
    trace_data["W_face"],
    trace_data["Vv_sigma_face"],
  )

  lam_from_T = T @ c_u
  lam_ref = direct_physical_trace_moments(
    D, V_phys, trace_data, face_scale, manufactured_u_phys
  )

  abs_mom = float(np.linalg.norm(lam_from_T - lam_ref))
  rel_mom = abs_mom / max(1e-300, float(np.linalg.norm(lam_ref)))

  # Reconstruct face values after removing the physical face scale.
  max_rel_face_l2 = 0.0
  for face_id in range(D + 1):
    sl = slice(face_id * trace_data["kf"], (face_id + 1) * trace_data["kf"])
    coeff_trace = lam_from_T[sl] / face_scale[face_id]
    uh_face = trace_data["Vt_common"] @ coeff_trace

    sigma = trace_data["sigma_arrays"][face_id]
    Xhat_f = face_points_in_volume_ref(D, face_id, sigma, trace_data["Y"])
    P_face = affine_map_ref_to_phys(V_phys, Xhat_f)
    u_face = manufactured_u_phys(P_face)

    err2 = float(np.sum(trace_data["W_face"] * (uh_face - u_face) ** 2))
    den2 = float(np.sum(trace_data["W_face"] * u_face ** 2))
    rel_l2 = math.sqrt(err2 / max(1e-300, den2))
    max_rel_face_l2 = max(max_rel_face_l2, rel_l2)

  if verbose:
    print(
      f"[physical manufactured trace] D={D} n={n} M={trace_data['M']} "
      f"kf={trace_data['kf']} nq_face={trace_data['nq']} "
      f"detB={geom['detB']:.6e} absdetB={geom['detBabs']:.6e} "
      f"rel_mom={rel_mom:.3e} max_rel_face_L2={max_rel_face_l2:.3e}"
    )
    print(f"  global_vids        = {global_vids.tolist()}")
    print(f"  face_sigma_index   = {trace_data['face_sigma_index'].tolist()}")
    print(f"  face_scale         = {face_scale.tolist()}")
    print(
      "  volume projection  = pullback u_phys(F(xhat)); "
      "physical moments are absdetB times these coefficients"
    )

  np.testing.assert_allclose(lam_from_T, lam_ref, rtol=3e-11, atol=3e-11)
  if max_rel_face_l2 > 3e-11:
    raise AssertionError(f"face L2 trace reconstruction too large: {max_rel_face_l2}")


def run_smoke():
  # n >= 3 so the affine pullback of the degree-3 physical polynomial is
  # exactly representable. Quadrature is padded for weighted products.
  cases = [
    # D, n, q_vol, q_face
    (2, 5, 12, 12),
    (3, 4, 10, 10),
    (4, 3, 8, 8),
  ]
  for case in cases:
    run_one(*case)
  print("\nPhysical manufactured-function jtrace pipeline tests passed.")


if __name__ == "__main__":
  run_smoke()
