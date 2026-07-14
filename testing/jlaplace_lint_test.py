import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jquad_tprod import jquad_mapped_build_kappa
from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod
from jgeom import affine_from_verts
from jlaplace import assemble_L_int


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
  The final kappa entry is the barycentric vertex-0 parameter.
  """
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def build_reference_second_partials(D, n, q_vol, kappa_src, kappa_lap):
  """
  Build promoted reference second-partial blocks Lij_ref[:,:,i,j].

  This mirrors the existing Python precompute path:

    D_i      : kappa_src -> kappa_src + dk_i
    D_j      : kappa_src + dk_i -> kappa_src + dk_i + dk_j
    K        : promote shifted second derivative back to kappa_lap
    Lij      = K @ D_j @ D_i
  """
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


def make_affine_vertices(D):
  """Return V with shape (D,D+1), columns are physical vertices."""
  v0 = np.array([0.15 * (i + 1) - 0.2 for i in range(D)], dtype=np.float64)
  B = np.zeros((D, D), dtype=np.float64)
  for i in range(D):
    B[i, i] = 1.15 + 0.17 * i
  for i in range(D):
    for j in range(D):
      if i != j:
        B[i, j] = 0.06 * (i + 1) - 0.035 * (j + 1)

  V = np.zeros((D, D + 1), dtype=np.float64, order="F")
  V[:, 0] = v0
  for j in range(D):
    V[:, j + 1] = v0 + B[:, j]
  return np.asfortranarray(V)


def physical_points_from_ref(Xhat, V_phys):
  v0 = V_phys[:, 0]
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return v0[None, :] + Xhat @ B.T


def manufactured_u_phys(P):
  D = P.shape[1]
  a = np.array([0.37 + 0.11 * i for i in range(D)], dtype=np.float64)
  b = np.array([0.21 - 0.04 * i for i in range(D)], dtype=np.float64)
  c = np.array([0.045 + 0.015 * i for i in range(D)], dtype=np.float64)

  out = 0.8 + P @ a + (P * P) @ b + (P * P * P) @ c

  for i in range(D):
    for j in range(i + 1, D):
      dij = 0.018 * (i + 1) * (j + 2)
      eij = -0.013 * (i + 2) * (j + 1)
      out += dij * (P[:, i] ** 2) * P[:, j]
      out += eij * P[:, i] * (P[:, j] ** 2)

  return out


def laplacian_manufactured_u_phys(P):
  D = P.shape[1]
  b = np.array([0.21 - 0.04 * i for i in range(D)], dtype=np.float64)
  c = np.array([0.045 + 0.015 * i for i in range(D)], dtype=np.float64)

  out = np.zeros(P.shape[0], dtype=np.float64)
  for i in range(D):
    out += 2.0 * b[i] + 6.0 * c[i] * P[:, i]

  for i in range(D):
    for j in range(i + 1, D):
      dij = 0.018 * (i + 1) * (j + 2)
      eij = -0.013 * (i + 2) * (j + 1)
      # Δ(dij*x_i^2*x_j) = 2*dij*x_j.
      # Δ(eij*x_i*x_j^2) = 2*eij*x_i.
      out += 2.0 * dij * P[:, j]
      out += 2.0 * eij * P[:, i]

  return out


def project_volume_coeffs_of_pullback(D, n, kappa, q_vol, V_phys, fun_phys):
  alpha, tail, invh, M = build_basis_structs(D, n, kappa)
  X, W = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha, tail, invh, D)
  P = physical_points_from_ref(X, V_phys)
  vals = fun_phys(P)
  c = V.T @ (W * vals)
  return c


def project_physical_laplacian_rhs(D, n_int, kappa, q_vol, V_phys, detBabs, lap_phys):
  alpha, tail, invh, m_int = build_basis_structs(D, n_int, kappa)
  X, W = jquad_mapped_build_kappa(D, q_vol, kappa)
  V = jbasis_eval_all(X, kappa, n_int, alpha, tail, invh, D)
  P = physical_points_from_ref(X, V_phys)
  vals = lap_phys(P)
  return detBabs * (V.T @ (W * vals))


def numpy_assemble_L_int(G, detBabs, Lij_ref, m_int):
  M = Lij_ref.shape[0]
  D = G.shape[0]
  A = np.zeros((M, M), dtype=np.float64)
  for i in range(D):
    for j in range(D):
      A += G[i, j] * Lij_ref[:, :, i, j]
  return detBabs * A[:m_int, :]


def run_one(D, n, q_vol):
  if n < 2:
    raise ValueError("n must be at least 2 for L_int")

  kappa = np.array([0.7 + 0.23 * i for i in range(D + 1)], dtype=np.float64)
  V_phys = make_affine_vertices(D)
  geom = affine_from_verts(V_phys)
  BinvT = geom["BinvT"]
  G = np.asfortranarray(BinvT.T @ BinvT)
  detBabs = float(geom["detBabs"])

  _, _, _, M = build_basis_structs(D, n, kappa)
  _, _, _, m_int = build_basis_structs(D, n - 2, kappa)

  Lij_ref = build_reference_second_partials(D, n, q_vol, kappa, kappa)

  L_c = assemble_L_int(D, G, detBabs, Lij_ref, m_int)
  L_np = numpy_assemble_L_int(G, detBabs, Lij_ref, m_int)
  rel_mat = np.linalg.norm(L_c - L_np) / max(1e-300, np.linalg.norm(L_np))

  c_u = project_volume_coeffs_of_pullback(
    D, n, kappa, q_vol, V_phys, manufactured_u_phys
  )
  lhs = L_c @ c_u
  rhs = project_physical_laplacian_rhs(
    D, n - 2, kappa, q_vol, V_phys, detBabs,
    laplacian_manufactured_u_phys
  )

  rel_mansol = np.linalg.norm(lhs - rhs) / max(1e-300, np.linalg.norm(rhs))

  print(
    f"[jlaplace L_int] D={D} n={n} M={M} m_int={m_int} "
    f"q_vol={q_vol} detB={detBabs:.6e} "
    f"rel_mat={rel_mat:.3e} rel_mansol={rel_mansol:.3e}"
  )

  assert rel_mat < 5e-14, rel_mat
  assert rel_mansol < 5e-11, rel_mansol


def run_smoke():
  run_one(D=1, n=6, q_vol=18)
  run_one(D=2, n=6, q_vol=18)
  run_one(D=3, n=5, q_vol=16)
  run_one(D=4, n=4, q_vol=14)
  print("\nAll jlaplace L_int tests passed.")


if __name__ == "__main__":
  run_smoke()
