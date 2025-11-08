import ctypes
import os
import sys
import numpy as np
from jweight import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


print("Testing jweight_w_kappa for κ ≡ 1/2:")

for D in (1, 2, 3):
    kappa = np.array([0.5] * (D + 1), dtype=np.float64)
    result = jweight_w_kappa(kappa, D)
    expected = np.prod(np.arange(1, D + 1))  # factorial using NumPy only
    print(kappa)
    print(f"D = {D}, w_kappa = {result:.12f} (expected {expected})")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri  # not used now, but handy if you switch back
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from jweight import *  # jweight_eval, jweight_w_kappa

EPS = 1e-12  # tiny nudge away from faces only when needed

def call_weight_aos(X, kappa, D):
  X = np.ascontiguousarray(X, dtype=np.float64)
  kappa = np.asarray(kappa, dtype=np.float64)
  npts, d = X.shape
  assert d == D
  # AoS layout: consecutive point rows, D values per point
  return jweight_eval(X, ld_point=D, ld_dim=1, npts=npts, kappa=kappa, D=D)

# --- D = 1 -------------------------------------------------------------------

def eval_1d(kappa, n=800, need_eps=False):
  # κ ≡ 1/2: exact constant 1; no nudge needed
  a = EPS if need_eps else 0.0
  b = 1.0 - (EPS if need_eps else 0.0)
  x = np.linspace(a, b, n, dtype=np.float64)
  X = x.reshape(-1, 1)           # (n, 1), AoS
  W = call_weight_aos(X, kappa, D=1)
  return x, W

# --- D = 2 -------------------------------------------------------------------

def tri_points_scatter(n_side=140, s=1.0, eps=0.0):
  gx = np.linspace(eps, s - eps, n_side)
  gy = np.linspace(eps, s - eps, n_side)
  Xg, Yg = np.meshgrid(gx, gy, indexing="xy")
  mask = (Xg >= eps) & (Yg >= eps) & (Xg + Yg <= s - eps)
  x = Xg[mask].astype(np.float64)
  y = Yg[mask].astype(np.float64)
  return x, y

def eval_2d(kappa, n_side=150, need_eps=False):
  eps = EPS if need_eps else 0.0
  x, y = tri_points_scatter(n_side, s=1.0, eps=eps)
  X = np.column_stack([x, y])    # (n, 2), AoS
  W = call_weight_aos(X, kappa, D=2)
  finite = np.isfinite(W)
  return x[finite], y[finite], W[finite]

# --- D = 3 -------------------------------------------------------------------

def sample_tetrahedron(n, eps=0.0):
  # Dirichlet(1,1,1,1) via normalized exponentials; optional tiny nudge
  u = np.clip(np.random.rand(n, 4), 1e-300, 1.0)  # avoid log(0)
  r = -np.log(u)
  r /= r.sum(axis=1, keepdims=True)
  if eps > 0:
    r = np.clip(r, eps, 1.0 - eps)
    r /= r.sum(axis=1, keepdims=True)
  # Return (x1,x2,x3); x4 is implied in weight via (1 - |x|)
  return r[:, 0], r[:, 1], r[:, 2]

def eval_3d(kappa, npts=10000, need_eps=False):
  eps = EPS if need_eps else 0.0
  x1, x2, x3 = sample_tetrahedron(npts, eps=eps)
  X = np.column_stack([x1, x2, x3])  # (n, 3), AoS
  W = call_weight_aos(X, kappa, D=3)
  finite = np.isfinite(W)
  return x1[finite], x2[finite], x3[finite], W[finite]

# --- κ sets per row ----------------------------------------------------------

kappas = {
  1: [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0]],
  2: [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 4.0]],
  3: [[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0, 0.0], [2.0, 3.0, 4.0, 5.0]],
}

row_labels = [
  "kappa ≡ 1/2",
  "kappa = (1,0[,0...])",
  "kappa = (1,1[,0...])",
  "kappa = (2,3[,4,5])"
]

# For rows with any κ_j < 1/2, we enable the tiny ε nudge to avoid ∞ at faces
def row_needs_eps(kappa_row):
  return any(ki < 0.5 for ki in kappa_row)

# --- Figure ------------------------------------------------------------------

fig = plt.figure(figsize=(13, 14))
gs = fig.add_gridspec(4, 3, wspace=0.25, hspace=0.28)

# Column headers
for c, title in enumerate(["D = 1", "D = 2 (scatter)", "D = 3 (3D scatter)"]):
  ax = fig.add_subplot(gs[0, c])
  ax.axis("off")
  ax.text(0.5, 1.15, title, ha="center", va="bottom", fontsize=12, transform=ax.transAxes)

# D = 1
for r in range(4):
  ax = fig.add_subplot(gs[r, 0])
  k = kappas[1][r]
  x, W = eval_1d(k, n=800, need_eps=row_needs_eps(k))
  ax.plot(x, W)
  ax.set_xlim(0, 1)
  ax.set_xlabel("x")
  ax.set_ylabel("W_kappa(x)")
  ax.grid(True, alpha=0.3)
  ax.text(-0.25, 0.5, row_labels[r], rotation=90, va="center",
          ha="right", transform=ax.transAxes, fontsize=10)

# D = 2 (surface scatter over triangle)
for r in range(4):
  ax = fig.add_subplot(gs[r, 1])
  k = kappas[2][r]
  x, y, W = eval_2d(k, n_side=160, need_eps=row_needs_eps(k))
  p = ax.scatter(x, y, c=W, s=6)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)

# D = 3 (true 3D scatter over tetrahedron)
for r in range(4):
  ax = fig.add_subplot(gs[r, 2], projection='3d')
  k = kappas[3][r]
  x1, x2, x3, W = eval_3d(k, npts=12000, need_eps=row_needs_eps(k))
  p = ax.scatter(x1, x2, x3, c=W, s=3, depthshade=False)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("x3")
  fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
  ax.view_init(elev=22, azim=40)

plt.show()
