import numpy as np
from jmat import *
import matplotlib.pyplot as plt 
import matplotlib.patches as patches  

def add_degree_block_boxes(D, n, ax=None, linewidth=1.0):
  """
  Overlay boxes for each total-degree block Hom(j) on a Pi_n-ordered matrix plot.

  Assumes global ordering is graded by total degree, so:
    block_start(j) = dim_Pi(D, j-1)
    block_end(j)   = dim_Pi(D, j)
    block_size(j)  = dim_Pi(D, j) - dim_Pi(D, j-1)

  Uses: libjpolyd.jbasis_dim_Pi(D, n)
  """
  if ax is None:
    ax = plt.gca()

  dim_Pi = libjpolyd.jbasis_dim_Pi

  # Draw diagonal block boxes and boundary lines
  for j in range(n + 1):
    b0 = 0 if j == 0 else int(dim_Pi(int(D), int(j - 1)))
    b1 = int(dim_Pi(int(D), int(j)))
    size = b1 - b0

    # diagonal block box (b0:b1, b0:b1)
    rect = patches.Rectangle(
      (b0 - 0.5, b0 - 0.5),
      size,
      size,
      fill=False,
      linewidth=linewidth,
      edgecolor="k",
    )
    ax.add_patch(rect)

  # Degree boundaries (vertical/horizontal lines)
  for j in range(n):
    b = int(dim_Pi(int(D), int(j)))
    ax.axvline(b - 0.5, linewidth=linewidth * 0.6, color="k")
    ax.axhline(b - 0.5, linewidth=linewidth * 0.6, color="k")

D = 3
n = 10
kappa = np.array([1.7, 3.3, 2.8, 3.3], dtype=np.float64)
J_all = jmat_build_csc(D, n, kappa)



spyx =J_all[0].toarray()#J_all[0,:,:]# (np.abs(J_all[0,:,:]) > 1e-14)
spyy =J_all[1].toarray()#J_all[1,:,:]# (np.abs(J_all[1,:,:]) > 1e-14)
spyz =J_all[2].toarray()#J_all[2,:,:]# (np.abs(J_all[2,:,:]) > 1e-14)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].spy(spyx, markersize=3)
axs[0].set_title("J₁")

axs[1].spy(spyy, markersize=3)
axs[1].set_title("J₂")

axs[2].spy(spyz, markersize=3)
axs[2].set_title("J₃")
add_degree_block_boxes(D, n, ax=axs[0], linewidth=1.0)
add_degree_block_boxes(D, n, ax=axs[1], linewidth=1.0)
add_degree_block_boxes(D, n, ax=axs[2], linewidth=1.0)

plt.tight_layout()
plt.show()

