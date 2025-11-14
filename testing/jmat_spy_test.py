import numpy as np
from jmat import *
import matplotlib.pyplot as plt 

D = 3
n = 10
kappa = np.array([1.7, 3.3, 2.8, 3.3], dtype=np.float64)
J_all = jmat_build(D, n, kappa)



spyx = (np.abs(J_all[0,:,:]) > 1e-10)
spyy = (np.abs(J_all[1,:,:]) > 1e-10)
spyz = (np.abs(J_all[2,:,:]) > 1e-10)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].spy(spyx, markersize=3)
axs[0].set_title("J₁")

axs[1].spy(spyy, markersize=3)
axs[1].set_title("J₂")

axs[2].spy(spyz, markersize=3)
axs[2].set_title("J₃")

plt.tight_layout()
plt.show()

