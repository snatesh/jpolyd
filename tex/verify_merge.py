#!/usr/bin/env python3
"""Independent numerical check of the two-child affine HPS merge formulas."""
import numpy as np

rng = np.random.default_rng(23)
nA, nB, nI = 5, 4, 3
A = rng.standard_normal((nA+nI, nA+nI))
B = rng.standard_normal((nB+nI, nB+nI))
# Synthetic nonsymmetric child maps, but with a nonsingular interface block.
A[nA:, nA:] += 4*np.eye(nI)
B[nB:, nB:] += 4*np.eye(nI)
bA, bB = rng.standard_normal(nA+nI), rng.standard_normal(nB+nI)
AE, AI, AIE, AII = A[:nA,:nA], A[:nA,nA:], A[nA:,:nA], A[nA:,nA:]
BE, BI, BIE, BII = B[:nB,:nB], B[:nB,nB:], B[nB:,:nB], B[nB:,nB:]
H = AII+BII
RA, RB = -np.linalg.solve(H,AIE), -np.linalg.solve(H,BIE)
r = -np.linalg.solve(H,bA[nA:]+bB[nB:])
Sp = np.block([[AE+AI@RA, AI@RB], [BI@RA, BE+BI@RB]])
bp = np.r_[bA[:nA]+AI@r, bB[:nB]+BI@r]
xA, xB = rng.standard_normal(nA), rng.standard_normal(nB)
gamma = RA@xA+RB@xB+r
muA = A@np.r_[xA,gamma]+bA
muB = B@np.r_[xB,gamma]+bB
muP = Sp@np.r_[xA,xB]+bp
print('Interface flux balance error:',np.linalg.norm(muA[nA:]+muB[nB:]))
print('Parent affine map error:   ',np.linalg.norm(np.r_[muA[:nA],muB[:nB]]-muP))
assert np.linalg.norm(muA[nA:]+muB[nB:])<1e-12
assert np.linalg.norm(np.r_[muA[:nA],muB[:nB]]-muP)<1e-12
