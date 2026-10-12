# jpolyd

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22137636.svg)](https://doi.org/10.5281/zenodo.22137636)

`jpolyd` is a C++17 / Fortran / Python library for Jacobi polynomial approximation on simplices, with a focus on high-order operator construction and hierarchical Poincaré–Steklov (HPS) solvers for elliptic PDEs on simplicial meshes.

The core numerical library supports Jacobi bases on the $D$-simplex, quadrature, differentiation and Jacobi-family promotion operators, multiplication operators, affine simplex geometry, trace and flux maps, local Poisson/elliptic operators, and mesh-level HPS merges. A C API and thin Python `ctypes` wrappers expose the same functionality for validation and higher-level mesh experiments.

The **persistent HPS solver** supports variable-coefficient elliptic operators in both non-divergence and divergence form. The divergence-form path uses physical co-normal fluxes for interface conservation and supports scalar or facewise polynomial Robin boundary conditions, mixed exterior boundary conditions, and pure Neumann problems with compatibility checking and a configurable boundary-mean gauge. Elementwise polynomial diffusion tensors may be anisotropic or discontinuous across interfaces.

The reference solver backend is dense and precomputed: local elliptic operators are materialized once, the tau-stabilized leaf least-squares problem is factorized with dense QR, and reusable leaf response maps are retained for subsequent boundary data and source terms. Matrix-free and dense/sparse variants remain available for the **non-divergence-form** solver as alternative research backends; the divergence-form implementation currently requires the Dense backend.

## Mathematical conventions

For the underlying simplex Jacobi definitions and notation, see [Dunkl & Xu](https://doi.org/10.1017/CBO9781107786134).

Let

```math
\Pi_n^D
```

denote polynomials of total degree at most $n$ on the $D$-simplex. Its dimension is

```math
\dim \Pi_n^D = \binom{n+D}{D}.
```

We use the reference simplex in Cartesian coordinates,

```math
\widehat{\Delta}_D
=
\left\{
x\in\mathbb{R}^{D}:
x_i\ge 0,\;
\sum_{i=0}^{D-1}x_i\le 1
\right\}.
```

The Jacobi parameter vector has length $D+1$. Its first $D$ entries are associated with the Cartesian coordinates $x_0,\ldots,x_{D-1}$, while the final entry is associated with the remaining simplex factor $1-\sum_i x_i$. The Jacobi weight is

```math
w_\kappa(x)
\propto
\left(1-\sum_{i=0}^{D-1}x_i\right)^{\kappa_D-\frac12}
\prod_{i=0}^{D-1}x_i^{\kappa_i-\frac12},
```

with componentwise admissibility

```math
\kappa_i > -\frac12,
\qquad i=0,\ldots,D.
```

Under this convention:

- `kappa = 0` gives the Dirichlet-half / Chebyshev-type simplex weight

```math
  w(x)
  \propto
  \left(1-\sum_{i=0}^{D-1}x_i\right)^{-\frac12}
  \prod_{i=0}^{D-1}x_i^{-\frac12},
```

- `kappa = 1/2` gives the unweighted simplex measure.

For second-order PDEs, derivative outputs are promoted into the common residual Jacobi family

```math
\kappa_{\mathrm{res}} = \kappa + 2.
```

The variable-coefficient elliptic path currently uses the full trial-degree residual space $R=n$, while the constant-coefficient Poisson path retains the natural second-derivative range $R=n-2$.

## Main capabilities

### Jacobi basis and quadrature

The library provides:

- $D$-simplex Jacobi basis evaluation in graded total-degree ordering;
- multi-index and tail-degree tables;
- tensor-product / collapsed-coordinate simplex quadrature;
- weighted basis evaluation and projection;
- face quadrature and canonical face bases;
- affine reference-to-physical simplex geometry.

Relevant headers include:

- `include/jbasis.hh`
- `include/jweight.hh`
- `include/jquad_tprod.hh`
- `include/jquad_optim.hh`
- `include/jgeom.hh`

### Differentiation and Jacobi-family promotion

Differentiation changes both polynomial degree and Jacobi parameters. The library therefore separates:

- differentiation operators (`DMat`);
- Jacobi-family promotion operators (`KMat`);
- coordinate/Jacobi operators (`JMat`).

Known sparse stencils are used when constructing these operators. `RefSimplexPrecomp` builds and caches the derivative/promotion DAG used by the PDE solvers while retaining dense compatibility maps for validation.

Relevant headers:

- `include/jdmat.hh`
- `include/jkmat.hh`
- `include/jmat.hh`
- `include/jprecomp.hh`

### Multiplication operators

For a coefficient field $q$, the dense elliptic path materializes restricted multiplication operators directly from their Galerkin definition,

```math
M_q^{R\leftarrow N}
=
V_R^T
\mathrm{diag}(w\,q(X))
V_N,
```

using quadrature chosen to resolve the corresponding polynomial triple products. No separate over-integration/de-aliasing stage is added beyond this construction.

A lifted operator-valued Clenshaw implementation is also retained. It is used by the matrix-free backend and as an algebraic reference for verification.

Relevant header:

- `include/jmult.hh`

### Trace, co-normal flux, and affine geometry

The library constructs modal trace and normal-derivative maps on every simplex face, including the face permutations/orientations required by a simplicial mesh. The divergence-form solver additionally constructs the physical **co-normal flux** $q=n^T A\nabla u$ and uses it in the HPS numerical flux and interface-conservation equations.

The HPS skeleton and prescribed boundary right-hand sides are stored as **physical-face-scaled Jacobi moments**. Facewise polynomial Robin coefficients, by contrast, are supplied as modal expansion coefficients in the common canonical face basis, without the physical face-measure scaling. This distinction also enters the pure-Neumann gauge construction.

Relevant headers:

- `include/jtrace.hh`
- `include/jflux.hh`
- `include/jconormal.hh`
- `include/jperms.hh`
- `include/jgeom.hh`

### Poisson solver

The Poisson path solves constant-coefficient problems on affine simplicial meshes with Robin boundary data

```math
\alpha u + \beta q = g.
```

For Poisson,

```math
q = n\cdot\nabla u.
```

The local PDE residual lies naturally in $\Pi_{n-2}^D$. The older one-shot Poisson mesh-tree entry point retains its restriction on globally pure Neumann data. For a mesh-level pure-Neumann solve with compatibility and gauge handling, use the **persistent divergence-form HPS interface** described below (with $A=I$, $b=c=0$).

Relevant headers:

- `include/jlaplace.hh`
- `include/jleaf.hh`
- `include/jnode.hh`
- `include/jmerge.hh`

### Variable-coefficient non-divergence elliptic solver

The existing expanded elliptic operator is written in non-divergence form,

```math
Lu
=
\sum_{r,s=0}^{D-1}
a_{rs}(x)\,\partial_{x_r x_s}u
+
\sum_{r=0}^{D-1}
b_r(x)\,\partial_{x_r}u
+
c(x)u.
```

In this expanded-form interface, coefficient fields are represented elementwise in the residual Jacobi family. Principal, first-order, and zero-order terms are assembled after derivative/promotion into the common residual space.

The current elliptic residual policy is

```math
R=n,
```

so

```math
L_{\mathrm{int}}:\Pi_n^D\to\Pi_n^D
```

after projection.

Relevant header:

- `include/jelliptic.hh`

### Divergence-form elliptic solver

The persistent divergence-form solver implements

```math
-\nabla\cdot\bigl(A(x)\nabla u\bigr)
+ b(x)\cdot\nabla u + c(x)u = f
```

on an affine simplicial mesh, with physical co-normal flux

```math
q = n^T A(x)\nabla u.
```

Here $A$ is an elementwise polynomial $D\times D$ diffusion tensor, and $b$ and $c$ are optional elementwise polynomial convection and reaction coefficients. **The input coefficient fields use the base $\kappa$ Jacobi family**, in contrast to the expanded non-divergence interface. The divergence adapter computes $\nabla \cdot A$ element by element, converts the PDE into the existing expanded operator machinery, and promotes terms into the $\kappa+2$ residual family. Callers supply the physical source $f$ for the equation above; the adapter handles the internal source-sign convention.

The local diffusion tensor may be anisotropic and discontinuous between elements. The HPS merge enforces cancellation of the two outward **co-normal numerical fluxes** across an interface; it does not impose continuity of ordinary normal derivatives. The established tau-stabilized merge algebra is reused unchanged.

The divergence-form implementation currently uses the **Dense leaf backend only**. The polynomial coefficient degrees are specified as `p_A`, `p_b`, and `p_c`; a degree of `-1` for `p_b` or `p_c` disables that term.

Relevant headers:

- `include/jdivergence.hh`
- `include/jconormal.hh`
- `include/jelliptic.hh`
- `include/jnode.hh`
- `include/jmerge.hh`

### Variable and mixed Robin boundary conditions

The persistent elliptic solvers support the exterior condition

```math
\alpha(x)u + \beta(x)q = g,
```

where $q$ is the normal derivative for the expanded-form solver and the physical co-normal flux for the divergence-form solver. Scalars $\alpha,\beta$ are provided at construction. Spatially variable, facewise polynomial coefficients are updated with `update_variable_robin()`, using the same canonical face basis for every face; this also accommodates **mixed Dirichlet, Neumann, and Robin conditions** by choosing the coefficients on each exterior face.

For variable $\beta$ and polynomial diffusion of degree $p_A\ge2$ ($D\ge2$), $q$ may exceed the degree- $n$ face skeleton. The Dense divergence-form solver therefore retains high-order exterior co-normal observations through degree $n+p_A-1$. The assembled root Robin operator has the form

```math
R_{\mathrm{root}}
= M_{\alpha}^{LL}
+ M_{\beta}^{LL}S_{\mathrm{root}}
+ M_{\beta}^{LH}H_{\mathrm{root}},
```

with a corresponding source-dependent right-hand-side correction $-M_{\beta}^{LH}z_{\mathrm{root}}$. The high-order term represents the coupling of physical flux modes above face degree $n$ back into the degree- $n$ boundary equations when multiplied by $\beta(x)$. The existing degree- $n$ HPS skeleton, numerical-flux stabilization, and merge algebra are unchanged.

Both source data and variable Robin coefficients can be updated without rebuilding the tree. Root refactorizations for Robin updates are **transactional**: an invalid update does not overwrite the previously usable root factorization.

### Pure Neumann compatibility and gauge

A globally pure-Neumann boundary condition is specified by $\alpha=0$ and nonzero $\beta$ on every exterior face, with $g=\beta q$. For a connected domain and $c\equiv0$, constants belong to the right nullspace of the divergence-form operator, including when $b\ne0$. The persistent solver uses a bordered root solve with a discrete **left-nullspace compatibility check** and the physical boundary-mean gauge

```math
\frac{1}{|\partial\Omega|}
\int_{\partial\Omega}u\,dS = \eta.
```

The default is $\eta=0$. The value is set with `set_neumann_gauge(eta)` and can be changed between solves without rebuilding or refactorizing the HPS tree. The gauge value persists across Robin/Neumann boundary updates and is ignored when the root operator is nonsingular. Changing $\eta$ in a constant-nullspace problem changes the solution only by an additive constant.

The compatibility check uses the **left null vector of the assembled root operator**, so it applies to nonsymmetric problems with drift; simply checking $\int_\Omega f+\int_{\partial\Omega}g=0$ is not sufficient in general when $b\ne0$. Incompatible data are rejected, rather than silently projected onto the compatible subspace.

Physical face scaling is included in the constant-trace vector and in the unweighted boundary-mean functional, even for arbitrary admissible Jacobi weights. The root implementation applies a safeguarded rank-one correction for the small discrete constant-mode defect when $c\equiv0$, verifies that exactly one null mode remains, and then factors the bordered system for reuse.

With a nonsingular pure-flux root operator (for example, a coercive positive-reaction problem), the usual root LU solve is used **without** imposing a gauge. More general spectral resonances, which may occur for certain reaction coefficients, are not addressed by this one-dimensional constant-nullspace branch.

The compatibility/gauge implementation is in the **persistent** elliptic/divergence solver APIs; the legacy one-shot Poisson and elliptic mesh-tree entry points still reject globally pure-Neumann problems.

Relevant header:

- `include/jneumann_root.hh`

### HPS leaf maps and merge algebra

Each leaf combines the interior PDE equations with trace penalty rows. With trace map $T$, flux map $F$, skeleton variable $\lambda$, and tau parameter $\tau$, the augmented numerical flux is

```math
\widehat\mu
=
Fc+\tau(Tc-\lambda).
```

For divergence form, $F$ denotes the co-normal flux map; for the original expanded-form operator it denotes the existing normal-derivative flux map. The dense leaf path factorizes the stacked system once with Householder QR and precomputes reusable response maps:

```math
c = U_\lambda \lambda + U_f f,
```

```math
\widehat\mu = S\lambda + G_f f.
```

These maps are then merged hierarchically. Source-transfer maps are retained through the tree so the expensive leaf factorization/materialization can be reused for new source terms and boundary data. The high-order exterior co-normal response is propagated separately when needed for variable-$\beta$ boundary equations.

The elliptic tau parameter is interpreted as a base constant and rescaled for the enlarged residual space,

```math
C_{\tau,\mathrm{eff}}
=
C_{\tau,\mathrm{base}}
\frac{m_R}{m_2},
```

before the usual face-size/degree scaling is applied. The base constant remains user-configurable; Robin data with nonzero $\beta$ can affect the most useful range.

Relevant headers:

- `include/jleaf.hh`
- `include/jnode.hh`
- `include/jmerge.hh`
- `include/jmesh.hh`

## Leaf operator backends

The HPS implementation currently exposes several leaf backends.

### Dense

This is the current main path and the required backend for divergence form and its high-order co-normal Robin correction.

- direct-quadrature multiplication-matrix materialization;
- dense local elliptic operator;
- tau-stabilized dense stacked system;
- one QR factorization per leaf;
- batched construction of `[U_lambda U_f]`;
- reusable `S` and `G_f` maps.

### DenseSparse

A hybrid experimental path retaining a dense interior operator with sparse/CSC trace and flux maps. It was useful for measuring sparsity and memory tradeoffs, but on the current CPU/OpenBLAS target the dense path is generally faster.

### MatrixFree

An apply-only path using the derivative DAG and lifted Clenshaw multiplication without materializing the full elliptic operator. This is substantially slower for the tested low/moderate-dimensional CPU cases because multiplication is repeated many times inside LSMR, but it remains useful for:

- high-dimensional problems where dense storage becomes prohibitive;
- memory-constrained settings;
- architectures where sparse/apply-only kernels may be more competitive;
- verification against the dense operator.

### Verify

Builds/uses both representations and checks dense versus matrix-free actions.

## Current PDE scope and limitations

| Solver path | Interior operator | Exterior boundary support | Leaf backend |
| --- | --- | --- | --- |
| Poisson / legacy one-shot mesh-tree | Constant-coefficient Laplacian | Scalar Robin; legacy one-shot pure Neumann not enabled | Existing Poisson modes |
| Persistent expanded elliptic (`HpsEllipticSolver`) | Polynomial non-divergence operator | Scalar and facewise polynomial Robin, mixed conditions, and constant-nullspace Neumann gauge | Dense, with alternative experimental modes |
| Persistent divergence form (`HpsDivergenceSolver`) | $-\nabla\cdot(A\nabla u)+b\cdot\nabla u+cu$ | Physical co-normal scalar/variable Robin, mixed conditions, and pure Neumann compatibility/gauge | **Dense only** |

The numerical implementation assumes affine simplices and the chosen polynomial truncation and quadrature. It does not automatically certify uniform ellipticity of a supplied diffusion tensor, uniqueness of every reaction/convection problem, or well-posedness at exceptional spectral resonances. The current pure-Neumann gauge is specifically the **single constant-mode** case, not a general multi-nullspace Fredholm solver.

## Repository layout

```text
include/                 Header-only C++ numerical core
bindings/c_api/include/  C API headers
bindings/c_api/src/      C/C++ wrappers around the template library
bindings/lsmr/           Fortran/C ABI glue for reverse-communication LSMR
third_party/lsmr/        SOL LSMR Fortran sources
python/                  Thin ctypes-based Python wrappers
testing/                 Python regression, manufactured-solution, and research tests
tex/                     Mathematical notes / derivations
src/                     C++ regression tests and Jacobi-codec experiments
```

### `src/`

The numerical-library regression executables in `src/` include tests for:

- `JMat`, `DMat`, and `KMat` stencil construction/caching;
- derivative/promotion DAG compatibility;
- direct-quadrature multiplication matrices versus lifted Clenshaw;
- dense versus matrix-free elliptic actions;
- leaf operator modes;
- Poisson HPS leaf modes;
- thread-safe LSMR reverse communication.

The remaining non-library-test programs in `src/` are experimental programs for a **Jacobi codec** and related transform/quantization/video experiments. They are not part of the PDE solver API and should be treated as research code.

## Python layer

The Python files in `python/` are thin wrappers around `libjpolyd` rather than a separate reimplementation.

Examples include:

- `jbasis.py`
- `jdmat.py`
- `jkmat.py`
- `jmat.py`
- `jmult.py`
- `jgeom.py`
- `jtrace.py`
- `jflux.py`
- `jlaplace.py`
- `jelliptic.py`
- `jprecomp.py`
- `jhps.py`

`python/libjpolyd_loader.py` loads the shared library from the same installed directory.

The higher-level HPS convergence and mesh experiments under `testing/` use these bindings.

### Persistent divergence-form API

`jhps.py` provides `HpsDivergenceSolver`, which inherits `update_variable_robin()`, `set_neumann_gauge()`, and `solve()` from `HpsEllipticSolver`. It holds the precomputed HPS tree and root factorization across repeated solves. A representative workflow is:

```python
from jhps import HpsDivergenceSolver, HpsLeafOperatorMode

# pc: RefSimplexPrecomp; mesh and modal coefficient arrays already prepared.
# A_coeffs: (nelem, D, D, dimPi(D, p_A)), in base kappa.
# b_coeffs: (nelem, D, dimPi(D, p_b)), or None when p_b == -1.
# f_int:    (nelem, pc.M), physical-source moments for the PDE.
# g_*:      (nboundary_faces, pc.kf), physical-face-scaled moments.
with HpsDivergenceSolver(
  pc, vertex_ids, coords, simplices, merge_pairs,
  A_coeffs, b_coeffs, None, boundary_face_keys,
  p_A=3, p_b=2, p_c=-1,
  alpha=1.0, beta=0.0,
  leaf_operator_mode=HpsLeafOperatorMode.DENSE,
) as solver:
  # Initial Dirichlet solve.
  u_dirichlet = solver.solve(f_int, g_dirichlet)

  # Facewise polynomial alpha=0 and beta(x) != 0: pure Neumann.
  # These arrays are unscaled modal expansion coefficients.
  solver.update_variable_robin(
    alpha_zero_faces, beta_variable_faces,
    p_alpha=2, p_beta=2,
  )
  solver.set_neumann_gauge(0.75)
  u_neumann = solver.solve(f_int, g_neumann)
```

The coefficient arrays for `update_variable_robin()` have shapes `(nboundary_faces, dimPi(D-1, p_alpha))` and `(nboundary_faces, dimPi(D-1, p_beta))`, respectively, with degrees in `[0,n]`. They use the **original boundary-face order** passed to the constructor; the C++ solver performs the root-face permutation internally. The source and boundary RHS arrays are modal moments, not pointwise values. For nonzero reaction, pass the corresponding `c_coeffs` and `p_c>=0` instead of `None` and `-1`.

For $c=0$, subsequent calls to `set_neumann_gauge()` select a different physical boundary mean without another factorization. For nonsingular Robin or pure-flux problems, the gauge is stored but not imposed. An exception from an incompatible Neumann right-hand side does not invalidate the persistent solver.

## Build requirements

The current CMake project requires:

- CMake 3.16 or newer;
- a C++17 compiler;
- C and Fortran compilers;
- OpenBLAS / BLAS;
- LAPACK;
- LAPACKE;
- OpenMP;
- NLopt.

The current top-level CMake configuration also requires OpenCV because the repository still builds the Jacobi-codec experiment targets.

Some experimental executables additionally use SuiteSparse/SPQR.

Python validation scripts generally require:

- Python 3;
- NumPy;

and, depending on the script:

- SciPy;
- SymPy;
- Matplotlib;
- PyMetis.

## Build

A normal release build is:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DJPOLYD_BUILD_CAPI=ON \
  -DJPOLYD_ENABLE_TEST=ON

cmake --build build -j
```

The current CMake configuration writes:

```text
lib/    shared/static libraries
bin/    executables
```

under the repository root.

Release mode is the default when `CMAKE_BUILD_TYPE` is not specified.

### Timing instrumentation

Detailed elliptic/HPS timing counters can be enabled with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTIMING=ON

cmake --build build -j
```

## Install the Python wrappers

By default, the install target places the Python wrappers and `libjpolyd` into the user's Python site-packages directory:

```bash
cmake --install build
```

The default is equivalent to:

```text
-DJPOLYD_PY_INSTALL_USER=ON
```

Set

```bash
-DJPOLYD_PY_INSTALL_USER=OFF
```

to target the system site-packages location instead.

After installation, a simple import check is:

```bash
python3 - <<'PY'
import jbasis
import jelliptic
import jhps

print("jpolyd Python bindings loaded")
print("Dense backend:", jhps.HpsLeafOperatorMode.DENSE)
PY
```

## Testing

Run the CMake-registered tests with:

```bash
ctest --test-dir build --output-on-failure
```

Useful C++ regression executables include:

```bash
./bin/kmat_stencil_test
./bin/dmat_kmat_stencil_cache_test
./bin/test_jmat_stencil_cache
./bin/test_jprecomp_partial_dag_compat
./bin/test_jleaf_dag_elliptic_compat
./bin/test_jelliptic_matrix_free_actions
./bin/test_jmult_quadrature_matrix
./bin/test_jleaf_operator_modes
./bin/test_jhps_poisson_leaf_operator_modes
./bin/test_lsmr_shim_threadsafe
```

The `testing/` directory contains larger Python validation programs, including:

- basis/quadrature checks;
- manufactured-solution tests;
- Poisson Robin convergence;
- single- and multi-simplex trace/flux tests;
- HPS merge-algebra tests;
- arbitrary-mesh HPS convergence studies;
- dense, dense/sparse, and matrix-free elliptic comparisons;
- variable-coefficient non-polynomial convergence tests;
- divergence-form manufactured solutions and co-normal flux tests;
- scalar and facewise polynomial Robin, mixed Dirichlet/Neumann/Robin data;
- pure-Neumann compatibility, boundary-mean gauge, and persistent Robin/Neumann transitions;
- high-order exterior co-normal flux corrections for variable Robin coefficients.

For example, after building and installing the updated Python wrappers:

```bash
python testing/test_hps_divergence_variable_robin_polynomial.py
python testing/test_hps_divergence_mixed_boundary_polynomial.py
python testing/test_hps_divergence_pure_neumann.py --D 1,2,3 --n 4
python testing/test_hps_divergence_neumann_robin_transitions.py --D 1,2,3 --n 4
python testing/test_hps_divergence_neumann_high_conormal_transitions.py --D 2,3 --n 5,6
python testing/test_hps_divergence_neumann_nonzero_gauge.py --D 1,2,3 --n 4
python testing/test_hps_divergence_neumann_nonzero_gauge.py --D 2,3 --n 5 --high-order
```

Many of these are research/regression drivers rather than a stable command-line interface; individual tests may require additional Python dependencies and runtime settings.

## Numerical validation status

The current solver stack has been exercised with:

- polynomial manufactured solutions recovering to roundoff;
- smooth non-polynomial manufactured solutions showing spectral convergence on fixed affine simplex meshes;
- full HPS mesh trees in dimensions $D=1,\ldots,4$ in the current elliptic convergence studies;
- dense versus matrix-free operator-action comparisons;
- direct-quadrature multiplication matrices versus lifted Clenshaw;
- multiple leaf operator backends;
- source and boundary response-map reuse;
- tau-stabilized HPS merge residual checks;
- elementwise polynomial divergence-form operators with co-normal interface conservation;
- polynomial manufactured-solution recovery for cubic anisotropic diffusion, polynomial drift/reaction, and variable Robin coefficients on $D=1,2,3$ meshes;
- mixed boundary conditions and persistent, source-independent Robin updates;
- scalar pure-Neumann compatibility/gauge tests in $D=1,2,3$, $n=3,4,5,6$;
- Robin $\leftrightarrow$ variable- $\beta$ Neumann transitions with rollback after invalid data/updates;
- nonzero physical boundary-mean gauges, including nonzero drift and variable diffusion;
- nontrivial high-order co-normal coupling $M_\beta^{LH}q_H$ for cubic diffusion in $D=2,3$ at $n=5,6$.

The recent polynomial Neumann/Robin tests generally recover solutions near floating-point accuracy (roughly $10^{-13}$ to $10^{-11}$ in relative $L^2$ error in the configurations tested). These are manufactured-solution regressions on fixed affine meshes, not blanket error bounds for arbitrary meshes or coefficient fields. The dense QR/direct-quadrature path remains the reference implementation for solver development.

## Mathematical notes

The `tex/` directory contains derivations and working notes for the simplex Jacobi and HPS constructions. In particular:

- `tex/jsimplex.tex`
- `tex/steklov.tex`
- `tex/steklov.pdf`

These files are useful references for the operator-valued Jacobi/Clenshaw formulation, trace/Steklov constructions, and solver derivations.

## Development philosophy

This repository is research software. The current emphasis is:

1. preserve mathematically explicit simplex/Jacobi operators;
2. validate each operator against manufactured solutions or independent constructions;
3. materialize reusable local HPS maps when doing so is faster and memory-feasible;
4. keep matrix-free/sparse alternatives available where dimensionality or hardware may favor them;
5. separate solver mathematics from experimental optimization paths.

The completed divergence-form/co-normal Robin/Neumann extension is now part of the persistent solver. Future work includes broader physical-problem validation and performance/scaling studies; general resonant Neumann/Fredholm problems remain outside the present constant-gauge treatment.

## Citation

If `jpolyd` contributes to published research, please cite the software.

The project-wide DOI is
[10.5281/zenodo.22137636](https://doi.org/10.5281/zenodo.22137636),
which always resolves to the latest archived release.

For reproducible research, please cite the DOI of the specific release used.
The preferred citation metadata is provided in
[`CITATION.cff`](CITATION.cff).
