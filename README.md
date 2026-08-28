# jpolyd

`jpolyd` is a C++17 / Fortran / Python library for Jacobi polynomial approximation on simplices, with a focus on high-order operator construction and hierarchical Poincaré–Steklov (HPS) solvers for elliptic PDEs on simplicial meshes.

The core numerical library supports Jacobi bases on the $D$-simplex, quadrature, differentiation and Jacobi-family promotion operators, multiplication operators, affine simplex geometry, trace and flux maps, local Poisson/elliptic operators, and mesh-level HPS merges. A C API and thin Python `ctypes` wrappers expose the same functionality for validation and higher-level mesh experiments.

The current production solver path is dense and precomputed: local elliptic operators are materialized once, the tau-stabilized leaf least-squares problem is factorized with dense QR, and reusable leaf response maps are retained for subsequent boundary data and source terms. Matrix-free and dense/sparse variants remain available as alternative backends and research paths.

## Mathematical conventions

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

using anti-aliased quadrature chosen from the degree of the triple product.

A lifted operator-valued Clenshaw implementation is also retained. It is used by the matrix-free backend and as an algebraic reference for verification.

Relevant header:

- `include/jmult.hh`

### Trace, normal derivative, and affine geometry

The library constructs modal trace and normal-derivative maps on every simplex face, including the face permutations/orientations required by a simplicial mesh.

Relevant headers:

- `include/jtrace.hh`
- `include/jflux.hh`
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

The local PDE residual lies naturally in $\Pi_{n-2}^D$. Pure Neumann problems require the compatibility/gauge branch implemented by the local solver.

Relevant headers:

- `include/jlaplace.hh`
- `include/jleaf.hh`
- `include/jnode.hh`
- `include/jmerge.hh`

### Variable-coefficient non-divergence elliptic solver

The current general elliptic operator is written in non-divergence form,

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

Coefficient fields are represented elementwise in the residual Jacobi family. Principal, first-order, and zero-order terms are assembled after derivative/promotion into the common residual space.

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

### HPS leaf maps and merge algebra

Each leaf combines the interior PDE equations with trace penalty rows. With trace map $T$, flux map $F$, skeleton variable $\lambda$, and tau parameter $\tau$, the augmented flux is

```math
\widehat\mu
=
Fc+\tau(Tc-\lambda).
```

The dense leaf path factorizes the stacked system once with Householder QR and precomputes reusable response maps:

```math
c = U_\lambda \lambda + U_f f,
```

```math
\widehat\mu = S\lambda + G_f f.
```

These maps are then merged hierarchically. Source-transfer maps are retained through the tree so the expensive leaf factorization/materialization can be reused for new source terms and boundary data.

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

This is the current main path.

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

## Current PDE scope and next extension

The current mesh-level variable-coefficient solver is a **non-divergence-form** elliptic solver. Its interface algebra uses the existing normal-derivative-style flux map.

The planned next extension is divergence form,

```math
-\nabla\cdot(A\nabla u)
+
b\cdot\nabla u
+
cu
=
f,
```

implemented by reusing the non-divergence volume machinery after expanding

```math
\nabla\cdot(A\nabla u)
=
A:D^2u
+
(\mathrm{div}A)\cdot\nabla u,
```

and replacing the face flux with the co-normal flux

```math
q = n^T A\nabla u.
```

This will allow correct conservation across interfaces with anisotropic and elementwise discontinuous diffusion tensors while leaving the HPS merge algebra unchanged.

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
- variable-coefficient non-polynomial convergence tests.

Many of these are research/regression drivers rather than a stable command-line interface.

## Numerical validation status

The current solver stack has been exercised with:

- polynomial manufactured solutions recovering to roundoff;
- smooth non-polynomial manufactured solutions showing spectral convergence on fixed affine simplex meshes;
- full HPS mesh trees in dimensions $D=1,\ldots,4$ in the current elliptic convergence studies;
- dense versus matrix-free operator-action comparisons;
- direct-quadrature multiplication matrices versus lifted Clenshaw;
- multiple leaf operator backends;
- source and boundary response-map reuse;
- tau-stabilized HPS merge residual checks.

The dense QR/direct-quadrature path is the current reference implementation for solver development.

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

The next solver-development target is divergence-form elliptic support with co-normal flux conservation.
