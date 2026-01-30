#ifndef JDSIMPLEX_GEOM_H
#define JDSIMPLEX_GEOM_H

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>

template<int D, class Real>
struct DSimplexGeom
{
  // B maps reference coordinates to physical: x = v0 + B * xhat
  std::array<Real, D*D> B;      // column-major or row-major; pick one and stick to it
  std::array<Real, D*D> BinvT;  // (B^{-T})
  Real detB = 0;
  Real detBabs = 0;
};

template<int D, class Real>
static inline void mat_copy(const Real* A, Real* B, int n)
{
  std::memcpy(B, A, sizeof(Real) * (std::size_t)n);
}

// Simple Gauss-Jordan inverse for small D (D<=6 typical). Replace with LU if desired.
template<int D, class Real>
static inline bool invert_matrix(const Real* A, Real* Ainv, Real* det_out)
{
  // Augment [A | I]
  Real aug[D][2*D];
  for (int i = 0; i < D; ++i)
  {
    for (int j = 0; j < D; ++j) aug[i][j] = A[i*D + j];
    for (int j = 0; j < D; ++j) aug[i][D + j] = (i == j) ? (Real)1 : (Real)0;
  }

  Real det = (Real)1;

  for (int k = 0; k < D; ++k)
  {
    // pivot
    int piv = k;
    Real best = std::abs(aug[k][k]);
    for (int i = k + 1; i < D; ++i)
    {
      Real v = std::abs(aug[i][k]);
      if (v > best) { best = v; piv = i; }
    }
    if (best == (Real)0) { *det_out = (Real)0; return false; }

    if (piv != k)
    {
      for (int j = 0; j < 2*D; ++j) std::swap(aug[k][j], aug[piv][j]);
      det = -det;
    }

    Real pivot = aug[k][k];
    det *= pivot;

    // normalize row
    Real invp = (Real)1 / pivot;
    for (int j = 0; j < 2*D; ++j) aug[k][j] *= invp;

    // eliminate others
    for (int i = 0; i < D; ++i)
    {
      if (i == k) continue;
      Real factor = aug[i][k];
      if (factor == (Real)0) continue;
      for (int j = 0; j < 2*D; ++j) aug[i][j] -= factor * aug[k][j];
    }
  }

  // extract inverse
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      Ainv[i*D + j] = aug[i][D + j];

  *det_out = det;
  return true;
}

template<int D, class Real>
void dsimplex_affine_from_verts(const Real* V /* (D+1) x D row-major */,
                                DSimplexGeom<D,Real>& geom)
{
  // Build B = [v1-v0, v2-v0, ..., vD-v0] as columns
  Real B[D*D];
  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
    {
      const Real vj1 = V[(j+1)*D + i];
      const Real v0  = V[0*D + i];
      // column-major storage: B(i,j)
      B[i + j*D] = vj1 - v0;
    }
  }

  // We’ll invert B in row-major for convenience; convert:
  Real Brow[D*D];
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      Brow[i*D + j] = B[i + j*D];

  Real Binv_row[D*D], detB;
  bool ok = invert_matrix<D,Real>(Brow, Binv_row, &detB);
  if (!ok) { geom.detB = (Real)0; geom.detBabs = (Real)0; return; }

  // Store B column-major
  for (int i = 0; i < D*D; ++i) geom.B[i] = B[i];

  // BinvT = (B^{-1})^T; Binv_row is row-major B^{-1}
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j)
      geom.BinvT[i + j*D] = Binv_row[j*D + i]; // column-major

  geom.detB = detB;
  geom.detBabs = std::abs(detB);
}

#endif // JDSIMPLEX_GEOM_H
