#ifndef JDSIMPLEX_GEOM_H
#define JDSIMPLEX_GEOM_H

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace jsimplex {
/*
  Affine D-simplex geometry for

    x = v0 + B xhat.

  Storage convention throughout this header:

    A(i,j) <-> A[i + D*j]

  Vertices are also passed as a D x (D+1) column-major matrix:

    V(coord, vertex) <-> V[coord + D*vertex].
*/
template<int D, class Real>
struct DSimplexGeom
{
  static_assert(D >= 1, "D must be positive");

  std::array<Real, D * D> B{};
  std::array<Real, D * D> BinvT{};
  Real detB = Real(0);
  Real detBabs = Real(0);
  bool valid = false;
};

/*
  Small dense inverse with partial pivoting.

  Input/output storage is column-major:

    A(i,j)    = A[i + D*j]
    Ainv(i,j) = Ainv[i + D*j].

  Returns false when the matrix is singular or numerically defective under
  the scale-aware pivot test.
*/
template<int D, class Real>
inline bool dsimplex_invert_matrix_col_major_pivoted(
  const Real* A,
  Real* Ainv,
  Real* det_out,
  Real rel_pivot_tol = Real(100) * std::numeric_limits<Real>::epsilon()
)
{
  static_assert(D >= 1, "D must be positive");
  assert(A != nullptr);
  assert(Ainv != nullptr);
  assert(det_out != nullptr);

  std::array<Real, D * D> M{};
  std::array<Real, D * D> I{};

  Real scale = Real(0);

  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
    {
      const Real aij = A[i + D * j];
      M[i + D * j] = aij;
      I[i + D * j] = (i == j) ? Real(1) : Real(0);
      scale = std::max(scale, std::abs(aij));
    }
  }

  if (scale == Real(0))
  {
    *det_out = Real(0);
    return false;
  }

  const Real pivot_tol = rel_pivot_tol * scale;

  Real det = Real(1);
  int sign = 1;

  for (int k = 0; k < D; ++k)
  {
    int pivot = k;
    Real pivot_abs = std::abs(M[k + D * k]);

    for (int i = k + 1; i < D; ++i)
    {
      const Real candidate_abs = std::abs(M[i + D * k]);

      if (candidate_abs > pivot_abs)
      {
        pivot_abs = candidate_abs;
        pivot = i;
      }
    }

    if (pivot_abs <= pivot_tol)
    {
      *det_out = Real(0);
      return false;
    }

    if (pivot != k)
    {
      for (int j = 0; j < D; ++j)
      {
        std::swap(M[k + D * j], M[pivot + D * j]);
        std::swap(I[k + D * j], I[pivot + D * j]);
      }

      sign = -sign;
    }

    const Real piv = M[k + D * k];
    det *= piv;

    const Real inv_piv = Real(1) / piv;

    for (int j = 0; j < D; ++j)
    {
      M[k + D * j] *= inv_piv;
      I[k + D * j] *= inv_piv;
    }

    for (int i = 0; i < D; ++i)
    {
      if (i == k)
        continue;

      const Real factor = M[i + D * k];

      if (factor == Real(0))
        continue;

      for (int j = 0; j < D; ++j)
      {
        M[i + D * j] -= factor * M[k + D * j];
        I[i + D * j] -= factor * I[k + D * j];
      }
    }
  }

  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
      Ainv[i + D * j] = I[i + D * j];
  }

  *det_out = Real(sign) * det;
  return true;
}

template<int D, class Real>
inline void dsimplex_affine_from_verts(
  const Real* V,
  DSimplexGeom<D, Real>& geom
)
{
  static_assert(D >= 1, "D must be positive");
  assert(V != nullptr);

  std::array<Real, D * D> B{};
  std::array<Real, D * D> Binv{};

  /*
    B(:,j) = vertex_{j+1} - vertex_0.

    V is D x (D+1) with column-major indexing:

      V(i,a) = V[i + D*a].
  */
  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
      B[i + D * j] = V[i + D * (j + 1)] - V[i + D * 0];
  }

  Real detB = Real(0);
  const bool ok = dsimplex_invert_matrix_col_major_pivoted<D, Real>(
    B.data(),
    Binv.data(),
    &detB
  );

  if (!ok)
  {
    geom = DSimplexGeom<D, Real>{};
    return;
  }

  for (int k = 0; k < D * D; ++k)
    geom.B[k] = B[k];

  /*
    BinvT(i,j) = Binv(j,i), both stored with A(i,j)=A[i+D*j].
  */
  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
      geom.BinvT[i + D * j] = Binv[j + D * i];
  }

  geom.detB = detB;
  geom.detBabs = std::abs(detB);
  geom.valid = true;
}

template<int AmbientD, int SimplexD, class Real>
inline Real dsimplex_embedded_simplex_measure_scale_colmajor(
  const Real* V,
  Real rel_pivot_tol = Real(100) * std::numeric_limits<Real>::epsilon()
)
{
  static_assert(AmbientD >= 1, "AmbientD must be positive");
  static_assert(SimplexD >= 0, "SimplexD must be nonnegative");
  static_assert(SimplexD <= AmbientD, "SimplexD must be <= AmbientD");

  assert(V != nullptr);

  if constexpr (SimplexD == 0)
  {
    return Real(1);
  }
  else
  {
    Real G[SimplexD * SimplexD];

    for (int a = 0; a < SimplexD; ++a)
    {
      for (int b = 0; b < SimplexD; ++b)
      {
        Real dot = Real(0);

        for (int r = 0; r < AmbientD; ++r)
        {
          const Real ea = V[r + AmbientD * (a + 1)] - V[r];
          const Real eb = V[r + AmbientD * (b + 1)] - V[r];
          dot += ea * eb;
        }

        // Column-major storage: G(a,b) = G[a + SimplexD*b]
        G[a + SimplexD * b] = dot;
      }
    }

    Real Ginv[SimplexD * SimplexD];
    Real detG = Real(0);

    const bool ok = dsimplex_invert_matrix_col_major_pivoted<SimplexD,Real>(
      G,
      Ginv,
      &detG,
      rel_pivot_tol
    );

    if (!ok)
    {
      return Real(0);
    }

    return std::sqrt(std::max(Real(0), detG));
  }
}

template<int D, class Real>
inline Real dsimplex_reference_face_scale_from_vertex_ids(
  const int face_vidx[D],
  Real rel_pivot_tol = Real(100) * std::numeric_limits<Real>::epsilon()
)
{
  static_assert(D >= 1, "D must be positive");
  assert(face_vidx != nullptr);

  if constexpr (D == 1)
  {
    return Real(1);
  }
  else
  {
    // Face has D vertices and is embedded in R^D.
    // Vface is D x D column-major:
    //   columns are the embedded reference vertices of the face.
    Real Vface[D * D];

    for (int j = 0; j < D; ++j)
    {
      const int v = face_vidx[j];
      assert(0 <= v && v <= D);

      for (int r = 0; r < D; ++r)
      {
        // Reference simplex vertices:
        //   v_0 = 0
        //   v_i = e_i, i=1,...,D
        Vface[r + D * j] = (v == r + 1) ? Real(1) : Real(0);
      }
    }

    return dsimplex_embedded_simplex_measure_scale_colmajor<D,D-1,Real>(
      Vface,
      rel_pivot_tol
    );
  }
}


} // namespace jsimplex

#endif // JDSIMPLEX_GEOM_H
