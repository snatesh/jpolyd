#ifndef JDSIMPLEX_GEOM_H
#define JDSIMPLEX_GEOM_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <jdetail.hh>
#include <jperms.hh>

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

template<int D, class Real>
struct DSimplexFaceGeom
{
  static_assert(D >= 1, "D must be positive");

  Real scale = Real(0);       // Embedded face measure scale, no factorial.
  Real h = Real(0);           // Characteristic face diameter: max edge length.
  std::array<Real,D> unit_normal{};
  std::array<Real,D> normal_scaled{}; // scale * unit_normal.
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

template<int D, class Real>
inline void dsimplex_metric_from_geom(
  const DSimplexGeom<D,Real>& geom,
  Real* G_colmajor
)
{
  static_assert(D >= 1, "D must be positive");
  assert(G_colmajor != nullptr);
  assert(geom.valid);

  // G = B^{-1} B^{-T}.  Since geom.BinvT = B^{-T},
  // G(i,j) = sum_k Binv(i,k) Binv(j,k)
  //        = sum_k BinvT(k,i) BinvT(k,j).
  for (int j = 0; j < D; ++j)
  {
    for (int i = 0; i < D; ++i)
    {
      Real acc = Real(0);
      for (int k = 0; k < D; ++k)
      {
        acc += geom.BinvT[k + D * i] * geom.BinvT[k + D * j];
      }
      G_colmajor[i + D * j] = acc;
    }
  }
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

template<int D, class Real>
inline Real dsimplex_face_diameter_colmajor(const Real* Vface)
{
  static_assert(D >= 1, "D must be positive");
  assert(Vface != nullptr);

  if constexpr (D == 1)
  {
    return Real(0);
  }
  else
  {
    Real h = Real(0);
    for (int a = 0; a < D; ++a)
    {
      for (int b = a + 1; b < D; ++b)
      {
        Real d2 = Real(0);
        for (int r = 0; r < D; ++r)
        {
          const Real diff = Vface[r + D * a] - Vface[r + D * b];
          d2 += diff * diff;
        }
        h = std::max(h, std::sqrt(d2));
      }
    }
    return h;
  }
}

template<int D, class Real>
inline void dsimplex_face_unit_normal_from_vertices_colmajor(
  const Real* Vface,
  Real* n_out
)
{
  static_assert(D >= 2, "normal helper only applies to D>=2");
  static_assert(std::is_same<Real,float>::value || std::is_same<Real,double>::value,
                "normal helper currently supports Real=float or Real=double");
  assert(Vface != nullptr);
  assert(n_out != nullptr);

  constexpr int m = D - 1;
  constexpr int ncols = D;

  // A = E^T, column-major m x D.  Its nullspace is the face normal.
  std::vector<Real> A((std::size_t)m * ncols, Real(0));
  for (int col = 0; col < ncols; ++col)
  {
    for (int row = 0; row < m; ++row)
    {
      A[(std::size_t)row + (std::size_t)m * col] =
        Vface[(std::size_t)col + (std::size_t)D * (row + 1)]
      - Vface[(std::size_t)col + (std::size_t)D * 0];
    }
  }

  std::vector<Real> S((std::size_t)std::min(m, ncols), Real(0));
  std::vector<Real> U((std::size_t)m * m, Real(0));
  std::vector<Real> VT((std::size_t)ncols * ncols, Real(0));

  const lapack_int ret = detail::LapackGesdd<Real>::run(
    'A',
    (lapack_int)m,
    (lapack_int)ncols,
    A.data(),
    (lapack_int)m,
    S.data(),
    U.data(),
    (lapack_int)m,
    VT.data(),
    (lapack_int)ncols);

  if (ret != 0)
  {
    throw std::runtime_error("dsimplex: SVD failed while computing face normal");
  }

  Real nrm2 = Real(0);
  for (int r = 0; r < D; ++r)
  {
    const Real nr = VT[(std::size_t)(D - 1) + (std::size_t)D * r];
    n_out[r] = nr;
    nrm2 += nr * nr;
  }

  const Real nrm = std::sqrt(nrm2);
  if (!(nrm > Real(0)))
  {
    throw std::runtime_error("dsimplex: zero face normal from SVD");
  }

  for (int r = 0; r < D; ++r)
  {
    n_out[r] /= nrm;
  }
}

template<int D, class Real>
inline void dsimplex_physical_face_geometry_colmajor(
  const Real* V_phys,
  int face_id,
  DSimplexFaceGeom<D,Real>& out
)
{
  static_assert(D >= 1, "D must be positive");
  assert(V_phys != nullptr);
  assert(0 <= face_id && face_id < D + 1);

  out = DSimplexFaceGeom<D,Real>{};

  if constexpr (D == 1)
  {
    const Real x0 = V_phys[0];
    const Real x1 = V_phys[1];
    const Real len = std::abs(x1 - x0);
    if (!(len > Real(0)))
    {
      return;
    }

    const Real sgn = (x1 >= x0) ? Real(1) : Real(-1);
    out.scale = Real(1);
    out.h = len;

    // face 0 is opposite vertex 0, i.e. endpoint vertex 1.
    // face 1 is opposite vertex 1, i.e. endpoint vertex 0.
    out.unit_normal[0] = (face_id == 0) ? sgn : -sgn;
    out.normal_scaled[0] = out.unit_normal[0];
    out.valid = true;
  }
  else
  {
    int fv[D];
    dsimplex_face_vertices<D>(face_id, fv);

    Real Vface[D * D]; // D x D, columns are the physical face vertices.
    for (int j = 0; j < D; ++j)
    {
      const int v = fv[j];
      for (int r = 0; r < D; ++r)
      {
        Vface[(std::size_t)r + (std::size_t)D * j] =
          V_phys[(std::size_t)r + (std::size_t)D * v];
      }
    }

    const Real s = dsimplex_embedded_simplex_measure_scale_colmajor<D,D-1,Real>(Vface);
    if (!(s > Real(0)))
    {
      return;
    }

    out.scale = s;
    out.h = dsimplex_face_diameter_colmajor<D,Real>(Vface);

    dsimplex_face_unit_normal_from_vertices_colmajor<D,Real>(
      Vface,
      out.unit_normal.data());

    // Orient the normal outward by checking the opposite vertex.
    Real dot_to_opp = Real(0);
    for (int r = 0; r < D; ++r)
    {
      const Real p_opp = V_phys[(std::size_t)r + (std::size_t)D * face_id];
      const Real p0 = Vface[(std::size_t)r];
      dot_to_opp += out.unit_normal[(std::size_t)r] * (p_opp - p0);
    }

    if (dot_to_opp > Real(0))
    {
      for (int r = 0; r < D; ++r)
      {
        out.unit_normal[(std::size_t)r] = -out.unit_normal[(std::size_t)r];
      }
    }

    for (int r = 0; r < D; ++r)
    {
      out.normal_scaled[(std::size_t)r] = s * out.unit_normal[(std::size_t)r];
    }

    out.valid = true;
  }
}

template<int D, class Real>
inline void dsimplex_all_physical_face_geometry_colmajor(
  const Real* V_phys,
  DSimplexFaceGeom<D,Real>* face_geom_out
)
{
  static_assert(D >= 1, "D must be positive");
  assert(V_phys != nullptr);
  assert(face_geom_out != nullptr);

  for (int f = 0; f < D + 1; ++f)
  {
    dsimplex_physical_face_geometry_colmajor<D,Real>(
      V_phys,
      f,
      face_geom_out[f]);
  }
}

} // namespace jsimplex

#endif // JDSIMPLEX_GEOM_H
