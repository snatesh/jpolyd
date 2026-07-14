#ifndef JLAPLACE_H
#define JLAPLACE_H

#include <cstddef>

namespace jsimplex {

/*
  Assemble the interior affine Laplacian matrix L_int for one affine D-simplex.

  The caller supplies reference coefficient-space second-derivative blocks

    Lij_ref[:,:,i,j] : Pi_n^D -> Pi_n^D, promoted back to the target/reference
                       Jacobi parameter convention.

  The affine physical map is

    x = v_0 + B xhat,

  and the physical Laplacian of a reference basis function is

    Delta_x phi = sum_{i,j=0}^{D-1} G_{ij} d^2 phi / dxhat_i dxhat_j,
    G = B^{-1} B^{-T}.

  This routine forms

    L_int = |det B| * [ sum_{i,j} G_{ij} Lij_ref[:,:,i,j] ]_{rows 0:m_int-1, :}.

  Matrix/array layout conventions:
    G             D x D column-major, index i + D*j.
    Lij_ref       M x M x D x D, Fortran order, index
                    row + M*(col + M*(i + D*j)).
    L_int_out     m_int x M column-major, index row + m_int*col.

  Here M is dim Pi_n^D, while m_int is usually dim Pi_{n-2}^D. The first
  m_int rows are used because the graded basis ordering puts Pi_{n-2}^D as a
  prefix of Pi_n^D.
*/
template<int D, class Real>
inline void jdsimplex_assemble_L_int(
  int M,
  int m_int,
  const Real* G,
  Real detBabs,
  const Real* Lij_ref,
  Real* L_int_out
)
{
  for (int col = 0; col < M; ++col)
  {
    for (int row = 0; row < m_int; ++row)
    {
      Real acc = (Real)0;
      for (int i = 0; i < D; ++i)
      {
        for (int j = 0; j < D; ++j)
        {
          const Real gij = G[(std::size_t)i + (std::size_t)D * (std::size_t)j];
          if (gij != (Real)0)
          {
            const std::size_t idx = (std::size_t)row
              + (std::size_t)M * ((std::size_t)col
              + (std::size_t)M * ((std::size_t)i
              + (std::size_t)D * (std::size_t)j));
            acc += gij * Lij_ref[idx];
          }
        }
      }
      L_int_out[(std::size_t)row + (std::size_t)m_int * (std::size_t)col]
        = detBabs * acc;
    }
  }
}

} // namespace jsimplex

#endif // JLAPLACE_H
