#include <jgeom_c.h>
#include <jgeom.hh>
#include <algorithm>

using namespace jsimplex;

static constexpr int JGEOM_C_MAX_D = 6;

template<int D>
static inline int jgeom_invert_colmajor_dispatch(const double* A,
                                                 double rel_pivot_tol,
                                                 double* Ainv,
                                                 double* det_out)
{
  if (!A || !Ainv || !det_out)
    return 1;

  bool ok = false;
  if (rel_pivot_tol > 0.0)
  {
    ok = dsimplex_invert_matrix_col_major_pivoted<D, double>(
      A,
      Ainv,
      det_out,
      rel_pivot_tol
    );
  }
  else
  {
    ok = dsimplex_invert_matrix_col_major_pivoted<D, double>(
      A,
      Ainv,
      det_out
    );
  }

  return ok ? 0 : 3;
}

template<int D>
static inline int jgeom_affine_from_verts_dispatch(const double* V,
                                                   double* B_out,
                                                   double* BinvT_out,
                                                   double* detB_out,
                                                   double* detBabs_out)
{
  if (!V || !B_out || !BinvT_out || !detB_out || !detBabs_out)
    return 1;

  DSimplexGeom<D, double> geom;
  dsimplex_affine_from_verts<D, double>(V, geom);

  if (!geom.valid)
  {
    *detB_out = 0.0;
    *detBabs_out = 0.0;
    return 3;
  }

  std::copy(geom.B.begin(), geom.B.end(), B_out);
  std::copy(geom.BinvT.begin(), geom.BinvT.end(), BinvT_out);
  *detB_out = geom.detB;
  *detBabs_out = geom.detBabs;

  return 0;
}

extern "C" {

int jgeom_c_max_D(void)
{
  return JGEOM_C_MAX_D;
}

int jgeom_invert_colmajor(int D,
                          const double* A,
                          double rel_pivot_tol,
                          double* Ainv,
                          double* det_out)
{
  if (!A || !Ainv || !det_out)
    return 1;

  if (D < 1 || D > JGEOM_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jgeom_invert_colmajor_dispatch<1>(A, rel_pivot_tol, Ainv, det_out);
    case 2: return jgeom_invert_colmajor_dispatch<2>(A, rel_pivot_tol, Ainv, det_out);
    case 3: return jgeom_invert_colmajor_dispatch<3>(A, rel_pivot_tol, Ainv, det_out);
    case 4: return jgeom_invert_colmajor_dispatch<4>(A, rel_pivot_tol, Ainv, det_out);
    case 5: return jgeom_invert_colmajor_dispatch<5>(A, rel_pivot_tol, Ainv, det_out);
    case 6: return jgeom_invert_colmajor_dispatch<6>(A, rel_pivot_tol, Ainv, det_out);
    default: return 2;
  }
}

int jgeom_affine_from_verts(int D,
                            const double* V,
                            double* B_out,
                            double* BinvT_out,
                            double* detB_out,
                            double* detBabs_out)
{
  if (!V || !B_out || !BinvT_out || !detB_out || !detBabs_out)
    return 1;

  if (D < 1 || D > JGEOM_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jgeom_affine_from_verts_dispatch<1>(V, B_out, BinvT_out, detB_out, detBabs_out);
    case 2: return jgeom_affine_from_verts_dispatch<2>(V, B_out, BinvT_out, detB_out, detBabs_out);
    case 3: return jgeom_affine_from_verts_dispatch<3>(V, B_out, BinvT_out, detB_out, detBabs_out);
    case 4: return jgeom_affine_from_verts_dispatch<4>(V, B_out, BinvT_out, detB_out, detBabs_out);
    case 5: return jgeom_affine_from_verts_dispatch<5>(V, B_out, BinvT_out, detB_out, detBabs_out);
    case 6: return jgeom_affine_from_verts_dispatch<6>(V, B_out, BinvT_out, detB_out, detBabs_out);
    default: return 2;
  }
}

} // extern "C"
