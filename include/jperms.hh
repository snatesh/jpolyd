#ifndef JDSIMPLEX_PERMS_H
#define JDSIMPLEX_PERMS_H

#include <algorithm>
#include <cassert>

namespace jsimplex {
/*
  Face-permutation convention
  ---------------------------
  A D-simplex face has D local face vertices. Let

    local_global_ids[i]

  be their global vertex IDs in the element's local face order, and let the
  canonical face order be those same IDs sorted increasingly.

  We store

    sigma_local_to_canonical[i] = canonical position of local face vertex i.

  Equivalently,

    local_global_ids[i]
      == canonical_global_ids[sigma_local_to_canonical[i]].

  This is the Python convention used in the reference diagnostics:

    local_bary[i] = canonical_bary[sigma_local_to_canonical[i]].

  Current Jacobi/HPS convention
  -----------------------------
  The operational face basis is one common canonical face basis on all faces.
  Its parameter vector is

    kappa_face_common = kappa_volume[0:D].

  This intentionally does NOT drop the exponent associated with the face's
  opposite vertex, and it intentionally does NOT permute face parameters from
  local face order to canonical face order.

  The sigma permutation is still needed for mapping canonical face quadrature
  coordinates to the local reference-simplex face before evaluating volume
  basis functions. It is not used to choose face Jacobi parameters in the
  current common-face-basis implementation.

  Volume Jacobi storage convention
  --------------------------------
  For a D-simplex volume basis, the stored vector

    kappa_volume = [kappa(lambda_1), ..., kappa(lambda_D), kappa(lambda_0)]

  is cyclic relative to barycentric vertex order. The final entry belongs to
  vertex 0. Helper routines below make this explicit so any future inherited
  face-parameter code does not accidentally treat kappa_volume[v] as the
  exponent attached to barycentric vertex v.
*/

template<int D>
inline void dsimplex_face_vertices(int face_id, int out_local_vidx[D])
{
  static_assert(D >= 1, "D must be positive");
  assert(out_local_vidx != nullptr);
  assert(0 <= face_id && face_id < D + 1);

  int k = 0;
  for (int v = 0; v < D + 1; ++v)
  {
    if (v != face_id)
      out_local_vidx[k++] = v;
  }
  assert(k == D);
}

/*
  Compute the local-face -> canonical-face permutation array.
*/
template<int D>
inline void dsimplex_compute_face_sigma_array(
  const int global_vids[D + 1],
  int face_id,
  int sigma_local_to_canonical[D]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(global_vids != nullptr);
  assert(sigma_local_to_canonical != nullptr);
  assert(0 <= face_id && face_id < D + 1);

  int face_local_vidx[D];
  dsimplex_face_vertices<D>(face_id, face_local_vidx);

  int ids_local[D];
  int ids_canonical[D];

  for (int i = 0; i < D; ++i)
  {
    ids_local[i] = global_vids[face_local_vidx[i]];
    ids_canonical[i] = ids_local[i];
  }

  std::sort(ids_canonical, ids_canonical + D);

  // Scatter convention:
  //   sigma[i_local] = i_canonical.
  for (int i_local = 0; i_local < D; ++i_local)
  {
    const int target = ids_local[i_local];
    int i_canonical = -1;

    for (int j = 0; j < D; ++j)
    {
      if (ids_canonical[j] == target)
      {
        i_canonical = j;
        break;
      }
    }

    // Distinct simplex vertex IDs are required.
    assert(i_canonical >= 0);
    sigma_local_to_canonical[i_local] = i_canonical;
  }
}

inline int factorial_int(int n)
{
  assert(n >= 0);
  int r = 1;
  for (int i = 2; i <= n; ++i)
    r *= i;
  return r;
}

/*
  Lexicographic permutation index via Lehmer code.

  For D=3 this gives

    0 -> (0,1,2)
    1 -> (0,2,1)
    2 -> (1,0,2)
    3 -> (1,2,0)
    4 -> (2,0,1)
    5 -> (2,1,0)

  matching Python all_S3_perms().
*/
template<int D>
inline int perm_to_lehmer_index(const int sigma[D])
{
  static_assert(D >= 1, "D must be positive");
  assert(sigma != nullptr);

  int idx = 0;
  bool used[D] = {};

  for (int i = 0; i < D; ++i)
  {
    const int s = sigma[i];
    assert(0 <= s && s < D);
    assert(!used[s]);

    int smaller_unused = 0;
    for (int v = 0; v < s; ++v)
    {
      if (!used[v])
        ++smaller_unused;
    }

    used[s] = true;
    idx += smaller_unused * factorial_int(D - 1 - i);
  }

  return idx;
}

template<int D>
inline void lehmer_index_to_perm(int index, int sigma[D])
{
  static_assert(D >= 1, "D must be positive");
  assert(sigma != nullptr);

  const int nperm = factorial_int(D);
  assert(0 <= index && index < nperm);

  int remaining[D];
  for (int i = 0; i < D; ++i)
  {
    remaining[i] = i;
  }

  int nrem = D;
  int idx = index;

  for (int i = 0; i < D; ++i)
  {
    const int f = factorial_int(D - 1 - i);
    const int q = idx / f;
    idx = idx % f;

    assert(0 <= q && q < nrem);
    sigma[i] = remaining[q];

    for (int j = q; j + 1 < nrem; ++j)
    {
      remaining[j] = remaining[j + 1];
    }
    --nrem;
  }
}

template<int D>
inline int dsimplex_compute_face_sigma(
  const int global_vids[D + 1],
  int face_id
)
{
  int sigma[D];
  dsimplex_compute_face_sigma_array<D>(global_vids, face_id, sigma);
  return perm_to_lehmer_index<D>(sigma);
}

/*
  Reorder a face-attached vector from local-face order to canonical-face order:

    value_canonical[sigma_local_to_canonical[i_local]]
      = value_local[i_local].
*/
template<int D, class Scalar>
inline void dsimplex_face_values_local_to_canonical(
  const Scalar values_local[D],
  const int sigma_local_to_canonical[D],
  Scalar values_canonical[D]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(values_local != nullptr);
  assert(sigma_local_to_canonical != nullptr);
  assert(values_canonical != nullptr);

  for (int i_local = 0; i_local < D; ++i_local)
  {
    const int i_canonical = sigma_local_to_canonical[i_local];
    assert(0 <= i_canonical && i_canonical < D);
    values_canonical[i_canonical] = values_local[i_local];
  }
}

/*
  Convert Jacobi storage order

    [lambda_1, ..., lambda_D, lambda_0]

  to barycentric vertex order

    [lambda_0, lambda_1, ..., lambda_D].
*/
template<int D, class Scalar>
inline void dsimplex_kappa_storage_to_vertex(
  const Scalar kappa_storage[D + 1],
  Scalar kappa_vertex[D + 1]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(kappa_storage != nullptr);
  assert(kappa_vertex != nullptr);

  kappa_vertex[0] = kappa_storage[D];
  for (int v = 1; v < D + 1; ++v)
    kappa_vertex[v] = kappa_storage[v - 1];
}

/*
  Convert barycentric vertex order

    [lambda_0, lambda_1, ..., lambda_D]

  to Jacobi storage order

    [lambda_1, ..., lambda_D, lambda_0].
*/
template<int D, class Scalar>
inline void dsimplex_kappa_vertex_to_storage(
  const Scalar kappa_vertex[D + 1],
  Scalar kappa_storage[D + 1]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(kappa_vertex != nullptr);
  assert(kappa_storage != nullptr);

  for (int v = 1; v < D + 1; ++v)
    kappa_storage[v - 1] = kappa_vertex[v];
  kappa_storage[D] = kappa_vertex[0];
}

/*
  Current operational face-parameter rule: one common canonical face basis on
  all faces, with parameters equal to the first D entries of the volume storage.

  For D=3:

    kappa_face = [kappa(lambda_1), kappa(lambda_2), kappa(lambda_3)].

  This routine deliberately ignores face_id and sigma.
*/
template<int D, class Scalar>
inline void dsimplex_common_face_kappa(
  const Scalar kappa_volume[D + 1],
  Scalar kappa_face_common[D]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(kappa_volume != nullptr);
  assert(kappa_face_common != nullptr);

  for (int i = 0; i < D; ++i)
    kappa_face_common[i] = kappa_volume[i];
}

/*
  Inherited/drop-opposite-vertex face parameters in canonical face order.

  This is NOT the current operational HPS face-basis rule. It is kept for
  diagnostics and future algebraic-inheritance experiments. Unlike the old
  helper, this version respects the cyclic volume Jacobi storage convention.
*/
template<int D, class Scalar>
inline void dsimplex_dropped_face_kappa_canonical(
  const Scalar kappa_volume[D + 1],
  int face_id,
  const int sigma_local_to_canonical[D],
  Scalar kappa_face_canonical[D]
)
{
  static_assert(D >= 1, "D must be positive");
  assert(kappa_volume != nullptr);
  assert(sigma_local_to_canonical != nullptr);
  assert(kappa_face_canonical != nullptr);
  assert(0 <= face_id && face_id < D + 1);

  int face_local_vidx[D];
  dsimplex_face_vertices<D>(face_id, face_local_vidx);

  Scalar kappa_vertex[D + 1];
  dsimplex_kappa_storage_to_vertex<D, Scalar>(
    kappa_volume,
    kappa_vertex
  );

  // First scatter inherited barycentric exponents into canonical vertex order.
  Scalar kappa_face_vertex_canonical[D];
  for (int i_local = 0; i_local < D; ++i_local)
  {
    const int i_canonical = sigma_local_to_canonical[i_local];
    assert(0 <= i_canonical && i_canonical < D);
    kappa_face_vertex_canonical[i_canonical] =
      kappa_vertex[face_local_vidx[i_local]];
  }

  // Face Jacobi storage is cyclic in one lower dimension:
  //   [mu_1, ..., mu_{D-1}, mu_0].
  // The canonical face vertex order is [mu_0, mu_1, ..., mu_{D-1}].
  for (int i = 1; i < D; ++i)
    kappa_face_canonical[i - 1] = kappa_face_vertex_canonical[i];
  kappa_face_canonical[D - 1] = kappa_face_vertex_canonical[0];
}

/*
  Backward-compatible name for the old inherited/drop-opposite helper.

  New common-face-basis code should call dsimplex_common_face_kappa() instead.
*/
template<int D, class Scalar>
inline void dsimplex_face_kappa_canonical(
  const Scalar kappa_volume[D + 1],
  int face_id,
  const int sigma_local_to_canonical[D],
  Scalar kappa_face_canonical[D]
)
{
  dsimplex_dropped_face_kappa_canonical<D, Scalar>(
    kappa_volume,
    face_id,
    sigma_local_to_canonical,
    kappa_face_canonical
  );
}

} // namespace jsimplex

#endif // JDSIMPLEX_PERMS_H
