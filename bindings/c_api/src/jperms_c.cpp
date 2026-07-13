#include <jperms_c.h>

#include <jperms.hh>

using namespace jsimplex;

static constexpr int JPERMS_C_MAX_D = 6;

template<int D>
static inline bool jperms_valid_face_id(int face_id)
{
  return 0 <= face_id && face_id < D + 1;
}

template<int D>
static inline bool jperms_valid_sigma(const int* sigma)
{
  if (!sigma)
    return false;

  bool used[D] = {};
  for (int i = 0; i < D; ++i)
  {
    const int s = sigma[i];
    if (s < 0 || s >= D || used[s])
      return false;
    used[s] = true;
  }
  return true;
}

template<int D>
static inline int jperms_face_vertices_dispatch(int face_id,
                                                int* out_local_vidx)
{
  if (!out_local_vidx)
    return 1;
  if (!jperms_valid_face_id<D>(face_id))
    return 4;

  dsimplex_face_vertices<D>(face_id, out_local_vidx);
  return 0;
}

template<int D>
static inline int jperms_face_sigma_array_dispatch(const int* global_vids,
                                                   int face_id,
                                                   int* sigma_out)
{
  if (!global_vids || !sigma_out)
    return 1;
  if (!jperms_valid_face_id<D>(face_id))
    return 4;

  dsimplex_compute_face_sigma_array<D>(global_vids, face_id, sigma_out);
  return 0;
}

template<int D>
static inline int jperms_face_sigma_index_dispatch(const int* global_vids,
                                                   int face_id,
                                                   int* sigma_index_out)
{
  if (!global_vids || !sigma_index_out)
    return 1;
  if (!jperms_valid_face_id<D>(face_id))
    return 4;

  *sigma_index_out = dsimplex_compute_face_sigma<D>(global_vids, face_id);
  return 0;
}

template<int D>
static inline int jperms_perm_to_lehmer_index_dispatch(const int* sigma,
                                                       int* index_out)
{
  if (!sigma || !index_out)
    return 1;
  if (!jperms_valid_sigma<D>(sigma))
    return 5;

  *index_out = perm_to_lehmer_index<D>(sigma);
  return 0;
}

template<int D>
static inline int jperms_face_values_local_to_canonical_double_dispatch(
  const double* values_local,
  const int* sigma_local_to_canonical,
  double* values_canonical)
{
  if (!values_local || !sigma_local_to_canonical || !values_canonical)
    return 1;
  if (!jperms_valid_sigma<D>(sigma_local_to_canonical))
    return 5;

  dsimplex_face_values_local_to_canonical<D, double>(
    values_local,
    sigma_local_to_canonical,
    values_canonical
  );
  return 0;
}

template<int D>
static inline int jperms_kappa_storage_to_vertex_dispatch(
  const double* kappa_storage,
  double* kappa_vertex)
{
  if (!kappa_storage || !kappa_vertex)
    return 1;

  dsimplex_kappa_storage_to_vertex<D, double>(kappa_storage, kappa_vertex);
  return 0;
}

template<int D>
static inline int jperms_kappa_vertex_to_storage_dispatch(
  const double* kappa_vertex,
  double* kappa_storage)
{
  if (!kappa_vertex || !kappa_storage)
    return 1;

  dsimplex_kappa_vertex_to_storage<D, double>(kappa_vertex, kappa_storage);
  return 0;
}

template<int D>
static inline int jperms_common_face_kappa_dispatch(
  const double* kappa_volume,
  double* kappa_face)
{
  if (!kappa_volume || !kappa_face)
    return 1;

  dsimplex_common_face_kappa<D, double>(kappa_volume, kappa_face);
  return 0;
}

template<int D>
static inline int jperms_dropped_face_kappa_canonical_dispatch(
  const double* kappa_volume,
  int face_id,
  const int* sigma_local_to_canonical,
  double* kappa_face_canonical)
{
  if (!kappa_volume || !sigma_local_to_canonical || !kappa_face_canonical)
    return 1;
  if (!jperms_valid_face_id<D>(face_id))
    return 4;
  if (!jperms_valid_sigma<D>(sigma_local_to_canonical))
    return 5;

  dsimplex_dropped_face_kappa_canonical<D, double>(
    kappa_volume,
    face_id,
    sigma_local_to_canonical,
    kappa_face_canonical
  );
  return 0;
}

extern "C" {

int jperms_c_max_D(void)
{
  return JPERMS_C_MAX_D;
}

int jperms_face_vertices(int D,
                         int face_id,
                         int* out_local_vidx)
{
  if (!out_local_vidx)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_face_vertices_dispatch<1>(face_id, out_local_vidx);
    case 2: return jperms_face_vertices_dispatch<2>(face_id, out_local_vidx);
    case 3: return jperms_face_vertices_dispatch<3>(face_id, out_local_vidx);
    case 4: return jperms_face_vertices_dispatch<4>(face_id, out_local_vidx);
    case 5: return jperms_face_vertices_dispatch<5>(face_id, out_local_vidx);
    case 6: return jperms_face_vertices_dispatch<6>(face_id, out_local_vidx);
    default: return 2;
  }
}

int jperms_face_sigma_array(int D,
                            const int* global_vids,
                            int face_id,
                            int* sigma_out)
{
  if (!global_vids || !sigma_out)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_face_sigma_array_dispatch<1>(global_vids, face_id, sigma_out);
    case 2: return jperms_face_sigma_array_dispatch<2>(global_vids, face_id, sigma_out);
    case 3: return jperms_face_sigma_array_dispatch<3>(global_vids, face_id, sigma_out);
    case 4: return jperms_face_sigma_array_dispatch<4>(global_vids, face_id, sigma_out);
    case 5: return jperms_face_sigma_array_dispatch<5>(global_vids, face_id, sigma_out);
    case 6: return jperms_face_sigma_array_dispatch<6>(global_vids, face_id, sigma_out);
    default: return 2;
  }
}

int jperms_face_sigma_index(int D,
                            const int* global_vids,
                            int face_id,
                            int* sigma_index_out)
{
  if (!global_vids || !sigma_index_out)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_face_sigma_index_dispatch<1>(global_vids, face_id, sigma_index_out);
    case 2: return jperms_face_sigma_index_dispatch<2>(global_vids, face_id, sigma_index_out);
    case 3: return jperms_face_sigma_index_dispatch<3>(global_vids, face_id, sigma_index_out);
    case 4: return jperms_face_sigma_index_dispatch<4>(global_vids, face_id, sigma_index_out);
    case 5: return jperms_face_sigma_index_dispatch<5>(global_vids, face_id, sigma_index_out);
    case 6: return jperms_face_sigma_index_dispatch<6>(global_vids, face_id, sigma_index_out);
    default: return 2;
  }
}

int jperms_perm_to_lehmer_index(int D,
                                const int* sigma,
                                int* index_out)
{
  if (!sigma || !index_out)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_perm_to_lehmer_index_dispatch<1>(sigma, index_out);
    case 2: return jperms_perm_to_lehmer_index_dispatch<2>(sigma, index_out);
    case 3: return jperms_perm_to_lehmer_index_dispatch<3>(sigma, index_out);
    case 4: return jperms_perm_to_lehmer_index_dispatch<4>(sigma, index_out);
    case 5: return jperms_perm_to_lehmer_index_dispatch<5>(sigma, index_out);
    case 6: return jperms_perm_to_lehmer_index_dispatch<6>(sigma, index_out);
    default: return 2;
  }
}

int jperms_face_values_local_to_canonical_double(int D,
                                                 const double* values_local,
                                                 const int* sigma_local_to_canonical,
                                                 double* values_canonical)
{
  if (!values_local || !sigma_local_to_canonical || !values_canonical)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_face_values_local_to_canonical_double_dispatch<1>(values_local, sigma_local_to_canonical, values_canonical);
    case 2: return jperms_face_values_local_to_canonical_double_dispatch<2>(values_local, sigma_local_to_canonical, values_canonical);
    case 3: return jperms_face_values_local_to_canonical_double_dispatch<3>(values_local, sigma_local_to_canonical, values_canonical);
    case 4: return jperms_face_values_local_to_canonical_double_dispatch<4>(values_local, sigma_local_to_canonical, values_canonical);
    case 5: return jperms_face_values_local_to_canonical_double_dispatch<5>(values_local, sigma_local_to_canonical, values_canonical);
    case 6: return jperms_face_values_local_to_canonical_double_dispatch<6>(values_local, sigma_local_to_canonical, values_canonical);
    default: return 2;
  }
}

int jperms_kappa_storage_to_vertex(int D,
                                   const double* kappa_storage,
                                   double* kappa_vertex)
{
  if (!kappa_storage || !kappa_vertex)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_kappa_storage_to_vertex_dispatch<1>(kappa_storage, kappa_vertex);
    case 2: return jperms_kappa_storage_to_vertex_dispatch<2>(kappa_storage, kappa_vertex);
    case 3: return jperms_kappa_storage_to_vertex_dispatch<3>(kappa_storage, kappa_vertex);
    case 4: return jperms_kappa_storage_to_vertex_dispatch<4>(kappa_storage, kappa_vertex);
    case 5: return jperms_kappa_storage_to_vertex_dispatch<5>(kappa_storage, kappa_vertex);
    case 6: return jperms_kappa_storage_to_vertex_dispatch<6>(kappa_storage, kappa_vertex);
    default: return 2;
  }
}

int jperms_kappa_vertex_to_storage(int D,
                                   const double* kappa_vertex,
                                   double* kappa_storage)
{
  if (!kappa_vertex || !kappa_storage)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_kappa_vertex_to_storage_dispatch<1>(kappa_vertex, kappa_storage);
    case 2: return jperms_kappa_vertex_to_storage_dispatch<2>(kappa_vertex, kappa_storage);
    case 3: return jperms_kappa_vertex_to_storage_dispatch<3>(kappa_vertex, kappa_storage);
    case 4: return jperms_kappa_vertex_to_storage_dispatch<4>(kappa_vertex, kappa_storage);
    case 5: return jperms_kappa_vertex_to_storage_dispatch<5>(kappa_vertex, kappa_storage);
    case 6: return jperms_kappa_vertex_to_storage_dispatch<6>(kappa_vertex, kappa_storage);
    default: return 2;
  }
}

int jperms_common_face_kappa(int D,
                             const double* kappa_volume,
                             double* kappa_face)
{
  if (!kappa_volume || !kappa_face)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_common_face_kappa_dispatch<1>(kappa_volume, kappa_face);
    case 2: return jperms_common_face_kappa_dispatch<2>(kappa_volume, kappa_face);
    case 3: return jperms_common_face_kappa_dispatch<3>(kappa_volume, kappa_face);
    case 4: return jperms_common_face_kappa_dispatch<4>(kappa_volume, kappa_face);
    case 5: return jperms_common_face_kappa_dispatch<5>(kappa_volume, kappa_face);
    case 6: return jperms_common_face_kappa_dispatch<6>(kappa_volume, kappa_face);
    default: return 2;
  }
}

int jperms_dropped_face_kappa_canonical(int D,
                                        const double* kappa_volume,
                                        int face_id,
                                        const int* sigma_local_to_canonical,
                                        double* kappa_face_canonical)
{
  if (!kappa_volume || !sigma_local_to_canonical || !kappa_face_canonical)
    return 1;
  if (D < 1 || D > JPERMS_C_MAX_D)
    return 2;

  switch (D)
  {
    case 1: return jperms_dropped_face_kappa_canonical_dispatch<1>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    case 2: return jperms_dropped_face_kappa_canonical_dispatch<2>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    case 3: return jperms_dropped_face_kappa_canonical_dispatch<3>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    case 4: return jperms_dropped_face_kappa_canonical_dispatch<4>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    case 5: return jperms_dropped_face_kappa_canonical_dispatch<5>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    case 6: return jperms_dropped_face_kappa_canonical_dispatch<6>(kappa_volume, face_id, sigma_local_to_canonical, kappa_face_canonical);
    default: return 2;
  }
}

} // extern "C"
