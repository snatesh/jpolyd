#ifndef JPERMS_C_H
#define JPERMS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum simplex dimension compiled into this C wrapper. */
int jperms_c_max_D(void);

/*
  Return local vertex indices on a face.

    out_local_vidx has length D.

  Returns:
    0 success
    1 null pointer
    2 unsupported or invalid D
    4 invalid face_id
*/
int jperms_face_vertices(int D,
                         int face_id,
                         int* out_local_vidx);

/*
  Compute local-face to canonical-face permutation:

    sigma_local_to_canonical[i_local] = i_canonical.

  global_vids has length D+1 and sigma_out has length D.

  Returns:
    0 success
    1 null pointer
    2 unsupported or invalid D
    4 invalid face_id
*/
int jperms_face_sigma_array(int D,
                            const int* global_vids,
                            int face_id,
                            int* sigma_out);

/* Same as jperms_face_sigma_array, but returns the Lehmer index. */
int jperms_face_sigma_index(int D,
                            const int* global_vids,
                            int face_id,
                            int* sigma_index_out);

/* Compute lexicographic Lehmer index of sigma, length D. */
int jperms_perm_to_lehmer_index(int D,
                                const int* sigma,
                                int* index_out);

/*
  Reorder length-D face values from local order to canonical order:

    values_canonical[sigma[i_local]] = values_local[i_local].
*/
int jperms_face_values_local_to_canonical_double(int D,
                                                 const double* values_local,
                                                 const int* sigma_local_to_canonical,
                                                 double* values_canonical);

/* Convert kappa storage order [lambda_1,...,lambda_D,lambda_0]
   to barycentric vertex order [lambda_0,lambda_1,...,lambda_D]. */
int jperms_kappa_storage_to_vertex(int D,
                                   const double* kappa_storage,
                                   double* kappa_vertex);

/* Convert barycentric vertex order [lambda_0,lambda_1,...,lambda_D]
   to storage order [lambda_1,...,lambda_D,lambda_0]. */
int jperms_kappa_vertex_to_storage(int D,
                                   const double* kappa_vertex,
                                   double* kappa_storage);

/* Current common face-basis rule: kappa_face = kappa_volume[0:D]. */
int jperms_common_face_kappa(int D,
                             const double* kappa_volume,
                             double* kappa_face);

/* Diagnostic/future inherited face parameters, respecting cyclic kappa storage. */
int jperms_dropped_face_kappa_canonical(int D,
                                        const double* kappa_volume,
                                        int face_id,
                                        const int* sigma_local_to_canonical,
                                        double* kappa_face_canonical);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPERMS_C_H
