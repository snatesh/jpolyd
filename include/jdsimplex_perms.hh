#ifndef JDSIMPLEX_PERMS_H
#define JDSIMPLEX_PERMS_H

#include <algorithm>


/* compute sigma permutation of face verts to sorted as array */

template<int D>
void dsimplex_face_vertices(int face_id, int out_local_vidx[D])
{
  // Local face vertex indices: all except face_id
  int k = 0;
  for (int v = 0; v < D+1; ++v)
    if (v != face_id) out_local_vidx[k++] = v;
}

template<int D>
void dsimplex_compute_face_sigma_array(const int global_vids[D+1],
                                       int face_id,
                                       int sigma_out[D])
{
  int fvidx[D];
  dsimplex_face_vertices<D>(face_id, fvidx);

  // local face global ids
  int ids_local[D];
  for (int j = 0; j < D; ++j) ids_local[j] = global_vids[fvidx[j]];

  // canonical = sorted ids
  int ids_canon[D];
  for (int j = 0; j < D; ++j) ids_canon[j] = ids_local[j];
  std::sort(ids_canon, ids_canon + D);

  // sigma_out[j] = index in local such that ids_local[sigma_out[j]] == ids_canon[j]
  // (assumes all vertex ids distinct)
  for (int j = 0; j < D; ++j)
  {
    int target = ids_canon[j];
    int pos = -1;
    for (int k = 0; k < D; ++k)
      if (ids_local[k] == target) { pos = k; break; }
    sigma_out[j] = pos;
  }
}

/* Convert sigma array to sigma index (Lehmer code) */

template<int D>
static inline int factorial_int(int n)
{
  int r = 1;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

template<int D>
int perm_to_lehmer_index(const int sigma[D])
{
  // sigma is a permutation of 0..D-1
  int idx = 0;
  bool used[D] = {false};

  for (int i = 0; i < D; ++i)
  {
    int s = sigma[i];
    int c = 0;
    for (int v = 0; v < s; ++v) if (!used[v]) ++c;
    used[s] = true;
    idx += c * factorial_int<D>(D - 1 - i);
  }
  return idx;
}

template<int D>
int dsimplex_compute_face_sigma(const int global_vids[D+1], int face_id)
{
  int sigma[D];
  dsimplex_compute_face_sigma_array<D>(global_vids, face_id, sigma);
  return perm_to_lehmer_index<D>(sigma);
}

#endif // JDSIMPLEX_PERMS_H
