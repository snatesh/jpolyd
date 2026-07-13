#ifndef JWEIGHT_C_H
#define JWEIGHT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns w_kappa for given D and kappa[0..D] */
double jweight_w_kappa(const double* kappa, int D);

/* Evaluate normalized weight at npts points from a single flat buffer X.
   Indexing: X[p*ld_point + j*ld_dim],  p=0..npts-1, j=0..D-1.
   Works for AoS (ld_point=D, ld_dim=1) and SoA/column-major (ld_point=1, ld_dim=npts). */
void jweight_eval(const double* X, int ld_point, int ld_dim,
                  int npts, const double* kappa, double* out, int D); 


#ifdef __cplusplus
} // extern "C"
#endif

#endif // JWEIGHT_C_H
