#ifndef JWEIGHT_C_H
#define JWEIGHT_C_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns w_kappa for given D and kappa[0..D]
double jweight_w_kappa(const double* kappa, int D);

// W_kappa(x) = w_kappa * monomial, x[0..D-1], kappa[0..D]
double jweight_eval(const double* x, const double* kappa, int D);

// Monomial only (no gamma ratio)
double jweight_monomial(const double* x, const double* kappa, int D);

// x_last = 1 - sum(x[0..D-1])
double jweight_x_last(const double* x, int D);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JWEIGHT_C_H
