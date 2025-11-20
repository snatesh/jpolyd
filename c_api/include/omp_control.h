#ifndef JPOLY_OMP_CONTROL_H
#define JPOLY_OMP_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Set the number of OpenMP threads for subsequent parallel regions.
 *
 * Equivalent to calling:
 *     omp_set_dynamic(0);
 *     omp_set_num_threads(n);
 *
 * Must be called *before* invoking functions that have 
 * #pragma omp parallel for inside C++ code (e.g. eval_all).
 */
void jpoly_set_omp_threads(int n);

#ifdef __cplusplus
}
#endif

#endif /* JPOLY_OMP_CONTROL_H */
