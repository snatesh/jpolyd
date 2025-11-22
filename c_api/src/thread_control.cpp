#include <thread_control.h>
#include <cblas.h>
#include <omp.h>

extern "C" {

void jpoly_set_omp_threads(int n)
{
  if (n > 0) 
  {
    omp_set_dynamic(0);
    omp_set_num_threads(n);
  }
}

void jpoly_set_openblas_threads(int n)
{
  if (n > 0)
  {
    // Declaration from OpenBLAS
    // (include <openblas_config.h> or declare it manually)
    //extern void openblas_set_num_threads(int);
    openblas_set_num_threads(n);
  }
}

} // extern "C"
