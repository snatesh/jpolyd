#include <omp_control.h>
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

} // extern "C"
