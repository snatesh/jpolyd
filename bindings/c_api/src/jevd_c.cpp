#include <jevd_c.h>
#include <jevd_serial.hh>

int jjevd_serial(double* J,
                 int m,
                 int n,
                 double tol,
                 int max_sweeps,
                 int accumulate_V,
                 double* V,
                 int* sweeps_out,
                 double* max_offdiag_out)
{
  if (!J || m <= 0 || n <= 0 || !sweeps_out || !max_offdiag_out)
  {
    return -1;
  }

  jevd_serial::Params<double> params;
  params.max_sweeps   = (max_sweeps > 0 ? max_sweeps : 1);
  params.tol          = tol;
  params.num_threads  = 1;                 // pure serial
  params.accumulate_V = (accumulate_V != 0) && (V != nullptr);

  jevd_serial::Result<double> res =
    jevd_serial::joint_evd_symmetric(m, n, J, V, params);

  *sweeps_out      = res.sweeps;
  *max_offdiag_out = res.max_offdiag;

  return res.info;
}
