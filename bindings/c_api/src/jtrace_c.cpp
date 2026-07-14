#include <jtrace_c.h>

#include <jtrace.hh>
#include <jperms.hh>

#include <vector>

using namespace jsimplex;

template<int D>
static inline int jtrace_validate_common(int M,
                                         int kf,
                                         int nq,
                                         const int* face_sigma_index,
                                         const double* face_scale,
                                         const double* Vt_common,
                                         const double* wS_hat_common,
                                         const double* Vv_sigma_face,
                                         const double* T_full_out)
{
  if (!face_sigma_index || !face_scale || !Vt_common || !wS_hat_common ||
      !Vv_sigma_face || !T_full_out)
  {
    return 1;
  }
  if (M <= 0 || kf <= 0 || nq <= 0)
  {
    return 2;
  }

  const int nface = D + 1;
  const int nsigma = factorial_int(D);
  for (int f = 0; f < nface; ++f)
  {
    if (face_sigma_index[f] < 0 || face_sigma_index[f] >= nsigma)
    {
      return 4;
    }
  }

  return 0;
}

template<int D>
static inline int jtrace_assemble_T_full_common_D(int M,
                                                  int kf,
                                                  int nq,
                                                  const int* face_sigma_index,
                                                  const double* face_scale,
                                                  const double* Vt_common,
                                                  const double* wS_hat_common,
                                                  const double* Vv_sigma_face,
                                                  double* T_full_out)
{
  const int ret = jtrace_validate_common<D>(M, kf, nq, face_sigma_index,
                                            face_scale, Vt_common, wS_hat_common,
                                            Vv_sigma_face, T_full_out);
  if (ret != 0)
  {
    return ret;
  }

  std::vector<double> work((std::size_t)nq * (std::size_t)M, 0.0);

  jdsimplex_assemble_T_full_common_face<D,double>(
    M, kf, nq,
    face_sigma_index,
    face_scale,
    Vt_common,
    wS_hat_common,
    Vv_sigma_face,
    T_full_out,
    work.data()
  );

  return 0;
}

template<int D>
static inline int jtrace_assemble_T_full_facepacked_D(int M,
                                                      int kf,
                                                      int nq,
                                                      const int* face_sigma_index,
                                                      const double* face_scale,
                                                      const double* Vt_face,
                                                      const double* wS_hat_face,
                                                      const double* Vv_sigma_face,
                                                      double* T_full_out)
{
  const int ret = jtrace_validate_common<D>(M, kf, nq, face_sigma_index,
                                            face_scale, Vt_face, wS_hat_face,
                                            Vv_sigma_face, T_full_out);
  if (ret != 0)
  {
    return ret;
  }

  std::vector<double> work((std::size_t)nq * (std::size_t)M, 0.0);

  jdsimplex_assemble_T_full_sigma<D,double>(
    M, kf, nq,
    face_sigma_index,
    face_scale,
    Vt_face,
    wS_hat_face,
    Vv_sigma_face,
    T_full_out,
    work.data()
  );

  return 0;
}

extern "C" {

int jtrace_assemble_T_full_common(int D,
                                  int M,
                                  int kf,
                                  int nq,
                                  const int* face_sigma_index,
                                  const double* face_scale,
                                  const double* Vt_common,
                                  const double* wS_hat_common,
                                  const double* Vv_sigma_face,
                                  double* T_full_out)
{
  switch (D)
  {
    case 1: return jtrace_assemble_T_full_common_D<1>(M, kf, nq, face_sigma_index,
                                                       face_scale, Vt_common,
                                                       wS_hat_common, Vv_sigma_face,
                                                       T_full_out);
    case 2: return jtrace_assemble_T_full_common_D<2>(M, kf, nq, face_sigma_index,
                                                       face_scale, Vt_common,
                                                       wS_hat_common, Vv_sigma_face,
                                                       T_full_out);
    case 3: return jtrace_assemble_T_full_common_D<3>(M, kf, nq, face_sigma_index,
                                                       face_scale, Vt_common,
                                                       wS_hat_common, Vv_sigma_face,
                                                       T_full_out);
    case 4: return jtrace_assemble_T_full_common_D<4>(M, kf, nq, face_sigma_index,
                                                       face_scale, Vt_common,
                                                       wS_hat_common, Vv_sigma_face,
                                                       T_full_out);
    case 5: return jtrace_assemble_T_full_common_D<5>(M, kf, nq, face_sigma_index,
                                                       face_scale, Vt_common,
                                                       wS_hat_common, Vv_sigma_face,
                                                       T_full_out);
    default: return 3;
  }
}

int jtrace_assemble_T_full_facepacked(int D,
                                      int M,
                                      int kf,
                                      int nq,
                                      const int* face_sigma_index,
                                      const double* face_scale,
                                      const double* Vt_face,
                                      const double* wS_hat_face,
                                      const double* Vv_sigma_face,
                                      double* T_full_out)
{
  switch (D)
  {
    case 1: return jtrace_assemble_T_full_facepacked_D<1>(M, kf, nq, face_sigma_index,
                                                           face_scale, Vt_face,
                                                           wS_hat_face, Vv_sigma_face,
                                                           T_full_out);
    case 2: return jtrace_assemble_T_full_facepacked_D<2>(M, kf, nq, face_sigma_index,
                                                           face_scale, Vt_face,
                                                           wS_hat_face, Vv_sigma_face,
                                                           T_full_out);
    case 3: return jtrace_assemble_T_full_facepacked_D<3>(M, kf, nq, face_sigma_index,
                                                           face_scale, Vt_face,
                                                           wS_hat_face, Vv_sigma_face,
                                                           T_full_out);
    case 4: return jtrace_assemble_T_full_facepacked_D<4>(M, kf, nq, face_sigma_index,
                                                           face_scale, Vt_face,
                                                           wS_hat_face, Vv_sigma_face,
                                                           T_full_out);
    case 5: return jtrace_assemble_T_full_facepacked_D<5>(M, kf, nq, face_sigma_index,
                                                           face_scale, Vt_face,
                                                           wS_hat_face, Vv_sigma_face,
                                                           T_full_out);
    default: return 3;
  }
}

} // extern "C"
