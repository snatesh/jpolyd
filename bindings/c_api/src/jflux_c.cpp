#include <jflux_c.h>

#include <jflux.hh>
#include <jperms.hh>

#include <vector>

using namespace jsimplex;

template<int D>
static inline int jflux_validate_common(int M,
                                        int kf,
                                        int nq,
                                        const int* face_sigma_index,
                                        const double* normal_scaled,
                                        const double* BinvT,
                                        const double* Vt,
                                        const double* wS_hat,
                                        const double* dVv_hat_sigma_face,
                                        const double* F_full_out)
{
  if (!face_sigma_index || !normal_scaled || !BinvT || !Vt || !wS_hat ||
      !dVv_hat_sigma_face || !F_full_out)
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
static inline int jflux_assemble_F_full_common_D(int M,
                                                  int kf,
                                                  int nq,
                                                  const int* face_sigma_index,
                                                  const double* normal_scaled,
                                                  const double* BinvT,
                                                  const double* Vt_common,
                                                  const double* wS_hat_common,
                                                  const double* dVv_hat_sigma_face,
                                                  double* F_full_out)
{
  const int ret = jflux_validate_common<D>(M, kf, nq, face_sigma_index,
                                           normal_scaled, BinvT,
                                           Vt_common, wS_hat_common,
                                           dVv_hat_sigma_face, F_full_out);
  if (ret != 0)
  {
    return ret;
  }

  std::vector<double> work((std::size_t)nq * (std::size_t)M, 0.0);

  jdsimplex_assemble_F_full_common_face<D,double>(
    M, kf, nq,
    face_sigma_index,
    normal_scaled,
    BinvT,
    Vt_common,
    wS_hat_common,
    dVv_hat_sigma_face,
    F_full_out,
    work.data()
  );

  return 0;
}

template<int D>
static inline int jflux_assemble_F_full_facepacked_D(int M,
                                                      int kf,
                                                      int nq,
                                                      const int* face_sigma_index,
                                                      const double* normal_scaled,
                                                      const double* BinvT,
                                                      const double* Vt_face,
                                                      const double* wS_hat_face,
                                                      const double* dVv_hat_sigma_face,
                                                      double* F_full_out)
{
  const int ret = jflux_validate_common<D>(M, kf, nq, face_sigma_index,
                                           normal_scaled, BinvT,
                                           Vt_face, wS_hat_face,
                                           dVv_hat_sigma_face, F_full_out);
  if (ret != 0)
  {
    return ret;
  }

  std::vector<double> work((std::size_t)nq * (std::size_t)M, 0.0);

  jdsimplex_assemble_F_full_sigma<D,double>(
    M, kf, nq,
    face_sigma_index,
    normal_scaled,
    BinvT,
    Vt_face,
    wS_hat_face,
    dVv_hat_sigma_face,
    F_full_out,
    work.data()
  );

  return 0;
}

extern "C" {

int jflux_assemble_F_full_common(int D,
                                  int M,
                                  int kf,
                                  int nq,
                                  const int* face_sigma_index,
                                  const double* normal_scaled,
                                  const double* BinvT,
                                  const double* Vt_common,
                                  const double* wS_hat_common,
                                  const double* dVv_hat_sigma_face,
                                  double* F_full_out)
{
  switch (D)
  {
    case 1: return jflux_assemble_F_full_common_D<1>(M, kf, nq, face_sigma_index,
                                                       normal_scaled, BinvT,
                                                       Vt_common, wS_hat_common,
                                                       dVv_hat_sigma_face,
                                                       F_full_out);
    case 2: return jflux_assemble_F_full_common_D<2>(M, kf, nq, face_sigma_index,
                                                       normal_scaled, BinvT,
                                                       Vt_common, wS_hat_common,
                                                       dVv_hat_sigma_face,
                                                       F_full_out);
    case 3: return jflux_assemble_F_full_common_D<3>(M, kf, nq, face_sigma_index,
                                                       normal_scaled, BinvT,
                                                       Vt_common, wS_hat_common,
                                                       dVv_hat_sigma_face,
                                                       F_full_out);
    case 4: return jflux_assemble_F_full_common_D<4>(M, kf, nq, face_sigma_index,
                                                       normal_scaled, BinvT,
                                                       Vt_common, wS_hat_common,
                                                       dVv_hat_sigma_face,
                                                       F_full_out);
    case 5: return jflux_assemble_F_full_common_D<5>(M, kf, nq, face_sigma_index,
                                                       normal_scaled, BinvT,
                                                       Vt_common, wS_hat_common,
                                                       dVv_hat_sigma_face,
                                                       F_full_out);
    default: return 3;
  }
}

int jflux_assemble_F_full_facepacked(int D,
                                      int M,
                                      int kf,
                                      int nq,
                                      const int* face_sigma_index,
                                      const double* normal_scaled,
                                      const double* BinvT,
                                      const double* Vt_face,
                                      const double* wS_hat_face,
                                      const double* dVv_hat_sigma_face,
                                      double* F_full_out)
{
  switch (D)
  {
    case 1: return jflux_assemble_F_full_facepacked_D<1>(M, kf, nq, face_sigma_index,
                                                           normal_scaled, BinvT,
                                                           Vt_face, wS_hat_face,
                                                           dVv_hat_sigma_face,
                                                           F_full_out);
    case 2: return jflux_assemble_F_full_facepacked_D<2>(M, kf, nq, face_sigma_index,
                                                           normal_scaled, BinvT,
                                                           Vt_face, wS_hat_face,
                                                           dVv_hat_sigma_face,
                                                           F_full_out);
    case 3: return jflux_assemble_F_full_facepacked_D<3>(M, kf, nq, face_sigma_index,
                                                           normal_scaled, BinvT,
                                                           Vt_face, wS_hat_face,
                                                           dVv_hat_sigma_face,
                                                           F_full_out);
    case 4: return jflux_assemble_F_full_facepacked_D<4>(M, kf, nq, face_sigma_index,
                                                           normal_scaled, BinvT,
                                                           Vt_face, wS_hat_face,
                                                           dVv_hat_sigma_face,
                                                           F_full_out);
    case 5: return jflux_assemble_F_full_facepacked_D<5>(M, kf, nq, face_sigma_index,
                                                           normal_scaled, BinvT,
                                                           Vt_face, wS_hat_face,
                                                           dVv_hat_sigma_face,
                                                           F_full_out);
    default: return 3;
  }
}

} // extern "C"
