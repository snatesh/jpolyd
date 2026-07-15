#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>

#include <jprecomp.hh>
#include <jprecomp_c.h>

using namespace jsimplex;

namespace {

static constexpr int JPRECOMP_MAX_D = 6;

struct JPrecompBase
{
  virtual ~JPrecompBase() = default;
  virtual int D() const = 0;
  virtual int n() const = 0;
  virtual int q_vol() const = 0;
  virtual int q_face() const = 0;
  virtual int M() const = 0;
  virtual int m_int() const = 0;
  virtual int kf() const = 0;
  virtual int nface() const = 0;
  virtual int nsigma() const = 0;
  virtual int nq_vol() const = 0;
  virtual int nq_face() const = 0;
  virtual const double* face_ref_scale() const = 0;
  virtual const double* Lij_ref() const = 0;
  virtual const double* T_ref() const = 0;
  virtual const double* Fgrad_ref() const = 0;
  virtual const double* Mface_ref() const = 0;
  virtual const double* X_vol() const = 0;
  virtual const double* W_vol() const = 0;
  virtual const double* V_vol() const = 0;
  virtual const double* Y_face() const = 0;
  virtual const double* W_face() const = 0;
  virtual const double* V_face() const = 0;
};

template<int Dim>
struct JPrecompHandle final : JPrecompBase
{
  RefSimplexPrecomp<Dim,double> ref;

  JPrecompHandle(int n, int q_pad, int q_vol, int q_face, const double* kappa)
    : ref(n, q_pad, q_vol, q_face, kappa)
  {}

  int D() const override { return Dim; }
  int n() const override { return ref.n; }
  int q_vol() const override { return ref.q_vol; }
  int q_face() const override { return ref.q_face; }
  int M() const override { return ref.M; }
  int m_int() const override { return ref.m_int; }
  int kf() const override { return ref.kf; }
  int nface() const override { return ref.nface; }
  int nsigma() const override { return ref.nsigma; }
  int nq_vol() const override { return ref.nq_vol; }
  int nq_face() const override { return ref.nq_face; }
  const double* face_ref_scale() const override { return ref.face_ref_scale.data(); }
  const double* Lij_ref() const override { return ref.Lij_ref.data(); }
  const double* T_ref() const override { return ref.T_ref.data(); }
  const double* Fgrad_ref() const override { return ref.Fgrad_ref.data(); }
  const double* Mface_ref() const override { return ref.Mface_ref.data(); }
  const double* X_vol() const override { return ref.X_vol.data(); }
  const double* W_vol() const override { return ref.W_vol.data(); }
  const double* V_vol() const override { return ref.V_vol.data(); }
  const double* Y_face() const override { return ref.Y_face.data(); }
  const double* W_face() const override { return ref.W_face.data(); }
  const double* V_face() const override { return ref.V_face.data(); }
};

static JPrecompBase* as_base(void* h)
{
  return reinterpret_cast<JPrecompBase*>(h);
}

static int check_handle(void* h)
{
  return h ? 0 : 1;
}

} // namespace

extern "C" {

int jprecomp_create(int D,
                    int n,
                    int q_pad,
                    int q_vol,
                    int q_face,
                    const double* kappa,
                    void** handle_out)
{
  if (!kappa || !handle_out) { return 1; }
  *handle_out = nullptr;
  if (n < 2) { return 2; }
  if (D < 1 || D > JPRECOMP_MAX_D) { return 3; }

  try
  {
    JPrecompBase* h = nullptr;
    switch (D)
    {
      case 1: h = new JPrecompHandle<1>(n, q_pad, q_vol, q_face, kappa); break;
      case 2: h = new JPrecompHandle<2>(n, q_pad, q_vol, q_face, kappa); break;
      case 3: h = new JPrecompHandle<3>(n, q_pad, q_vol, q_face, kappa); break;
      case 4: h = new JPrecompHandle<4>(n, q_pad, q_vol, q_face, kappa); break;
      case 5: h = new JPrecompHandle<5>(n, q_pad, q_vol, q_face, kappa); break;
      case 6: h = new JPrecompHandle<6>(n, q_pad, q_vol, q_face, kappa); break;
      default: return 3;
    }
    *handle_out = reinterpret_cast<void*>(h);
    return 0;
  }
  catch (const std::exception&)
  {
    return 4;
  }
  catch (...)
  {
    return 4;
  }
}

void jprecomp_destroy(void* handle)
{
  delete as_base(handle);
}

int jprecomp_dims(void* handle,
                  int* D_out,
                  int* n_out,
                  int* q_vol_out,
                  int* q_face_out,
                  int* M_out,
                  int* m_int_out,
                  int* kf_out,
                  int* nface_out,
                  int* nsigma_out,
                  int* nq_vol_out,
                  int* nq_face_out)
{
  if (!handle || !D_out || !n_out || !q_vol_out || !q_face_out ||
      !M_out || !m_int_out || !kf_out || !nface_out || !nsigma_out ||
      !nq_vol_out || !nq_face_out)
  {
    return 1;
  }
  const JPrecompBase* h = as_base(handle);
  *D_out = h->D();
  *n_out = h->n();
  *q_vol_out = h->q_vol();
  *q_face_out = h->q_face();
  *M_out = h->M();
  *m_int_out = h->m_int();
  *kf_out = h->kf();
  *nface_out = h->nface();
  *nsigma_out = h->nsigma();
  *nq_vol_out = h->nq_vol();
  *nq_face_out = h->nq_face();
  return 0;
}

int jprecomp_get_face_ref_scale(void* handle, double* scale_out)
{
  if (check_handle(handle) || !scale_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  std::copy(h->face_ref_scale(), h->face_ref_scale() + h->nface(), scale_out);
  return 0;
}

int jprecomp_get_Lij_ref(void* handle, double* Lij_out)
{
  if (check_handle(handle) || !Lij_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->M() * h->M() * h->D() * h->D();
  std::copy(h->Lij_ref(), h->Lij_ref() + n, Lij_out);
  return 0;
}

int jprecomp_get_T_ref(void* handle, double* T_out)
{
  if (check_handle(handle) || !T_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->kf() * h->M() * h->nsigma() * h->nface();
  std::copy(h->T_ref(), h->T_ref() + n, T_out);
  return 0;
}

int jprecomp_get_Fgrad_ref(void* handle, double* Fg_out)
{
  if (check_handle(handle) || !Fg_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->kf() * h->M() * h->D() * h->nsigma() * h->nface();
  std::copy(h->Fgrad_ref(), h->Fgrad_ref() + n, Fg_out);
  return 0;
}

int jprecomp_get_Mface_ref(void* handle, double* Mface_out)
{
  if (check_handle(handle) || !Mface_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->kf() * h->kf() * h->nface();
  std::copy(h->Mface_ref(), h->Mface_ref() + n, Mface_out);
  return 0;
}

int jprecomp_get_volume_quad(void* handle, double* X_out, double* W_out)
{
  if (check_handle(handle) || !X_out || !W_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  std::copy(h->X_vol(), h->X_vol() + (std::size_t)h->nq_vol() * h->D(), X_out);
  std::copy(h->W_vol(), h->W_vol() + h->nq_vol(), W_out);
  return 0;
}

int jprecomp_get_volume_basis(void* handle, double* V_out)
{
  if (check_handle(handle) || !V_out) { return 1; }
  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->nq_vol() * h->M();
  std::copy(h->V_vol(), h->V_vol() + n, V_out);
  return 0;
}

int jprecomp_get_face_quad(void* handle, double* Y_out, double* W_out)
{
  if (check_handle(handle) || !W_out) { return 1; }

  const JPrecompBase* h = as_base(handle);

  // For D=1, Y_face has shape (nq_face,0), so Y_out may be null.
  if (h->D() > 1 && !Y_out) { return 1; }

  if (h->D() > 1)
  {
    const std::size_t ny = (std::size_t)h->nq_face() * (h->D() - 1);
    std::copy(h->Y_face(), h->Y_face() + ny, Y_out);
  }

  std::copy(h->W_face(), h->W_face() + h->nq_face(), W_out);
  return 0;
}

int jprecomp_get_face_basis(void* handle, double* V_out)
{
  if (check_handle(handle) || !V_out) { return 1; }

  const JPrecompBase* h = as_base(handle);
  const std::size_t n = (std::size_t)h->nq_face() * h->kf();
  std::copy(h->V_face(), h->V_face() + n, V_out);
  return 0;
}

} // extern "C"
