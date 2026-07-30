#include <exception>
#include <memory>

#include <jelliptic.hh>
#include <jelliptic_c.h>

using namespace jsimplex;

namespace {

static constexpr int JELLIPTIC_MAX_D = 6;

struct JEllipticPlanBase
{
  virtual ~JEllipticPlanBase() = default;
  virtual int D() const = 0;
  virtual int n() const = 0;
  virtual int M() const = 0;
  virtual int m2() const = 0;
  virtual int m1() const = 0;
  virtual int Mp2() const = 0;
  virtual int Mp1() const = 0;
  virtual int Mp0() const = 0;
  virtual void* create_workspace() const = 0;
  virtual void destroy_workspace(void* workspace) const = 0;
  virtual int assemble(void* workspace,
                       const double* BinvT,
                       double detBabs,
                       const double* Lij_ref,
                       const double* Li_ref,
                       const double* L0_ref,
                       const double* A,
                       const double* b,
                       const double* c,
                       double* L_int_out) const = 0;
};

struct JEllipticWorkspaceBase
{
  virtual ~JEllipticWorkspaceBase() = default;
  const JEllipticPlanBase* owner = nullptr;
};

template<int Dim>
struct JEllipticWorkspace final : JEllipticWorkspaceBase
{
  EllipticWorkspace<Dim,double> workspace;

  JEllipticWorkspace(const JEllipticPlanBase* owner_in,
                     const EllipticPlan<Dim,double>& plan)
    : workspace(plan)
  {
    owner = owner_in;
  }
};

template<int Dim>
struct JEllipticPlan final : JEllipticPlanBase
{
  EllipticPlan<Dim,double> plan;

  JEllipticPlan(int n,
                const double* kappa_res,
                int p2,
                int p1,
                int p0,
                bool assume_symmetric)
    : plan(
        n,
        kappa_res,
        EllipticDegreeSpec{p2, p1, p0},
        assume_symmetric)
  {}

  int D() const override { return Dim; }
  int n() const override { return plan.n; }
  int M() const override { return plan.M; }
  int m2() const override { return plan.m2; }
  int m1() const override { return plan.m1; }
  int Mp2() const override { return plan.coefficient_size(2); }
  int Mp1() const override { return plan.coefficient_size(1); }
  int Mp0() const override { return plan.coefficient_size(0); }

  void* create_workspace() const override
  {
    return new JEllipticWorkspace<Dim>(this, plan);
  }

  void destroy_workspace(void* workspace) const override
  {
    delete static_cast<JEllipticWorkspace<Dim>*>(workspace);
  }

  int assemble(void* workspace,
               const double* BinvT,
               double detBabs,
               const double* Lij_ref,
               const double* Li_ref,
               const double* L0_ref,
               const double* A,
               const double* b,
               const double* c,
               double* L_int_out) const override
  {
    if (!workspace)
      return 1;

    auto* W = static_cast<JEllipticWorkspace<Dim>*>(workspace);
    if (W->owner != this)
      return 2;

    EllipticElementCoefficientsView<Dim,double> coeffs;
    coeffs.A = A;
    coeffs.b = b;
    coeffs.c = c;

    jdsimplex_assemble_elliptic_L_int<Dim,double>(
      plan,
      W->workspace,
      BinvT,
      detBabs,
      Lij_ref,
      Li_ref,
      L0_ref,
      coeffs,
      L_int_out);
    return 0;
  }
};

static JEllipticPlanBase* as_plan(jelliptic_plan_t h)
{
  return reinterpret_cast<JEllipticPlanBase*>(h);
}

static JEllipticWorkspaceBase* as_workspace(jelliptic_workspace_t h)
{
  return reinterpret_cast<JEllipticWorkspaceBase*>(h);
}

} // namespace

extern "C" {

int jelliptic_plan_create(int D,
                          int n,
                          const double* kappa_res,
                          int p2,
                          int p1,
                          int p0,
                          int assume_symmetric,
                          jelliptic_plan_t* plan_out)
{
  if (!kappa_res || !plan_out)
    return 1;
  *plan_out = nullptr;
  if (n < 2 || p2 < -1 || p1 < -1 || p0 < -1)
    return 2;
  if (p2 < 0 && p1 < 0 && p0 < 0)
    return 2;
  if (D < 1 || D > JELLIPTIC_MAX_D)
    return 3;

  try
  {
    JEllipticPlanBase* H = nullptr;
    const bool sym = assume_symmetric != 0;
    switch (D)
    {
      case 1: H = new JEllipticPlan<1>(n, kappa_res, p2, p1, p0, sym); break;
      case 2: H = new JEllipticPlan<2>(n, kappa_res, p2, p1, p0, sym); break;
      case 3: H = new JEllipticPlan<3>(n, kappa_res, p2, p1, p0, sym); break;
      case 4: H = new JEllipticPlan<4>(n, kappa_res, p2, p1, p0, sym); break;
      case 5: H = new JEllipticPlan<5>(n, kappa_res, p2, p1, p0, sym); break;
      case 6: H = new JEllipticPlan<6>(n, kappa_res, p2, p1, p0, sym); break;
      default: return 3;
    }
    *plan_out = reinterpret_cast<jelliptic_plan_t>(H);
    return 0;
  }
  catch (...)
  {
    return 4;
  }
}

void jelliptic_plan_destroy(jelliptic_plan_t plan)
{
  delete as_plan(plan);
}

int jelliptic_plan_dims(jelliptic_plan_t plan,
                        int* D_out,
                        int* n_out,
                        int* M_out,
                        int* m2_out,
                        int* m1_out,
                        int* Mp2_out,
                        int* Mp1_out,
                        int* Mp0_out)
{
  if (!plan || !D_out || !n_out || !M_out || !m2_out || !m1_out ||
      !Mp2_out || !Mp1_out || !Mp0_out)
    return 1;

  const JEllipticPlanBase* H = as_plan(plan);
  *D_out = H->D();
  *n_out = H->n();
  *M_out = H->M();
  *m2_out = H->m2();
  *m1_out = H->m1();
  *Mp2_out = H->Mp2();
  *Mp1_out = H->Mp1();
  *Mp0_out = H->Mp0();
  return 0;
}

int jelliptic_workspace_create(jelliptic_plan_t plan,
                               jelliptic_workspace_t* workspace_out)
{
  if (!plan || !workspace_out)
    return 1;
  *workspace_out = nullptr;

  try
  {
    const JEllipticPlanBase* H = as_plan(plan);
    *workspace_out = reinterpret_cast<jelliptic_workspace_t>(
      H->create_workspace());
    return 0;
  }
  catch (...)
  {
    return 4;
  }
}

void jelliptic_workspace_destroy(jelliptic_workspace_t workspace)
{
  if (!workspace)
    return;
  delete as_workspace(workspace);
}

int jelliptic_assemble_L_int(jelliptic_plan_t plan,
                             jelliptic_workspace_t workspace,
                             const double* BinvT,
                             double detBabs,
                             const double* Lij_ref,
                             const double* Li_ref,
                             const double* L0_ref,
                             const double* A,
                             const double* b,
                             const double* c,
                             double* L_int_out)
{
  if (!plan || !workspace || !BinvT || !Lij_ref || !Li_ref ||
      !L0_ref || !L_int_out)
    return 1;

  JEllipticPlanBase* H = as_plan(plan);
  JEllipticWorkspaceBase* W = as_workspace(workspace);
  if (W->owner != H)
    return 2;

  try
  {
    return H->assemble(
      workspace,
      BinvT,
      detBabs,
      Lij_ref,
      Li_ref,
      L0_ref,
      A,
      b,
      c,
      L_int_out);
  }
  catch (const std::invalid_argument&)
  {
    return 2;
  }
  catch (...)
  {
    return 4;
  }
}

} // extern "C"
