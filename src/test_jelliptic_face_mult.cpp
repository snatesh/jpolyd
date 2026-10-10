#include <jelliptic.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Real = double;

template<int D>
using FaceBasis = jsimplex::Basis<D - 1,Real>;

template<int D>
std::array<Real,D + 1> make_kappa()
{
  // Deliberately nonuniform, strictly admissible Jacobi parameters.
  std::array<Real,D + 1> kappa{};
  for (int r = 0; r <= D; ++r)
    kappa[(std::size_t)r] =
      Real(0.12 + 0.09 * r + 0.035 * (r % 2));
  return kappa;
}

void require_close(
  const std::string& label,
  const std::vector<Real>& got,
  const std::vector<Real>& expected,
  Real tolerance)
{
  if (got.size() != expected.size())
    throw std::runtime_error(label + ": size mismatch");

  Real largest_difference = 0;
  Real largest_reference = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
  {
    largest_difference = std::max(
      largest_difference, std::abs(got[i] - expected[i]));
    largest_reference = std::max(
      largest_reference, std::abs(expected[i]));
  }

  const Real scaled_error =
    largest_difference / std::max(Real(1), largest_reference);

  std::cout << "    " << label
            << " : scaled max error = "
            << std::setprecision(4) << std::scientific
            << scaled_error << '\n';

  if (!(scaled_error <= tolerance))
    throw std::runtime_error(label + ": tolerance exceeded");
}

template<int D>
std::vector<Real> direct_reference_matrix(
  const jsimplex::MultByQQuadraturePlan<D - 1,Real>& quad,
  const Real* coeffs,
  int p,
  int N,
  int R)
{
  // Deliberately independent of assemble_restricted_mult_quadrature:
  // direct sum of weighted triple products at a HIGHER quadrature order.
  const int nq = quad.nq;
  const int Mp = FaceBasis<D>::dim_Pi(p);
  const int MN = FaceBasis<D>::dim_Pi(N);
  const int MR = FaceBasis<D>::dim_Pi(R);

  std::vector<Real> M((std::size_t)MR * MN, Real(0));
  for (int j = 0; j < MN; ++j)
  {
    for (int i = 0; i < MR; ++i)
    {
      long double integral = 0;
      for (int k = 0; k < nq; ++k)
      {
        long double q_value = 0;
        for (int a = 0; a < Mp; ++a)
          q_value +=
            (long double)coeffs[a]
            * (long double)quad.V[(std::size_t)k + (std::size_t)nq * a];

        integral +=
          (long double)quad.W[(std::size_t)k]
          * q_value
          * (long double)quad.V[(std::size_t)k + (std::size_t)nq * i]
          * (long double)quad.V[(std::size_t)k + (std::size_t)nq * j];
      }
      M[(std::size_t)i + (std::size_t)MR * j] = (Real)integral;
    }
  }
  return M;
}

template<int D, class Assemble>
void check_face_multiplication(
  int n,
  const Real* kappa_face,
  const jsimplex::MultByQQuadraturePlan<D - 1,Real>& face_plan,
  int q_face_mult,
  Assemble&& assemble)
{
  const int p_beta_max = 2;
  const int p_H = n + 2;
  const int kL = FaceBasis<D>::dim_Pi(n);
  const int kAll = FaceBasis<D>::dim_Pi(p_H);
  const int kH = kAll - kL;

  if (face_plan.max_degree < p_H)
    throw std::runtime_error("face basis degree too small");
  if (2 * q_face_mult - 1 < n + p_H + p_beta_max)
    throw std::runtime_error("configured face quadrature underintegrates");
  if (face_plan.nq != (int)face_plan.W.size())
    throw std::runtime_error("face quadrature weights size mismatch");

  std::cout << "D=" << D
            << ", face dim=" << D - 1
            << ", n=" << n
            << ", kL=" << kL
            << ", kH=" << kH
            << ", q_face_mult=" << q_face_mult
            << '\n';

  // Independent reference uses a higher-order quadrature rule.
  jsimplex::MultByQQuadraturePlan<D - 1,Real> reference_quad(
    kappa_face, p_H, q_face_mult + 3);

  jsimplex::MultByQQuadratureWorkspace<Real> work;
  const Real tolerance = 2e-10;

  // 1) Constant multiplier: M_1^{n<-p_H} = [ I | 0 ].
  // psi_0 is constant but not necessarily equal to 1, so its
  // modal coefficient representing the function 1 is 1/psi_0.
  const Real psi0 = face_plan.V[0];
  if (std::abs(psi0) < Real(1e-14))
    throw std::runtime_error("zero constant basis value");
  const Real one[] = {Real(1) / psi0};

  std::vector<Real> M_one((std::size_t)kL * kAll, Real(0));
  assemble(one, 0, p_H, n, work, M_one.data());

  std::vector<Real> expected_identity(M_one.size(), Real(0));
  for (int j = 0; j < kL; ++j)
    expected_identity[(std::size_t)j + (std::size_t)kL * j] = Real(1);
  require_close("rectangular identity [I | 0]",
                M_one, expected_identity, tolerance);

  // 2) Compare degree-0/1/2 multipliers for square and rectangular maps.
  // Coefficients are in the common intrinsic face family pre.kappa_face.
  for (int p = 0; p <= p_beta_max; ++p)
  {
    const int Mp = FaceBasis<D>::dim_Pi(p);
    std::vector<Real> q((std::size_t)Mp, Real(0));
    q[0] = Real(0.9) / psi0;
    for (int a = 1; a < Mp; ++a)
      q[(std::size_t)a] =
        Real(0.25 * std::sin(Real(0.7 * (a + 1) + 0.3 * D)));
    if (Mp > 1)
      q.back() += Real(0.35);

    for (const int N : {n, p_H})
    {
      const int MN = FaceBasis<D>::dim_Pi(N);
      std::vector<Real> assembled((std::size_t)kL * MN, Real(0));
      assemble(q.data(), p, N, n, work, assembled.data());

      const auto reference = direct_reference_matrix<D>(
        reference_quad, q.data(), p, N, n);

      require_close(
        "degree " + std::to_string(p)
          + ", N=" + std::to_string(N)
          + " direct weighted inner products",
        assembled, reference, tolerance);

      if (N == n)
      {
        // M_q^{n<-n} is symmetric for a real scalar multiplier.
        Real asymmetry = 0;
        for (int j = 0; j < kL; ++j)
          for (int i = 0; i < j; ++i)
            asymmetry = std::max(
              asymmetry,
              std::abs(assembled[(std::size_t)i + (std::size_t)kL * j]
                       - assembled[(std::size_t)j + (std::size_t)kL * i]));
        if (asymmetry > tolerance)
          throw std::runtime_error("square face multiplier is not symmetric");
      }
      else
      {
        // 3) A full rectangular matrix is two contiguous column blocks:
        // [ M_LL | M_LH ]. Verify application through BLAS-ready pointers.
        const Real* M_LL = assembled.data();
        const Real* M_LH = assembled.data() + (std::size_t)kL * kL;
        std::vector<Real> x((std::size_t)kAll, Real(0));
        for (int j = 0; j < kAll; ++j)
          x[(std::size_t)j] = Real(std::cos(0.43 * (j + 1)));

        std::vector<Real> y_full((std::size_t)kL, Real(0));
        std::vector<Real> y_split((std::size_t)kL, Real(0));
        for (int j = 0; j < kAll; ++j)
          for (int i = 0; i < kL; ++i)
            y_full[(std::size_t)i] +=
              assembled[(std::size_t)i + (std::size_t)kL * j] * x[(std::size_t)j];

        for (int j = 0; j < kL; ++j)
          for (int i = 0; i < kL; ++i)
            y_split[(std::size_t)i] +=
              M_LL[(std::size_t)i + (std::size_t)kL * j] * x[(std::size_t)j];
        for (int j = 0; j < kH; ++j)
          for (int i = 0; i < kL; ++i)
            y_split[(std::size_t)i] +=
              M_LH[(std::size_t)i + (std::size_t)kL * j]
              * x[(std::size_t)(kL + j)];

        require_close("[LL | LH] column-block action",
                      y_split, y_full, tolerance);
      }
    }
  }

  // 4) The wrapper must reject triple products beyond its exactness order,
  // even if the requested basis degrees are otherwise present in the plan.
  const int Mp = FaceBasis<D>::dim_Pi(p_beta_max);
  std::vector<Real> q((std::size_t)Mp, Real(0));
  q[0] = Real(1) / psi0;
  std::vector<Real> too_large((std::size_t)kAll * kAll, Real(0));
  bool rejected = false;
  try
  {
    assemble(
      q.data(), p_beta_max, p_H, p_H, work, too_large.data());
  }
  catch (const std::invalid_argument&)
  {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("expected insufficient-quadrature rejection");

  if constexpr (D == 1)
  {
    if (kL != 1 || kH != 0 || face_plan.nq != 1)
      throw std::runtime_error("incorrect zero-dimensional face dimensions");
  }

  std::cout << "  PASS D=" << D << "\n";
}


template<int D>
void run_dimension(int n, const std::string& stencil_directory)
{
  const auto kappa = make_kappa<D>();
  constexpr int p_beta_max = 2;
  const int p_H = n + 2;

  // Exercise the actual EllipticPlan wrapper in every dimension, including
  // D=4.  The reference precompute composes +e_j KMat promotions internally.
  jsimplex::RefSimplexPrecomp<D,Real> pre(
    n, 2, 0, 0, kappa.data(), stencil_directory);
  jsimplex::EllipticPlan<D,Real> plan(
    pre,
    jsimplex::EllipticDegreeSpec{0, -1, -1},
    true,
    jsimplex::EllipticResidualPolicy::TrialDegree,
    jsimplex::EllipticMultiplicationAssembler::Quadrature,
    false);
  plan.configure_face_multiplication(pre, p_beta_max, p_H);

  auto assemble = [&](const Real* coefficients,
                      int p, int N, int R,
                      jsimplex::MultByQQuadratureWorkspace<Real>& workspace,
                      Real* output)
  {
    plan.assemble_face_mult(
      coefficients, p, N, R, workspace, output);
  };

  check_face_multiplication<D>(
    n, pre.kappa_face.data(),
    plan.face_quadrature_plan(), plan.q_face_mult, assemble);
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const std::string stencil_directory =
      argc >= 2 ? argv[1] : "testing/stencils";

    constexpr int n = 3;
    run_dimension<1>(n, stencil_directory);
    run_dimension<2>(n, stencil_directory);
    run_dimension<3>(n, stencil_directory);
    run_dimension<4>(n, stencil_directory);

    std::cout << "All D=1..4 face multiplication tests passed.\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
