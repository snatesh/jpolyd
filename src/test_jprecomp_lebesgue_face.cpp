#include <jprecomp.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Real = double;

template<int D>
using Precomp = jsimplex::RefSimplexPrecomp<D, Real>;

Real factorial(int n)
{
  Real f = Real(1);
  for (int k = 2; k <= n; ++k)
  {
    f *= static_cast<Real>(k);
  }
  return f;
}

void check_close(const std::string& what,
                 Real actual,
                 Real expected,
                 Real tolerance)
{
  const Real error = std::abs(actual - expected);
  const Real threshold = tolerance * std::max(Real(1), std::abs(expected));
  if (!std::isfinite(actual) || error > threshold)
  {
    std::ostringstream out;
    out << std::setprecision(17)
        << what << ": actual=" << actual
        << ", expected=" << expected
        << ", error=" << error
        << ", threshold=" << threshold;
    throw std::runtime_error(out.str());
  }
}

template<int D>
std::array<Real, D + 1> make_kappa(bool half)
{
  std::array<Real, D + 1> kappa{};
  for (int i = 0; i <= D; ++i)
  {
    kappa[static_cast<std::size_t>(i)] = half
      ? Real(0.5)
      : Real(0.12 + 0.17 * i + 0.04 * (i % 2));
  }
  return kappa;
}

template<int D>
Real monomial(const Real* x, const std::array<int, D>& powers)
{
  Real value = Real(1);
  for (int a = 0; a < D; ++a)
  {
    for (int i = 0; i < powers[static_cast<std::size_t>(a)]; ++i)
    {
      value *= x[a];
    }
  }
  return value;
}

template<int D>
Real exact_simplex_monomial(const std::array<int, D>& powers)
{
  Real numerator = Real(1);
  int degree = 0;
  for (int a = 0; a < D; ++a)
  {
    const int exponent = powers[static_cast<std::size_t>(a)];
    numerator *= factorial(exponent);
    degree += exponent;
  }
  return numerator / factorial(D + degree);
}

// Integral on face `face` of the reference D-simplex with vertices
// 0, e_1, ..., e_D. The face is opposite vertex `face`.
// Coordinate x_a is barycentric coordinate lambda_(a+1).
// The face Jacobian (not its area) is pre.face_ref_scale[face].
template<int D>
Real exact_embedded_face_monomial(
  const Precomp<D>& pre,
  int face,
  const std::array<int, D>& powers)
{
  int degree = 0;
  Real numerator = Real(1);
  for (int a = 0; a < D; ++a)
  {
    const int exponent = powers[static_cast<std::size_t>(a)];
    if (exponent > 0 && face == a + 1)
    {
      return Real(0); // x_a vanishes on the face opposite e_(a+1).
    }
    degree += exponent;
    numerator *= factorial(exponent);
  }

  return pre.face_ref_scale[static_cast<std::size_t>(face)] *
         numerator / factorial(D - 1 + degree);
}

template<int D>
std::vector<std::array<int, D>> volume_test_monomials(int degree)
{
  std::vector<std::array<int, D>> result;
  std::array<int, D> powers{};
  result.push_back(powers); // 1

  for (int axis = 0; axis < D; ++axis)
  {
    powers.fill(0);
    powers[static_cast<std::size_t>(axis)] = 1;
    result.push_back(powers); // x_axis
    powers[static_cast<std::size_t>(axis)] = degree;
    result.push_back(powers); // x_axis^degree
  }
  if constexpr (D >= 2)
  {
    powers.fill(0);
    powers[0] = 1;
    powers[1] = 1;
    result.push_back(powers); // x_0 x_1
  }
  return result;
}

template<int D>
std::vector<Real> project_volume_monomial(
  const Precomp<D>& pre,
  const std::array<int, D>& powers)
{
  std::vector<Real> coefficients(static_cast<std::size_t>(pre.M), Real(0));
  for (int j = 0; j < pre.M; ++j)
  {
    Real sum = Real(0);
    for (int iq = 0; iq < pre.nq_vol; ++iq)
    {
      const Real* x = pre.X_vol.data() +
        static_cast<std::size_t>(iq) * D;
      sum += pre.W_vol[static_cast<std::size_t>(iq)] *
             pre.V_vol[static_cast<std::size_t>(iq) +
                       static_cast<std::size_t>(pre.nq_vol) * j] *
             monomial<D>(x, powers);
    }
    coefficients[static_cast<std::size_t>(j)] = sum;
  }
  return coefficients;
}

template<int D>
Real face_block_integral(
  const Precomp<D>& pre,
  const Real* block,
  const std::vector<Real>& volume_coefficients)
{
  Real total = Real(0);
  for (int i = 0; i < pre.kf; ++i)
  {
    Real weighted_moment = Real(0);
    for (int j = 0; j < pre.M; ++j)
    {
      weighted_moment += block[static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(pre.kf) * j] *
                         volume_coefficients[static_cast<std::size_t>(j)];
    }
    total += pre.face_moment_to_lebesgue[static_cast<std::size_t>(i)] *
             weighted_moment;
  }
  return total;
}

template<int D>
Real test_intrinsic_face_monomials(const Precomp<D>& pre, Real tolerance)
{
  if constexpr (D == 1)
  {
    check_close("zero-dimensional face functional",
                pre.face_moment_to_lebesgue[0], Real(1), tolerance);
    return Real(0);
  }
  else
  {
    constexpr int d = D - 1;
    std::vector<int> alpha(
      static_cast<std::size_t>(pre.kf) * d, 0);
    jsimplex::Basis<d, Real>::build_alpha_table(pre.n, alpha.data());

    Real maximum_error = Real(0);
    // Test *all* canonical-face monomials of total degree <= n.
    for (int m = 0; m < pre.kf; ++m)
    {
      std::array<int, d> powers{};
      for (int a = 0; a < d; ++a)
      {
        powers[static_cast<std::size_t>(a)] =
          alpha[static_cast<std::size_t>(m) * d + a];
      }

      // Orthogonal projection into the existing canonical face basis
      // using its own normalized, kappa_face-weighted quadrature.
      // These are the expansion coefficients of the monomial.
      std::vector<Real> coefficients(
        static_cast<std::size_t>(pre.kf), Real(0));
      for (int j = 0; j < pre.kf; ++j)
      {
        Real coefficient = Real(0);
        for (int iq = 0; iq < pre.nq_face; ++iq)
        {
          const Real* y = pre.Y_face.data() +
            static_cast<std::size_t>(iq) * d;
          coefficient +=
            pre.W_face[static_cast<std::size_t>(iq)] *
            pre.V_face[static_cast<std::size_t>(iq) +
                       static_cast<std::size_t>(pre.nq_face) * j] *
            monomial<d>(y, powers);
        }
        coefficients[static_cast<std::size_t>(j)] = coefficient;
      }

      Real actual = Real(0);
      for (int j = 0; j < pre.kf; ++j)
      {
        actual +=
          pre.face_moment_to_lebesgue[static_cast<std::size_t>(j)] *
          coefficients[static_cast<std::size_t>(j)];
      }
      const Real expected = exact_simplex_monomial<d>(powers);
      maximum_error = std::max(maximum_error, std::abs(actual - expected));
      check_close("intrinsic face monomial", actual, expected, tolerance);
    }
    return maximum_error;
  }
}

template<int D>
Real test_mass_and_trace(
  const Precomp<D>& pre,
  Real tolerance)
{
  const Real zero = Real(0);
  Real maximum_error = Real(0);
  for (int face = 0; face < pre.nface; ++face)
  {
    // The canonical weighted face basis is orthonormal under the
    // normalized Jacobi measure. Mface_ref = face_scale * I.
    const Real scale = pre.face_ref_scale[static_cast<std::size_t>(face)];
    const Real expected_scale = (face == 0)
      ? std::sqrt(static_cast<Real>(D)) : Real(1);
    check_close("reference embedded face Jacobian",
                scale, expected_scale, tolerance);

    const Real* mass = pre.Mface_ref.data() +
      static_cast<std::size_t>(face) * pre.kf * pre.kf;
    for (int j = 0; j < pre.kf; ++j)
    {
      for (int i = 0; i < pre.kf; ++i)
      {
        const Real expected = (i == j) ? scale : zero;
        check_close("weighted face mass matrix",
                    mass[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(pre.kf) * j],
                    expected, tolerance);
      }
    }
  }

  // Test the actual trace AND reference-gradient moment maps, for every
  // reference face and every local-to-canonical face permutation.
  for (const auto& powers : volume_test_monomials<D>(pre.n))
  {
    const auto c = project_volume_monomial<D>(pre, powers);
    for (int face = 0; face < pre.nface; ++face)
    {
      const Real expected_trace =
        exact_embedded_face_monomial<D>(pre, face, powers);
      for (int sigma = 0; sigma < pre.nsigma; ++sigma)
      {
        const std::size_t face_sigma =
          static_cast<std::size_t>(sigma) +
          static_cast<std::size_t>(pre.nsigma) * face;
        const Real* trace = pre.T_ref.data() +
          static_cast<std::size_t>(pre.kf) * pre.M * face_sigma;

        const Real actual = face_block_integral<D>(pre, trace, c);
        maximum_error = std::max(
          maximum_error, std::abs(actual - expected_trace));
        check_close("trace moment -> physical face integral",
                    actual, expected_trace, tolerance);

        // The Fgrad_ref block represents weighted moments of d/dx_a.
        // Differentiating the known monomial gives an exact test that
        // also exercises nonconstant derivative moments and scaling.
        for (int axis = 0; axis < D; ++axis)
        {
          const std::size_t factor_index =
            static_cast<std::size_t>(axis) +
            static_cast<std::size_t>(D) * face_sigma;
          const Real* gradient = pre.Fgrad_ref.data() +
            static_cast<std::size_t>(pre.kf) * pre.M * factor_index;

          const int exponent = powers[static_cast<std::size_t>(axis)];
          Real expected_gradient = Real(0);
          if (exponent > 0)
          {
            auto derivative_powers = powers;
            --derivative_powers[static_cast<std::size_t>(axis)];
            expected_gradient = static_cast<Real>(exponent) *
              exact_embedded_face_monomial<D>(
                pre, face, derivative_powers);
          }
          const Real actual_gradient =
            face_block_integral<D>(pre, gradient, c);
          maximum_error = std::max(
            maximum_error,
            std::abs(actual_gradient - expected_gradient));
          check_close("gradient moment -> physical face integral",
                      actual_gradient, expected_gradient, tolerance);
        }
      }
    }
  }
  return maximum_error;
}

template<int D>
void run_case(int degree,
              bool half,
              Real tolerance,
              const std::string& stencil_folder)
{
  const auto kappa = make_kappa<D>(half);
  const Precomp<D> pre(
    degree, 3, 0, 0, kappa.data(), stencil_folder);

  if (pre.kf != (D == 1
        ? 1 : jsimplex::Basis<D - 1, Real>::dim_Pi(degree)))
  {
    throw std::runtime_error("Wrong canonical face dimension");
  }
  if (pre.face_moment_to_lebesgue.size() !=
      static_cast<std::size_t>(pre.kf))
  {
    throw std::runtime_error("Face integration vector must have kf entries");
  }
  for (int a = 0; a < D; ++a)
  {
    check_close("common face kappa", pre.kappa_face[a], kappa[a],
                tolerance);
  }
  for (Real value : pre.face_moment_to_lebesgue)
  {
    if (!std::isfinite(value))
    {
      throw std::runtime_error("Face functional contains NaN/Inf");
    }
  }

  // The normalized Jacobi basis has P_0 = 1; the intrinsic reference
  // (D-1)-simplex Lebesgue volume is 1/(D-1)!.
  check_close("constant face mode",
              pre.face_moment_to_lebesgue.front(),
              Real(1) / factorial(D - 1), tolerance);
  if (half)
  {
    for (int j = 1; j < pre.kf; ++j)
    {
      check_close("unweighted nonconstant face moment",
                  pre.face_moment_to_lebesgue[static_cast<std::size_t>(j)],
                  Real(0), tolerance);
    }
  }

  const Real intrinsic_error =
    test_intrinsic_face_monomials<D>(pre, tolerance);
  const Real mapped_error = test_mass_and_trace<D>(pre, tolerance);

  std::cout << "D=" << D
            << ", n=" << degree
            << ", kf=" << pre.kf
            << ", kappa=" << (half ? "half" : "mixed")
            << ", max_intrinsic_error=" << std::scientific
            << std::setprecision(6) << intrinsic_error
            << ", max_mapped_error=" << mapped_error
            << "  PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  // Arguments: [degree] [tolerance] [stencil_folder] [dimension].
  // dimension=0 tests D=1..4; otherwise tests just the given dimension.
  const int degree = (argc > 1) ? std::stoi(argv[1]) : 4;
  const Real tolerance = (argc > 2) ? std::stod(argv[2]) : 2.0e-10;
  const std::string stencil_folder =
    (argc > 3) ? argv[3] : "stencils";
  const int dimension = (argc > 4) ? std::stoi(argv[4]) : 0;

  if (degree < 2 || dimension < 0 || dimension > 4)
  {
    std::cerr << "Require degree >= 2 and dimension 0, 1, 2, 3, or 4\n";
    return 2;
  }

  try
  {
    if (dimension == 0 || dimension == 1)
    {
      run_case<1>(degree, false, tolerance, stencil_folder);
      run_case<1>(degree, true, tolerance, stencil_folder);
    }
    if (dimension == 0 || dimension == 2)
    {
      run_case<2>(degree, false, tolerance, stencil_folder);
      run_case<2>(degree, true, tolerance, stencil_folder);
    }
    if (dimension == 0 || dimension == 3)
    {
      run_case<3>(degree, false, tolerance, stencil_folder);
      run_case<3>(degree, true, tolerance, stencil_folder);
    }
    if (dimension == 0 || dimension == 4)
    {
      run_case<4>(degree, false, tolerance, stencil_folder);
      run_case<4>(degree, true, tolerance, stencil_folder);
    }
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "All reference Lebesgue face tests passed.\n";
  return 0;
}
