#include <jprecomp.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
  const Real difference = std::abs(actual - expected);
  const Real scale = std::max(Real(1), std::abs(expected));
  if (!std::isfinite(actual) || difference > tolerance * scale)
  {
    std::ostringstream message;
    message << std::setprecision(17)
            << what << ": actual=" << actual
            << ", expected=" << expected
            << ", abs_error=" << difference
            << ", tolerance=" << tolerance * scale;
    throw std::runtime_error(message.str());
  }
}

template<int D>
std::array<Real, D + 1> make_kappa(bool unweighted)
{
  std::array<Real, D + 1> kappa{};
  for (int i = 0; i <= D; ++i)
  {
    kappa[static_cast<std::size_t>(i)] = unweighted
      ? Real(0.5)
      : Real(0.12 + 0.17 * i + 0.04 * (i % 2));
  }
  return kappa;
}

template<int D>
Real monomial(const Real* x, const std::array<int, D>& powers)
{
  Real result = Real(1);
  for (int axis = 0; axis < D; ++axis)
  {
    for (int k = 0; k < powers[static_cast<std::size_t>(axis)]; ++k)
    {
      result *= x[axis];
    }
  }
  return result;
}

template<int D>
Real exact_monomial_integral(const std::array<int, D>& powers)
{
  int degree = 0;
  Real numerator = Real(1);
  for (int axis = 0; axis < D; ++axis)
  {
    const int exponent = powers[static_cast<std::size_t>(axis)];
    degree += exponent;
    numerator *= factorial(exponent);
  }
  return numerator / factorial(D + degree);
}

template<int D>
std::vector<std::array<int, D>> make_monomials(int n)
{
  std::vector<std::array<int, D>> monomials;
  std::array<int, D> powers{};

  monomials.push_back(powers);  // 1
  powers[0] = 1;
  monomials.push_back(powers);  // x_0
  powers[0] = 2;
  monomials.push_back(powers);  // x_0^2
  powers[0] = n;
  monomials.push_back(powers);  // x_0^n: highest-degree test

  if constexpr (D >= 2)
  {
    powers.fill(0);
    powers[0] = 1;
    powers[1] = 1;
    monomials.push_back(powers); // x_0 x_1

    powers[0] = n - 1;
    monomials.push_back(powers); // x_0^(n-1) x_1
  }

  if constexpr (D >= 3)
  {
    powers.fill(0);
    powers[0] = 1;
    powers[1] = 1;
    powers[2] = 1;
    monomials.push_back(powers); // x_0 x_1 x_2
  }

  return monomials;
}

template<int D>
Real test_family(const Precomp<D>& pre,
                 bool residual,
                 Real tolerance)
{
  const std::vector<Real>& moments = residual
    ? pre.volume_lebesgue_res : pre.volume_lebesgue_base;
  const std::vector<Real>& X = residual ? pre.X_res : pre.X_vol;
  const std::vector<Real>& W = residual ? pre.W_res : pre.W_vol;
  const std::vector<Real>& V = residual ? pre.V_res : pre.V_vol;
  const char* family = residual ? "kappa+2" : "kappa";

  if (moments.size() != static_cast<std::size_t>(pre.M))
  {
    throw std::runtime_error(
      std::string(family) + ": integral vector must have M entries");
  }
  if (X.size() != static_cast<std::size_t>(pre.nq_vol) * D ||
      W.size() != static_cast<std::size_t>(pre.nq_vol) ||
      V.size() != static_cast<std::size_t>(pre.nq_vol) * pre.M)
  {
    throw std::runtime_error("Unexpected volume quadrature/basis layout");
  }

  for (Real value : moments)
  {
    if (!std::isfinite(value))
    {
      throw std::runtime_error("Lebesgue integration vector contains NaN/Inf");
    }
  }
  check_close(std::string(family) + " constant mode",
              moments.front(), Real(1) / factorial(D), tolerance);

  // Orthogonal projection in the *family's own weighted measure*:
  // c_j = integral P_j^(family)(x) x^alpha dmu_family(x).
  // The new m^(family) must then satisfy
  //       m^T c = integral_{T^D} x^alpha dx,
  // where the right side is an independent exact factorial formula.
  // All monomials have degree <= n; the full residual degree is n.
  Real largest_error = Real(0);
  for (const auto& powers : make_monomials<D>(pre.n))
  {
    std::vector<Real> coefficients(
      static_cast<std::size_t>(pre.M), Real(0));
    for (int j = 0; j < pre.M; ++j)
    {
      Real coefficient = Real(0);
      for (int iq = 0; iq < pre.nq_vol; ++iq)
      {
        const Real* x = X.data() + static_cast<std::size_t>(iq) * D;
        coefficient +=
          W[static_cast<std::size_t>(iq)] *
          V[static_cast<std::size_t>(iq) +
            static_cast<std::size_t>(pre.nq_vol) * j] *
          monomial<D>(x, powers);
      }
      coefficients[static_cast<std::size_t>(j)] = coefficient;
    }

    Real integrated = Real(0);
    for (int j = 0; j < pre.M; ++j)
    {
      integrated += moments[static_cast<std::size_t>(j)] *
                    coefficients[static_cast<std::size_t>(j)];
    }

    const Real expected = exact_monomial_integral<D>(powers);
    const Real difference = std::abs(integrated - expected);
    largest_error = std::max(largest_error, difference);
    check_close(std::string(family) + " monomial integral",
                integrated, expected, tolerance);
  }
  return largest_error;
}

template<int D>
void run_case(int degree,
              bool unweighted,
              Real tolerance,
              const std::string& stencil_folder)
{
  const auto kappa = make_kappa<D>(unweighted);
  const Precomp<D> pre(
    degree, 3, 0, 0, kappa.data(), stencil_folder);

  if (pre.M != jsimplex::Basis<D, Real>::dim_Pi(degree))
  {
    throw std::runtime_error("Full-degree M is incorrect");
  }
  for (int j = 0; j <= D; ++j)
  {
    check_close("residual family shift",
                pre.kappa_res[static_cast<std::size_t>(j)],
                kappa[static_cast<std::size_t>(j)] + Real(2),
                tolerance);
  }

  const Real base_error = test_family<D>(pre, false, tolerance);
  const Real res_error = test_family<D>(pre, true, tolerance);

  // With kappa == 1/2, the source basis is orthonormal with respect
  // to normalized unweighted measure. All nonconstant moments vanish.
  if (unweighted)
  {
    for (int j = 1; j < pre.M; ++j)
    {
      check_close("unweighted nonconstant basis moment",
                  pre.volume_lebesgue_base[static_cast<std::size_t>(j)],
                  Real(0), tolerance);
    }
  }

  std::cout << "D=" << D
            << ", n=" << degree
            << ", M=" << pre.M
            << ", kappa=" << (unweighted ? "half" : "mixed")
            << ", max_base_error=" << std::scientific << base_error
            << ", max_res_error=" << res_error
            << "  PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
  // Command line: [degree] [tolerance] [stencil_folder] [dimension].
  // dimension=0 runs all dimensions 1..4; otherwise runs that one dimension.
  const int degree = (argc > 1) ? std::stoi(argv[1]) : 4;
  const Real tolerance = (argc > 2) ? std::stod(argv[2]) : 2.0e-10;
  const std::string stencil_folder =
    (argc > 3) ? argv[3] : "stencils";
  const int dimension = (argc > 4) ? std::stoi(argv[4]) : 0;

  if (dimension < 0 || dimension > 4)
  {
    std::cerr << "dimension must be 0 (all), 1, 2, 3, or 4\n";
    return 2;
  }

  if (degree < 3)
  {
    std::cerr << "degree must be >= 3 for the cubic monomial test\n";
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
  catch (const std::exception& exception)
  {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }

  std::cout << "All reference Lebesgue volume tests passed.\n";
  return 0;
}
