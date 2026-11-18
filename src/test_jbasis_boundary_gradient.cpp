#include <jbasis.hh>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

template<int D>
using Point = std::array<double, D>;

// Generalized binomial coefficient, with the Jacobi parameters included.
double binomial(double q, int r) {
  return std::exp(std::lgamma(q + 1) - std::lgamma(r + 1)
                  - std::lgamma(q - r + 1));
}

// Evaluate a basis polynomial and its gradient using the explicit
// polynomial expansion of each collapsed Jacobi factor. This provides
// a reference independent of the recurrence used in jbasis.hh.
template<int D>
std::pair<double, Point<D>> reference_basis(
    const Point<D>& x, const double* kappa, const int* a,
    const int* t, double normalization) {
  Point<D> factors{};
  Point<D> df_dx{};
  Point<D> df_domega{};

  double omega = 1.0;

  for (int k = 0; k < D; ++k) {
    const int degree = a[k];

    double tail = 0.0;
    for (int r = k + 1; r <= D; ++r) {
      tail += kappa[r];
    }

    const double alpha = 2.0 * t[k] + tail + 0.5 * (D - k - 2);
    const double beta = kappa[k] - 0.5;
    const double v = x[k] - omega;

    double factor = 0.0;
    double dx = 0.0;
    double domega = 0.0;

    for (int r = 0; r <= degree; ++r) {
      const int p = degree - r;
      const double c = binomial(degree + alpha, r)
                       * binomial(degree + beta, degree - r);

      factor += c * std::pow(v, p) * std::pow(x[k], r);

      if (p > 0) {
        const double term = c * p * std::pow(v, p - 1)
                            * std::pow(x[k], r);
        dx += term;
        domega -= term;
      }

      if (r > 0) {
        dx += c * r * std::pow(v, p) * std::pow(x[k], r - 1);
      }
    }

    factors[k] = factor;
    df_dx[k] = dx;
    df_domega[k] = domega;
    omega -= x[k];
  }

  double value = normalization;
  for (int k = 0; k < D; ++k) {
    value *= factors[k];
  }

  Point<D> gradient{};
  for (int j = 0; j < D; ++j) {
    double sum = 0.0;

    for (int k = j; k < D; ++k) {
      double product = 1.0;
      for (int ell = 0; ell < D; ++ell) {
        if (ell != k) {
          product *= factors[ell];
        }
      }

      const double derivative = (k == j) ? df_dx[k] : -df_domega[k];
      sum += product * derivative;
    }

    gradient[j] = normalization * sum;
  }

  return {value, gradient};
}

template<int D>
void test(const std::vector<Point<D>>& points, int n, double base = 0.71) {
  using Basis = jsimplex::Basis<D, double>;

  double kappa[D + 1];
  for (int j = 0; j <= D; ++j) {
    kappa[j] = base + 0.17 * j;
  }

  std::vector<int> a, t;
  std::vector<double> h;
  Basis::build_structures(n, kappa, a, t, h);

  const int M = Basis::dim_Pi(n);
  const int P = static_cast<int>(points.size());

  std::vector<double> x(P * D);
  std::vector<double> values(P * M);
  std::vector<double> gradients(P * M * D);

  for (int p = 0; p < P; ++p) {
    for (int j = 0; j < D; ++j) {
      x[p * D + j] = points[p][j];
    }
  }

  Basis::eval_all(x.data(), D, 1, P, kappa, n,
                  a.data(), t.data(), h.data(),
                  values.data(), P, gradients.data());

  constexpr double tolerance = 1e-7;

  for (int p = 0; p < P; ++p) {
    for (int m = 0; m < M; ++m) {
      const int index = p + m * P;
      const auto expected = reference_basis<D>(
          points[p], kappa, &a[m * D], &t[m * D], h[m]);

      const double value = values[index];
      if (!std::isfinite(value)) {
        throw std::runtime_error("nonfinite basis value");
      }

      if (std::abs(expected.first - value)
          > tolerance * (1.0 + std::abs(value))) {
        std::cerr << "value mismatch: D=" << D << " p=" << p
                  << " m=" << m << " value=" << value
                  << " expected=" << expected.first << '\n';
        throw std::runtime_error("value mismatch");
      }

      for (int j = 0; j < D; ++j) {
        const double gradient = gradients[index * D + j];
        if (!std::isfinite(gradient)) {
          throw std::runtime_error("nonfinite basis gradient");
        }

        const double error = std::abs(expected.second[j] - gradient);
        if (error > tolerance * (1.0 + std::abs(gradient))) {
          std::cerr << "gradient mismatch: D=" << D << " p=" << p
                    << " m=" << m << " j=" << j
                    << " gradient=" << gradient
                    << " expected=" << expected.second[j]
                    << " error=" << error << '\n';
          throw std::runtime_error("gradient mismatch");
        }
      }
    }
  }

  std::cout << "PASS D=" << D << " n=" << n
            << " points=" << P << " polys=" << M << '\n';
}

int main() {
  test<1>({{0.0}, {1.0}, {1.0 - 1e-15}, {0.5}}, 7, 0.0);
  test<1>({{0.0}, {1.0}, {1.0 - 1e-15}, {0.5}}, 7, 0.5);

  test<2>({
    {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {0.2, 0.8},
    {1.0 - 1e-15, 0.0}, {0.25, 0.35}, {0.0, 0.0}
  }, 7, 0.0);
  test<2>({
    {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {0.2, 0.8},
    {1.0 - 1e-15, 0.0}, {0.25, 0.35}, {0.0, 0.0}
  }, 7, 0.71);

  test<3>({
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
    {0.5, 0.5, 0.0}, {0.2, 0.3, 0.5}, {1.0 - 1e-15, 0.0, 0.0},
    {0.15, 0.22, 0.31}, {0.0, 0.0, 0.0}
  }, 6, 0.0);
  test<3>({
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
    {0.5, 0.5, 0.0}, {0.2, 0.3, 0.5}, {1.0 - 1e-15, 0.0, 0.0},
    {0.15, 0.22, 0.31}, {0.0, 0.0, 0.0}
  }, 6, 0.71);

  test<4>({
    {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0},
    {0.25, 0.25, 0.25, 0.25}, {0.10, 0.20, 0.15, 0.30}
  }, 5, 0.0);
}
