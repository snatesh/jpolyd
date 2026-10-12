#include <jneumann_root.hh>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

template<int D, class Real>
void check()
{
  const Real tol = std::is_same<Real,float>::value ? Real(2e-5) : Real(1e-12);

  // Column-major R, nonsymmetric, R * [1,1] = 0.
  const std::vector<Real> R = {Real(1), Real(-2), Real(-1), Real(2)};
  const std::vector<Real> v = {Real(1), Real(1)};
  const std::vector<Real> w = {Real(0.5), Real(0.5)};
  const auto plan = jsimplex::factor_neumann_root<D,Real>(R, v, w);

  std::vector<Real> x;
  jsimplex::solve_neumann_root(plan, {Real(-1), Real(2)}, Real(3), x);
  if (x.size() != 2 || std::abs(x[0] - Real(2.5)) > tol ||
      std::abs(x[1] - Real(3.5)) > tol)
    throw std::runtime_error("compatible solve failed");

  jsimplex::solve_neumann_root(plan, {Real(-1), Real(2)}, Real(5), x);
  if (std::abs(x[0] - Real(4.5)) > tol ||
      std::abs(x[1] - Real(5.5)) > tol)
    throw std::runtime_error("gauge shift failed");

  bool rejected = false;
  try
  {
    jsimplex::solve_neumann_root(plan, {Real(1), Real(2)}, Real(3), x);
  }
  catch (const std::runtime_error&)
  {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("incompatible data not rejected");

  rejected = false;
  try
  {
    jsimplex::factor_neumann_root<D,Real>(
      {Real(1), Real(0), Real(0), Real(1)}, v, w);
  }
  catch (const std::runtime_error&)
  {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("nonsingular matrix not rejected");

  // Three-by-three root with a nontrivial nonsymmetric left kernel.
  // Rows of R3 sum to zero, but columns need not sum to zero.
  const std::vector<Real> R3 = {
    Real(1), Real(0), Real(-1),
    Real(-1), Real(2), Real(0),
    Real(0), Real(-2), Real(1)
  };
  const std::vector<Real> v3 = {Real(1), Real(1), Real(1)};
  const std::vector<Real> w3 = {Real(0.2), Real(0.3), Real(0.5)};
  const std::vector<Real> x3 = {Real(1.2), Real(-0.7), Real(2.3)};
  std::vector<Real> h3(3, Real(0));
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      h3[(std::size_t)i] += R3[(std::size_t)i + 3 * j] * x3[(std::size_t)j];
  const Real eta3 = w3[0]*x3[0] + w3[1]*x3[1] + w3[2]*x3[2];
  const auto plan3 = jsimplex::factor_neumann_root<D,Real>(R3, v3, w3);
  std::vector<Real> recovered3;
  jsimplex::solve_neumann_root(plan3, h3, eta3, recovered3);
  for (int i = 0; i < 3; ++i)
    if (std::abs(recovered3[(std::size_t)i] - x3[(std::size_t)i]) > Real(4) * tol)
      throw std::runtime_error("three-by-three nonsymmetric solve failed");

  rejected = false;
  try
  {
    jsimplex::factor_neumann_root<D,Real>(
      {Real(0), Real(0), Real(0), Real(0)}, v, w);
  }
  catch (const std::runtime_error&)
  {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("multiple null modes not rejected");
}

int main()
{
  check<1,float>();
  check<2,float>();
  check<3,float>();
  check<1,double>();
  check<2,double>();
  check<3,double>();
  std::cout << "PASS: templated Neumann root (D=1..3, float/double)\n";
}
