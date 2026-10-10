
#ifndef JDIVERGENCE_HH
#define JDIVERGENCE_HH

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstddef>

#include <jelliptic.hh>

namespace jsimplex {

/*
  Determine the expanded non-divergence coefficient
  degrees corresponding to

    -div(A grad u) + b.grad(u) + c*u = f.

  All input polynomial degrees are <= n.
  A is required; b and c may be absent (degree -1).
*/
inline EllipticDegreeSpec
divergence_form_expanded_degrees(
  int n,
  int p_A,
  int p_b,
  int p_c)
{
  if (n < 2)
    throw std::invalid_argument(
      "divergence form: require n >= 2");

  if (p_A < 0 || p_A > n ||
      p_b < -1 || p_b > n ||
      p_c < -1 || p_c > n)
    throw std::invalid_argument(
      "divergence form: invalid coefficient degree");

  EllipticDegreeSpec degrees;

  degrees.p2 = p_A;
  degrees.p1 = std::max(p_b, p_A - 1);
  degrees.p0 = p_c;

  return degrees;
}

template<int D, class Real>
struct DivergenceFormExpandedCoefficients
{
  EllipticDegreeSpec degrees{};

  std::vector<Real> A;
  std::vector<Real> b;
  std::vector<Real> c;

  EllipticElementCoefficientsView<D,Real> view() const
  {
    return {
      A.empty() ? nullptr : A.data(),
      b.empty() ? nullptr : b.data(),
      c.empty() ? nullptr : c.data()
    };
  }
};


/*
  Convert divergence-form coefficients

    -div(A grad u) + b.grad(u) + c*u = f

  to the expanded nondivergence-form coefficients

    a_rs = -A_rs
    btilde_s = b_s - sum_r d_{x_r} A_rs
    ctilde = c.

  Input:
    Coefficients are in the base kappa family.
    A: D*D*dimPi(p_A), physical component indices (r,s)
    b: D*dimPi(p_b), optional when p_b == -1
    c: dimPi(p_c), optional when p_c == -1

  Output:
    Coefficients are in kappa_res = kappa + 2,
    packed to the expanded polynomial degrees.

  All coefficient degrees must be <= pre.n.
*/
template<int D, class Real>
DivergenceFormExpandedCoefficients<D,Real>
expand_divergence_form_element(
  const RefSimplexPrecomp<D,Real>& pre,
  const DSimplexGeom<D,Real>& geom,
  int p_A,
  const Real* A,
  int p_b,
  const Real* b,
  int p_c,
  const Real* c)
{
  const auto degrees = divergence_form_expanded_degrees(
    pre.n, p_A, p_b, p_c);

  if (!geom.valid)
    throw std::invalid_argument(
      "expand_divergence_form_element: invalid geometry");

  if (!A || (p_b >= 0 && !b) || (p_c >= 0 && !c))
    throw std::invalid_argument(
      "expand_divergence_form_element: null coefficients");

  const int M = pre.M;
  const int MpA = Basis<D,Real>::dim_Pi(p_A);
  const int Mp2 = Basis<D,Real>::dim_Pi(degrees.p2);
  const int Mp1 = degrees.p1 >= 0
    ? Basis<D,Real>::dim_Pi(degrees.p1) : 0;
  const int Mp0 = degrees.p0 >= 0
    ? Basis<D,Real>::dim_Pi(degrees.p0) : 0;

  const int MpAd = p_A >= 1
    ? Basis<D,Real>::dim_Pi(p_A - 1) : 0;

  DivergenceFormExpandedCoefficients<D,Real> out;
  out.degrees = degrees;

  out.A.assign((std::size_t)D * D * Mp2, Real(0));
  out.b.assign((std::size_t)D * Mp1, Real(0));
  out.c.assign((std::size_t)Mp0, Real(0));

  const Real* L0 = pre.L0_ref.data();

  // Principal part and divergence contribution:
  //
  //   a_rs = -A_rs
  //   btilde_s -= sum_r d_{x_r} A_rs.
  //
  // Physical differentiation:
  //
  //   d_{x_r} = sum_a BinvT(r,a) d_{xhat_a}.

  for (int r = 0; r < D; ++r)
  {
    for (int s = 0; s < D; ++s)
    {
      const Real* A_rs =
        A + ((std::size_t)r * D + s) * MpA;

      Real* a_rs =
        out.A.data() + ((std::size_t)r * D + s) * Mp2;

      Real* b_s = Mp1 > 0
        ? out.b.data() + (std::size_t)s * Mp1
        : nullptr;

      for (int j = 0; j < MpA; ++j)
      {
        const Real coeff = A_rs[j];

        if (coeff == Real(0))
          continue;

        // Promote A_rs from kappa to kappa_res.
        const Real* L0_col =
          L0 + (std::size_t)M * j;

        for (int i = 0; i < Mp2; ++i)
          a_rs[i] -= L0_col[i] * coeff;

        // Differentiate and promote into kappa_res.
        for (int a = 0; a < D && p_A >= 1; ++a)
        {
          const Real scale =
            -geom.BinvT[r + D * a] * coeff;

          const Real* Li_col =
            pre.Li_ref.data()
            + (std::size_t)a * M * M
            + (std::size_t)M * j;

          for (int i = 0; i < MpAd; ++i)
            b_s[i] += scale * Li_col[i];
        }
      }
    }
  }

  // Add the original first-order coefficient b_s.

  if (p_b >= 0)
  {
    const int MpB = Basis<D,Real>::dim_Pi(p_b);

    for (int s = 0; s < D; ++s)
    {
      const Real* b_s = b + (std::size_t)s * MpB;
      Real* bout_s = out.b.data() + (std::size_t)s * Mp1;

      for (int j = 0; j < MpB; ++j)
      {
        const Real coeff = b_s[j];
        const Real* L0_col = L0 + (std::size_t)M * j;

        for (int i = 0; i < MpB; ++i)
          bout_s[i] += L0_col[i] * coeff;
      }
    }
  }

  // Promote the zero-order coefficient c.

  if (p_c >= 0)
  {
    const int MpC = Basis<D,Real>::dim_Pi(p_c);

    for (int j = 0; j < MpC; ++j)
    {
      const Real coeff = c[j];
      const Real* L0_col = L0 + (std::size_t)M * j;

      for (int i = 0; i < MpC; ++i)
        out.c[i] += L0_col[i] * coeff;
    }
  }

  return out;
}


} // namespace jsimplex

#endif // JDIVERGENCE_HH

