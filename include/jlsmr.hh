#ifndef JLSMR_HH
#define JLSMR_HH

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>
#include <jdetail.hh>
namespace jsimplex {

// Solve min_x ||A*x - b||_2 using dense column-major A.
//
// A_colmajor has shape m-by-n with A(i,j)=A_colmajor[i + m*j].
// The Fortran SOL LSMR core is double precision only.  For Real=double this
// calls it directly.  For other arithmetic Real types, A and b are copied to
// double, the solve is performed in double, and x/info are cast back to Real.
//
// Return codes:
//   0   wrapper success; inspect info.istop/info.stat for LSMR status
//  -1   invalid dimensions
//  -2   null required pointer
//  -3   allocation failure or C++ exception
//  -4   unexpected reverse-communication action
//
// The underlying Fortran shim currently uses singleton solver state, so this
// function is not thread-safe for concurrent solves.
template<class Real>
inline int lsmr_dense_solve_colmajor(
  int m,
  int n,
  const Real* A_colmajor,
  const Real* b,
  Real* x,
  const detail::LsmrOptions<Real>& options = detail::LsmrOptions<Real>(),
  detail::LsmrInfo<Real>* info = nullptr)
{
  static_assert(std::is_floating_point<Real>::value,
                "lsmr_dense_solve_colmajor requires a floating-point Real type");
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
              "lsmr_dense_solve_colmajor currently supports only Real=float or Real=double");

  if (m <= 0 || n <= 0)
  {
    return -1;
  }

  if (A_colmajor == nullptr || b == nullptr || x == nullptr)
  {
    return -2;
  }

  try
  {
    detail::LsmrOptions<double> opt_d = detail::to_double_options(options);
    detail::LsmrInfo<double> info_d;

    if constexpr (std::is_same_v<Real, double>)
    {
      const int ret = detail::lsmr_dense_solve_colmajor_double(
        m,
        n,
        A_colmajor,
        b,
        x,
        opt_d,
        &info_d);

      detail::copy_info_from_double(info_d, info);
      return ret;
    }
    else
    {
      const std::size_t nnz = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
      std::vector<double> A_d(nnz);
      std::vector<double> b_d(static_cast<std::size_t>(m));
      std::vector<double> x_d(static_cast<std::size_t>(n), 0.0);

      for (std::size_t k = 0; k < nnz; ++k)
      {
        A_d[k] = static_cast<double>(A_colmajor[k]);
      }
      for (int i = 0; i < m; ++i)
      {
        b_d[static_cast<std::size_t>(i)] = static_cast<double>(b[i]);
      }

      const int ret = detail::lsmr_dense_solve_colmajor_double(
        m,
        n,
        A_d.data(),
        b_d.data(),
        x_d.data(),
        opt_d,
        &info_d);

      for (int j = 0; j < n; ++j)
      {
        x[static_cast<std::size_t>(j)] = static_cast<Real>(x_d[static_cast<std::size_t>(j)]);
      }

      detail::copy_info_from_double(info_d, info);
      return ret;
    }
  }
  catch (...)
  {
    return -3;
  }
}

} // namespace jsimplex

#endif // JLSMR_HH
