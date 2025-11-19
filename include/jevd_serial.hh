#ifndef JEVD_SERIAL_H
#define JEVD_SERIAL_H

#include <vector>
#include <cmath>
#include <cstddef>

/*
   Serial JEV D (Joint Eigenvalue Decomposition) for L real symmetric
   N x N matrices, using the original "jointDiag" pattern you provided.

   Layout of J (AoS, column-major per matrix):

     - We have L symmetric N x N matrices.
     - Matrix ell (0 <= ell < L) is stored as:

         J_aos[ell * N*N + i + N * j]

       which is column-major in (i,j) for each matrix, and AoS over ell.

   The routine applies Jacobi-like plane rotations to jointly (approximately)
   diagonalize these L matrices by similarity transforms:

       J_ell <- R^T J_ell R,   ell = 0..L-1

   The rotation angles follow the Cardoso–Souloumiac direction, as in
   your original jointDiag<T> implementation.

   Interface matches the parallel version:

     namespace jevd {
       template<class Real>
       struct Params {
         int   max_sweeps;
         Real  tol;
         int   num_threads;   // ignored here
         bool  accumulate_V;
       };

       template<class Real>
       struct Result {
         int  sweeps;
         Real max_offdiag;
         int  info;           // 0=success, >0=no convergence in max_sweeps
       };

       template<class Real>
       Result<Real> joint_evd_symmetric(int N, int L,
                                        Real* J_aos,
                                        Real* V,
                                        const Params<Real>& params);
     }
*/

namespace jevd_serial
{

template<class Real>
struct Params
{
  int   max_sweeps;     // maximum sweeps over all (p,q)
  Real  tol;            // threshold on |s| for rotation
  int   num_threads;    // unused in serial version
  bool  accumulate_V;   // if true, update V as we go

  Params()
    : max_sweeps(50),
      tol(static_cast<Real>(1e-10)),
      num_threads(1),
      accumulate_V(true)
  {}
};

template<class Real>
struct Result
{
  int  sweeps;        // number of completed sweeps
  Real max_offdiag;   // max |s| in the last sweep
  int  info;          // 0 = success, >0 = did not converge in max_sweeps
};

/* Minimal absolute-value helper. */
template<class Real>
inline Real jevd_abs(Real x)
{
  return (x >= Real(0)) ? x : -x;
}

/*
  Serial JEV D, matching your original jointDiag<T> algorithm.

  N : matrix size (m in your code)
  L : number of matrices (n in your code)
  J_aos : pointer to AoS, column-major per matrix, length L * N * N
  V     : optional N x N matrix (column-major). If params.accumulate_V
          is true and V != nullptr, we initialize V = I and apply the
          same rotations so that J' ≈ V^T J V.

  params.tol        : used as the threshold on |s|
  params.max_sweeps : safety cap on the number of outer sweeps.
*/
template<class Real>
Result<Real> joint_evd_symmetric(int N,
                                 int L,
                                 Real* J_aos,
                                 Real* V,
                                 const Params<Real>& params)
{
  Result<Real> res;
  res.sweeps      = 0;
  res.max_offdiag = Real(0);
  res.info        = 0;

  if (N <= 1 || L <= 0 || !J_aos)
  {
    return res;
  }

  const unsigned int m  = static_cast<unsigned int>(N);
  const unsigned int nM = static_cast<unsigned int>(L);
  const unsigned int nm = m * nM;

  const Real thresh = params.tol;
  const int  max_sweeps = (params.max_sweeps > 0) ? params.max_sweeps : 50;

  bool hasV = params.accumulate_V && (V != nullptr);

  // Initialize V to identity if we accumulate it, as in your original "hasV".
  if (hasV)
  {
    for (unsigned int j = 0; j < m; ++j)
    {
      for (unsigned int i = 0; i < m; ++i)
      {
        V[i + m * j] = (i == j) ? Real(1) : Real(0);
      }
    }
  }

  // Work buffers, reused across the sweep to avoid repeated allocations.
  std::vector<Real> App(nM);
  std::vector<Real> Aqq(nM);
  std::vector<Real> Apq(nM);
  std::vector<Real> Aqp(nM);
  std::vector<Real> g1(nM);
  std::vector<Real> g2(nM);

  std::vector<Real> Mp(m);
  std::vector<Real> Mq(m);

  std::vector<Real> rowp(nm);
  std::vector<Real> rowq(nm);

  unsigned int iter = 0;
  bool go = true;

  while (go && static_cast<int>(iter) < max_sweeps)
  {
    go = false;
    ++iter;

    Real max_s_abs = Real(0);

    // p,q are 1-based in the original code; we keep that here.
    for (unsigned int p = 1; p <= m - 1; ++p)
    {
      for (unsigned int q = p + 1; q <= m; ++q)
      {
        // Gather App, Aqq, Apq, Aqp across all L matrices.
        // Original indexing: J[p-1 + m*(p-1 + nn*m)]
        for (unsigned int nn = 0; nn < nM; ++nn)
        {
          std::size_t base = static_cast<std::size_t>(nn) * m * m;

          App[nn] = J_aos[(p - 1) + m * ((p - 1) + base / m)];
          Aqq[nn] = J_aos[(q - 1) + m * ((q - 1) + base / m)];
          Apq[nn] = J_aos[(p - 1) + m * ((q - 1) + base / m)];
          Aqp[nn] = J_aos[(q - 1) + m * ((p - 1) + base / m)];
        }

        for (unsigned int nn = 0; nn < nM; ++nn)
        {
          g1[nn] = App[nn] - Aqq[nn];
          g2[nn] = Apq[nn] + Aqp[nn];
        }

        Real G00 = Real(0);
        Real G11 = Real(0);
        Real G01 = Real(0);

        for (unsigned int nn = 0; nn < nM; ++nn)
        {
          G00 += g1[nn] * g1[nn];
          G11 += g2[nn] * g2[nn];
          G01 += g1[nn] * g2[nn];
        }

        Real ton  = G00 - G11;
        Real toff = G01 * Real(2);

        Real denom = std::sqrt(ton * ton + toff * toff);
        if (denom == Real(0))
        {
          continue;
        }

        Real theta = Real(0.5) * std::atan2(toff, ton + denom);
        Real c = std::cos(theta);
        Real s = std::sin(theta);

        Real s_abs = jevd_abs(s);
        if (s_abs > max_s_abs)
        {
          max_s_abs = s_abs;
        }

        bool do_rotate = (s_abs > thresh);
        go = (go || do_rotate);

        if (!do_rotate)
        {
          continue;
        }

        // Apply column rotations for each matrix (p,q).
        for (unsigned int nn = 0; nn < nM; ++nn)
        {
          std::size_t base = static_cast<std::size_t>(nn) * m * m;

          for (unsigned int i = 0; i < m; ++i)
          {
            Mp[i] = J_aos[i + m * ((p - 1) + base / m)];
            Mq[i] = J_aos[i + m * ((q - 1) + base / m)];
          }

          for (unsigned int i = 0; i < m; ++i)
          {
            J_aos[i + m * ((p - 1) + base / m)] = c * Mp[i] + s * Mq[i];
            J_aos[i + m * ((q - 1) + base / m)] = c * Mq[i] - s * Mp[i];
          }
        }

        // Apply row rotations across all matrices.
        for (unsigned int i = 0; i < nm; ++i)
        {
          rowp[i] = J_aos[(p - 1) + m * i];
          rowq[i] = J_aos[(q - 1) + m * i];
        }

        for (unsigned int i = 0; i < nm; ++i)
        {
          J_aos[(p - 1) + m * i] = c * rowp[i] + s * rowq[i];
          J_aos[(q - 1) + m * i] = c * rowq[i] - s * rowp[i];
        }

        // Update V if requested.
        if (hasV)
        {
          Real* vp = V + (p - 1) * m;
          Real* vq = V + (q - 1) * m;

          for (unsigned int i = 0; i < m; ++i)
          {
            Real tmp = vp[i];
            vp[i] = c * vp[i] + s * vq[i];
            vq[i] = c * vq[i] - s * tmp;
          }
        }
      }
    }

    res.max_offdiag = max_s_abs;
  }

  res.sweeps = static_cast<int>(iter);
  if (go)
  {
    // We exited due to max_sweeps cap.
    res.info = 1;
  }
  else
  {
    res.info = 0;
  }

  return res;
}

} // namespace jevd

#endif // JEVD_SERIAL_H
