#ifndef JLEAF_HH
#define JLEAF_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <jdetail.hh>
#include <jgeom.hh>
#include <jlaplace.hh>
#include <jperms.hh>
#include <jprecomp.hh>
#include <jlsmr.hh>

namespace jsimplex {

template<int D, class Real = double>
class Leaf
{
public:
  static_assert(D >= 1, "Leaf requires D>=1");
  static_assert(std::is_floating_point<Real>::value, "Leaf requires floating-point Real");
  static_assert(std::is_same_v<Real,float> || std::is_same_v<Real,double>,
                "Leaf currently supports only Real=float or Real=double");

  using LsmrOptions = detail::LsmrOptions<Real>;
  using LsmrInfo = detail::LsmrInfo<Real>;

  const RefSimplexPrecomp<D,Real>* pre = nullptr;

  int n = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int nface = D + 1;
  int nb = 0;
  int ntau_rows = 0;

  std::array<int,D + 1> global_vids{};

  // Geometry. V_phys is D x (D+1), column-major.
  std::vector<Real> V_phys;
  DSimplexGeom<D,Real> geom{};
  std::array<Real,D * D> G{}; // G = B^{-1} B^{-T}, column-major.

  // Face metadata and physical geometry.
  std::array<int,D + 1> face_sigma_index{};
  std::vector<Real> face_scale;       // physical face measure scale, length nface
  std::vector<Real> face_ref_scale;   // reference face measure scale, length nface
  std::vector<Real> face_h;           // max physical edge length per face, length nface
  std::vector<Real> unit_normal;      // row-major nface x D
  std::vector<Real> normal_scaled;    // row-major nface x D, face_scale * unit_normal

  // Physical coefficient maps, all column-major.
  //   L: m_int x M
  //   T: nb x M, trace moment map
  //   F: nb x M, physical outward flux moment map
  std::vector<Real> L;
  std::vector<Real> T;
  std::vector<Real> F;

  // Tau-balanced least-squares matrix:
  //   A_tau = [ sL * L ; sqrt(tau_f) * T_f ]
  // where each face block T_f uses its own tau_f.
  std::vector<Real> A_tau; // (m_int+nb) x M, column-major
  std::vector<Real> tau_face;
  std::vector<Real> tau_rows;
  std::vector<Real> sqrt_tau_rows;
  Real tau_C = Real(10);
  Real sL = Real(1);

  LsmrOptions lsmr_options{};

  Leaf() = default;

  Leaf(const RefSimplexPrecomp<D,Real>& pre_in,
              const Real* V_phys_colmajor,
              const int* global_vids_in,
              Real tau_C_in = Real(10),
              const LsmrOptions& opts = LsmrOptions())
  {
    reset(pre_in, V_phys_colmajor, global_vids_in, tau_C_in, opts);
  }

  void reset(const RefSimplexPrecomp<D,Real>& pre_in,
             const Real* V_phys_colmajor,
             const int* global_vids_in,
             Real tau_C_in = Real(10),
             const LsmrOptions& opts = LsmrOptions())
  {
    if (!V_phys_colmajor) { throw std::invalid_argument("Leaf: null V_phys"); }
    if (!global_vids_in) { throw std::invalid_argument("Leaf: null global_vids"); }

    pre = &pre_in;
    n = pre->n;
    M = pre->M;
    m_int = pre->m_int;
    kf = pre->kf;
    nface = D + 1;
    nb = nface * kf;
    ntau_rows = m_int + nb;
    tau_C = tau_C_in;
    lsmr_options = opts;

    V_phys.assign(V_phys_colmajor,
                  V_phys_colmajor + (std::size_t)D * (D + 1));
    for (int i = 0; i <= D; ++i) { global_vids[(std::size_t)i] = global_vids_in[i]; }

    build_geometry();
    assemble_LTF();
    build_tau_from_face_diameters(tau_C);
    build_A_tau();
  }

  int face_offset(int face_id) const
  {
    check_face_id(face_id);
    return face_id * kf;
  }

  std::array<int,D> face_key(int face_id) const
  {
    check_face_id(face_id);
    int fv[D];
    dsimplex_face_vertices<D>(face_id, fv);

    std::array<int,D> key{};
    for (int i = 0; i < D; ++i) { key[(std::size_t)i] = global_vids[(std::size_t)fv[i]]; }
    std::sort(key.begin(), key.end());
    return key;
  }

  void set_tau_face(const Real* tau_face_in)
  {
    if (!tau_face_in) { throw std::invalid_argument("Leaf::set_tau_face: null input"); }

    tau_face.assign((std::size_t)nface, Real(0));
    tau_rows.assign((std::size_t)nb, Real(0));
    sqrt_tau_rows.assign((std::size_t)nb, Real(0));

    for (int f = 0; f < nface; ++f)
    {
      if (!(tau_face_in[f] > Real(0)))
      {
        throw std::invalid_argument("Leaf::set_tau_face: tau must be positive");
      }
      tau_face[(std::size_t)f] = tau_face_in[f];
      const Real st = std::sqrt(tau_face_in[f]);
      for (int r = 0; r < kf; ++r)
      {
        const int row = f * kf + r;
        tau_rows[(std::size_t)row] = tau_face_in[f];
        sqrt_tau_rows[(std::size_t)row] = st;
      }
    }

    build_A_tau();
  }

  void build_tau_from_face_diameters(Real C)
  {
    if (!(C > Real(0))) { throw std::invalid_argument("Leaf: tau constant must be positive"); }

    tau_face.assign((std::size_t)nface, Real(0));
    tau_rows.assign((std::size_t)nb, Real(0));
    sqrt_tau_rows.assign((std::size_t)nb, Real(0));

    const Real np1 = Real(n + 1);
    for (int f = 0; f < nface; ++f)
    {
      const Real h = std::max(face_h[(std::size_t)f], Real(100) * std::numeric_limits<Real>::epsilon());
      const Real tau = C * np1 * np1 / h;
      tau_face[(std::size_t)f] = tau;
      const Real st = std::sqrt(tau);

      for (int r = 0; r < kf; ++r)
      {
        const int row = f * kf + r;
        tau_rows[(std::size_t)row] = tau;
        sqrt_tau_rows[(std::size_t)row] = st;
      }
    }
  }

  LsmrInfo apply(const Real* lambda,
                       const Real* f_int,
                       Real* c_out,
                       Real* trace_out,
                       Real* raw_flux_out,
                       Real* aug_flux_out,
                       Real* trace_mismatch_out = nullptr,
                       Real* pde_residual_out = nullptr) const
  {
    if (!lambda) { throw std::invalid_argument("Leaf::apply: null lambda"); }
    if (!f_int) { throw std::invalid_argument("Leaf::apply: null f_int"); }
    if (!c_out) { throw std::invalid_argument("Leaf::apply: null c_out"); }
    if (!trace_out) { throw std::invalid_argument("Leaf::apply: null trace_out"); }
    if (!raw_flux_out) { throw std::invalid_argument("Leaf::apply: null raw_flux_out"); }
    if (!aug_flux_out) { throw std::invalid_argument("Leaf::apply: null aug_flux_out"); }

    std::vector<Real> b((std::size_t)ntau_rows, Real(0));

    for (int i = 0; i < m_int; ++i)
    {
      b[(std::size_t)i] = -sL * f_int[i];
    }
    for (int r = 0; r < nb; ++r)
    {
      b[(std::size_t)m_int + r] = sqrt_tau_rows[(std::size_t)r] * lambda[r];
    }

    LsmrInfo info{};
    const int ret = lsmr_dense_solve_colmajor<Real>(
      ntau_rows,
      M,
      A_tau.data(),
      b.data(),
      c_out,
      lsmr_options,
      &info);

    if (ret != 0)
    {
      throw std::runtime_error("Leaf::apply: lsmr_dense_solve_colmajor failed");
    }

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      nb,
      1,
      M,
      Real(1),
      T.data(),
      nb,
      c_out,
      M,
      Real(0),
      trace_out,
      nb);

    detail::BlasGemm<Real>::run(
      CblasColMajor,
      CblasNoTrans,
      CblasNoTrans,
      nb,
      1,
      M,
      Real(1),
      F.data(),
      nb,
      c_out,
      M,
      Real(0),
      raw_flux_out,
      nb);

    for (int r = 0; r < nb; ++r)
    {
      const Real mismatch = trace_out[r] - lambda[r];
      if (trace_mismatch_out) { trace_mismatch_out[r] = mismatch; }
      aug_flux_out[r] = raw_flux_out[r] + tau_rows[(std::size_t)r] * mismatch;
    }

    if (pde_residual_out)
    {
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        m_int,
        1,
        M,
        Real(1),
        L.data(),
        m_int,
        c_out,
        M,
        Real(0),
        pde_residual_out,
        m_int);
      for (int i = 0; i < m_int; ++i) { pde_residual_out[i] += f_int[i]; }
    }

    return info;
  }

  LsmrInfo apply_zero_source(const Real* lambda,
                                   Real* c_out,
                                   Real* trace_out,
                                   Real* raw_flux_out,
                                   Real* aug_flux_out,
                                   Real* trace_mismatch_out = nullptr,
                                   Real* pde_residual_out = nullptr) const
  {
    std::vector<Real> f0((std::size_t)m_int, Real(0));
    return apply(lambda, f0.data(), c_out, trace_out, raw_flux_out,
                 aug_flux_out, trace_mismatch_out, pde_residual_out);
  }

private:
  void check_face_id(int face_id) const
  {
    if (face_id < 0 || face_id >= nface) { throw std::out_of_range("Leaf: face_id out of range"); }
  }

  void build_geometry()
  {
    dsimplex_affine_from_verts<D,Real>(V_phys.data(), geom);
    if (!geom.valid) { throw std::runtime_error("Leaf: singular affine simplex"); }

    for (int i = 0; i < D; ++i)
    {
      for (int j = 0; j < D; ++j)
      {
        Real acc = Real(0);
        for (int k = 0; k < D; ++k)
        {
          acc += geom.BinvT[(std::size_t)k + (std::size_t)D * i]
               * geom.BinvT[(std::size_t)k + (std::size_t)D * j];
        }
        G[(std::size_t)i + (std::size_t)D * j] = acc;
      }
    }

    for (int f = 0; f < nface; ++f)
    {
      face_sigma_index[(std::size_t)f] =
        dsimplex_compute_face_sigma<D>(global_vids.data(), f);
    }

    build_physical_face_geometry();
  }

  void build_physical_face_geometry()
  {
    face_scale.assign((std::size_t)nface, Real(0));
    face_ref_scale.assign((std::size_t)nface, Real(0));
    face_h.assign((std::size_t)nface, Real(0));
    unit_normal.assign((std::size_t)nface * D, Real(0));
    normal_scaled.assign((std::size_t)nface * D, Real(0));

    for (int f = 0; f < nface; ++f)
    {
      face_ref_scale[(std::size_t)f] = pre->face_ref_scale[(std::size_t)f];
    }

    if constexpr (D == 1)
    {
      const Real x0 = V_phys[0];
      const Real x1 = V_phys[1];
      const Real len = std::abs(x1 - x0);
      const Real sgn = (x1 >= x0) ? Real(1) : Real(-1);

      face_scale[0] = Real(1);
      face_scale[1] = Real(1);
      face_h[0] = len;
      face_h[1] = len;

      // face 0 is opposite vertex 0, i.e. endpoint vertex 1.
      unit_normal[0] = sgn;
      unit_normal[1] = -sgn;
      normal_scaled[0] = unit_normal[0];
      normal_scaled[1] = unit_normal[1];
    }
    else
    {
      for (int f = 0; f < nface; ++f)
      {
        int fv[D];
        dsimplex_face_vertices<D>(f, fv);

        Real Vface[D * D]; // D x D, columns are face vertices.
        for (int j = 0; j < D; ++j)
        {
          const int v = fv[j];
          for (int r = 0; r < D; ++r)
          {
            Vface[(std::size_t)r + (std::size_t)D * j] =
              V_phys[(std::size_t)r + (std::size_t)D * v];
          }
        }

        const Real s = dsimplex_embedded_simplex_measure_scale_colmajor<D,D-1,Real>(Vface);
        if (!(s > Real(0))) { throw std::runtime_error("Leaf: degenerate physical face"); }
        face_scale[(std::size_t)f] = s;

        Real h = Real(0);
        for (int a = 0; a < D; ++a)
        {
          for (int b = a + 1; b < D; ++b)
          {
            Real d2 = Real(0);
            for (int r = 0; r < D; ++r)
            {
              const Real diff = Vface[(std::size_t)r + (std::size_t)D * a]
                              - Vface[(std::size_t)r + (std::size_t)D * b];
              d2 += diff * diff;
            }
            h = std::max(h, std::sqrt(d2));
          }
        }
        face_h[(std::size_t)f] = h;

        std::array<Real,D> nvec{};
        compute_face_unit_normal_from_svd(Vface, nvec.data());

        const int opp = f;
        Real dot_to_opp = Real(0);
        for (int r = 0; r < D; ++r)
        {
          const Real p_opp = V_phys[(std::size_t)r + (std::size_t)D * opp];
          const Real p0 = Vface[(std::size_t)r];
          dot_to_opp += nvec[(std::size_t)r] * (p_opp - p0);
        }
        if (dot_to_opp > Real(0))
        {
          for (int r = 0; r < D; ++r) { nvec[(std::size_t)r] = -nvec[(std::size_t)r]; }
        }

        for (int r = 0; r < D; ++r)
        {
          unit_normal[(std::size_t)f * D + r] = nvec[(std::size_t)r];
          normal_scaled[(std::size_t)f * D + r] = s * nvec[(std::size_t)r];
        }
      }
    }
  }

  static void compute_face_unit_normal_from_svd(const Real* Vface, Real* n_out)
  {
    static_assert(D >= 2, "SVD normal helper only used for D>=2");
    if (!Vface || !n_out) { throw std::invalid_argument("Leaf: null normal helper input"); }

    constexpr int m = D - 1;
    constexpr int ncols = D;

    std::vector<Real> A((std::size_t)m * ncols, Real(0)); // A = E^T, col-major m x D.
    for (int col = 0; col < ncols; ++col)
    {
      for (int row = 0; row < m; ++row)
      {
        A[(std::size_t)row + (std::size_t)m * col] =
          Vface[(std::size_t)col + (std::size_t)D * (row + 1)]
        - Vface[(std::size_t)col + (std::size_t)D * 0];
      }
    }

    std::vector<Real> S((std::size_t)std::min(m, ncols), Real(0));
    std::vector<Real> U((std::size_t)m * m, Real(0));
    std::vector<Real> VT((std::size_t)ncols * ncols, Real(0));

    const lapack_int ret = detail::LapackGesdd<Real>::run(
      'A',
      (lapack_int)m,
      (lapack_int)ncols,
      A.data(),
      (lapack_int)m,
      S.data(),
      U.data(),
      (lapack_int)m,
      VT.data(),
      (lapack_int)ncols);

    if (ret != 0) { throw std::runtime_error("Leaf: SVD failed while computing face normal"); }

    Real nrm2 = Real(0);
    for (int r = 0; r < D; ++r)
    {
      const Real nr = VT[(std::size_t)(D - 1) + (std::size_t)D * r];
      n_out[r] = nr;
      nrm2 += nr * nr;
    }
    const Real nrm = std::sqrt(nrm2);
    if (!(nrm > Real(0))) { throw std::runtime_error("Leaf: zero face normal from SVD"); }
    for (int r = 0; r < D; ++r) { n_out[r] /= nrm; }
  }

  void assemble_LTF()
  {
    L.assign((std::size_t)m_int * M, Real(0));
    T.assign((std::size_t)nb * M, Real(0));
    F.assign((std::size_t)nb * M, Real(0));

    jdsimplex_assemble_L_int<D,Real>(
      M,
      m_int,
      G.data(),
      geom.detBabs,
      pre->Lij_ref.data(),
      L.data());

    for (int f = 0; f < nface; ++f)
    {
      const int sigma = face_sigma_index[(std::size_t)f];
      const Real ref_s = face_ref_scale[(std::size_t)f];
      if (!(ref_s > Real(0))) { throw std::runtime_error("Leaf: bad reference face scale"); }

      const Real trace_ratio = face_scale[(std::size_t)f] / ref_s;
      const int row0 = f * kf;

      for (int col = 0; col < M; ++col)
      {
        for (int r = 0; r < kf; ++r)
        {
          const std::size_t ref_idx = (std::size_t)r
            + (std::size_t)kf * ((std::size_t)col
            + (std::size_t)M * ((std::size_t)sigma
            + (std::size_t)pre->nsigma * (std::size_t)f));

          T[(std::size_t)(row0 + r) + (std::size_t)nb * col] =
            trace_ratio * pre->T_ref[ref_idx];
        }
      }

      Real eta[D];
      for (int a = 0; a < D; ++a)
      {
        Real ea = Real(0);
        for (int i = 0; i < D; ++i)
        {
          const Real ns_i = normal_scaled[(std::size_t)f * D + i] / ref_s;
          ea += geom.BinvT[(std::size_t)i + (std::size_t)D * a] * ns_i;
        }
        eta[a] = ea;
      }

      for (int col = 0; col < M; ++col)
      {
        for (int r = 0; r < kf; ++r)
        {
          Real acc = Real(0);
          for (int a = 0; a < D; ++a)
          {
            const std::size_t ref_idx = (std::size_t)r
              + (std::size_t)kf * ((std::size_t)col
              + (std::size_t)M * ((std::size_t)a
              + (std::size_t)D * ((std::size_t)sigma
              + (std::size_t)pre->nsigma * (std::size_t)f)));
            acc += eta[a] * pre->Fgrad_ref[ref_idx];
          }
          F[(std::size_t)(row0 + r) + (std::size_t)nb * col] = acc;
        }
      }
    }

    sL = row_rms_scale(L.data(), m_int, M);
  }

  void build_A_tau()
  {
    if (tau_rows.size() != (std::size_t)nb || sqrt_tau_rows.size() != (std::size_t)nb)
    {
      return;
    }

    A_tau.assign((std::size_t)ntau_rows * M, Real(0));

    for (int col = 0; col < M; ++col)
    {
      for (int row = 0; row < m_int; ++row)
      {
        A_tau[(std::size_t)row + (std::size_t)ntau_rows * col] =
          sL * L[(std::size_t)row + (std::size_t)m_int * col];
      }

      for (int row = 0; row < nb; ++row)
      {
        A_tau[(std::size_t)m_int + row + (std::size_t)ntau_rows * col] =
          sqrt_tau_rows[(std::size_t)row]
        * T[(std::size_t)row + (std::size_t)nb * col];
      }
    }
  }

  static Real row_rms_scale(const Real* A, int rows, int cols)
  {
    Real n2 = Real(0);
    const std::size_t N = (std::size_t)rows * cols;
    for (std::size_t i = 0; i < N; ++i) { n2 += A[i] * A[i]; }
    const Real rms = std::sqrt(n2 / std::max<std::size_t>(N, 1));
    const Real tiny = Real(1e-300);
    return Real(1) / std::max(tiny, rms);
  }
};

} // namespace jsimplex

#endif // JLEAF_HH
