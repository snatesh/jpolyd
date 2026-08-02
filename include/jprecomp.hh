#ifndef JPRECOMP_HH
#define JPRECOMP_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <jbasis.hh>
#include <jquad_tprod.hh>
#include <jkmat.hh>
#include <jdmat.hh>
#include <jgeom.hh>
#include <jperms.hh>
#include <jdetail.hh>

namespace jsimplex {

template<int D, class Real>
class RefSimplexPrecomp
{
public:
  static_assert(D >= 1, "RefSimplexPrecomp requires D>=1");

  struct CSCBlock
  {
    int rows = 0;
    int cols = 0;
    std::vector<int> colptr;
    std::vector<int> rowind;
    std::vector<Real> values;

    std::size_t nnz() const { return values.size(); }
  };

  // Degree-n CSC structure generated from a degree-independent DMat/KMat
  // delta stencil. Numerical values are filled separately for each shifted
  // factor. Equal signatures are interned automatically within each operator
  // family, so multiple axes/parameters may reference the same structure.
  struct SharedOperatorStructure
  {
    int rows = 0;
    int cols = 0;
    std::string signature;
    std::vector<int> colptr;
    std::vector<int> rowind;

    std::size_t nnz() const { return rowind.size(); }
  };

  int n = 0;
  int q_pad = 2;
  int q_vol = 0;
  int q_face = 0;
  int M = 0;
  int m_int = 0;
  int kf = 0;
  int nface = D + 1;
  int nsigma = factorial_int(D);
  int nq_vol = 0;
  int nq_face = 0;

  std::array<Real, D + 1> kappa{};
  std::array<Real, D + 1> kappa_res{};
  std::array<Real, D> kappa_face{};

  // Persistent stencil cache directory used by DMat/KMat. Missing keyed
  // stencils are discovered once and written here; subsequent precomputes
  // load them without running stencil stabilization.
  std::string stencil_folder = "stencils";

  // Interned finite-degree structures and key-to-structure maps.
  std::vector<SharedOperatorStructure> d_structures;
  std::vector<SharedOperatorStructure> k_structures;
  std::array<int, D> d_axis_structure{};
  std::array<int, D + 1> k_parameter_structure{};

  // Volume quadrature/basis in source-kappa convention.
  std::vector<Real> X_vol;     // row-major nq_vol x D (for basis evaluator)
  std::vector<Real> W_vol;     // nq_vol
  std::vector<Real> V_vol;     // col-major nq_vol x M
  std::vector<Real> V_vol_int; // col-major nq_vol x m_int

  // Volume quadrature/basis in PDE-residual kappa+2 convention.
  // The full degree-n basis is stored so the same data can test/build
  // second-, first-, and zero-order residual operators.
  std::vector<Real> X_res;      // row-major nq_vol x D
  std::vector<Real> W_res;      // nq_vol
  std::vector<Real> V_res;      // col-major nq_vol x M

  // Canonical face quadrature/basis. For D=1, Y_face is empty, W_face=[1], V_face=[1].
  std::vector<Real> Y_face;    // row-major nq_face x (D-1) (for basis evaluator)
  std::vector<Real> W_face;    // nq_face, intrinsic canonical face weights
  std::vector<Real> V_face;    // col-major nq_face x kf

  // Embedded reference-face measure scale from canonical (D-1)-simplex to the
  // corresponding face of the reference D-simplex.
  std::vector<Real> face_ref_scale; // nface

  // Dense reference operator blocks, all Fortran/column-major in public layout.
  // Lij_ref:    M  x M x D x D
  // Li_ref:     M  x M x D
  // L0_ref:     M  x M
  // T_ref:      kf x M x nsigma x nface
  // Fgrad_ref:  kf x M x D x nsigma x nface
  // Mface_ref:  kf x kf x nface
  std::vector<Real> Lij_ref;
  std::vector<Real> Li_ref;
  std::vector<Real> L0_ref;
  std::vector<Real> T_ref;
  std::vector<Real> Fgrad_ref;
  std::vector<Real> Mface_ref;

  // Pruned CSC copies of the boundary reference blocks.  Dense storage and
  // all existing dense interfaces remain unchanged.
  //
  // T_ref_csc block order:       sigma + nsigma * face
  // Fgrad_ref_csc block order:   axis + D * (sigma + nsigma * face)
  // Mface_ref_csc block order:   face
  std::vector<CSCBlock> T_ref_csc;
  std::vector<CSCBlock> Fgrad_ref_csc;
  std::vector<CSCBlock> Mface_ref_csc;

  // Relative Frobenius threshold used only for the boundary CSC copies.
  Real boundary_csc_rel_prune = Real(1.0e-12);

  const CSCBlock& T_ref_csc_block(int sigma, int face) const
  {
    if (sigma < 0 || sigma >= nsigma || face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: T_ref_csc block index");
    }
    return T_ref_csc[(std::size_t)sigma + (std::size_t)nsigma * face];
  }

  const CSCBlock& Fgrad_ref_csc_block(int axis, int sigma, int face) const
  {
    if (axis < 0 || axis >= D || sigma < 0 || sigma >= nsigma ||
        face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: Fgrad_ref_csc block index");
    }
    return Fgrad_ref_csc[(std::size_t)axis + (std::size_t)D *
      ((std::size_t)sigma + (std::size_t)nsigma * face)];
  }

  const CSCBlock& Mface_ref_csc_block(int face) const
  {
    if (face < 0 || face >= nface)
    {
      throw std::out_of_range("RefSimplexPrecomp: Mface_ref_csc block index");
    }
    return Mface_ref_csc[(std::size_t)face];
  }

  RefSimplexPrecomp(int n_in,
                    int q_pad_in,
                    int q_vol_in,
                    int q_face_in,
                    const Real* kappa_in,
                    const std::string& stencil_folder_in = "stencils")
  {
    if (!kappa_in) { throw std::invalid_argument("RefSimplexPrecomp: null kappa"); }
    if (n_in < 2) { throw std::invalid_argument("RefSimplexPrecomp: require n>=2"); }

    n = n_in;
    stencil_folder = stencil_folder_in.empty()
      ? std::string("stencils")
      : stencil_folder_in;
    q_pad = (q_pad_in > 0) ? q_pad_in : 2;
    q_vol = (q_vol_in > 0) ? q_vol_in : (n + q_pad);
    q_face = (q_face_in > 0) ? q_face_in : q_vol;
    if (q_face > q_vol) { throw std::invalid_argument("RefSimplexPrecomp: require q_face <= q_vol"); }

    for (int i = 0; i <= D; ++i)
    {
      kappa[(std::size_t)i] = kappa_in[i];
      kappa_res[(std::size_t)i] = kappa_in[i] + Real(2);
    }

    // Current common-face convention: all faces use source kappa[0:D].
    dsimplex_common_face_kappa<D,Real>(kappa.data(), kappa_face.data());

    M = Basis<D,Real>::dim_Pi(n);
    m_int = Basis<D,Real>::dim_Pi(n - 2);
    if constexpr (D == 1)
    {
      kf = 1;
      nq_face = 1;
    }
    else
    {
      kf = Basis<D - 1,Real>::dim_Pi(n);
      nq_face = (int)QuadMapped<D - 1,Real>::npoints((unsigned int)q_face);
    }
    nq_vol = (int)QuadMapped<D,Real>::npoints((unsigned int)q_vol);

    build_volume_data();
    build_residual_data();
    build_face_data();
    build_operator_stencil_data();
    build_derivative_factor_data();
    build_second_partials();
    build_first_partials();
    build_zero_partials();
    build_face_operator_data();
    build_face_sparse_data();
  }

private:
  // Stage-A numerical factors: dense row-major materializations assembled
  // from shared stencil structures. These preserve existing public dense
  // operators while eliminating repeated stencil discovery.
  std::vector<Real> D1_factor_rm;
  std::vector<Real> D2_factor_rm;
  std::vector<std::array<Real, D + 1>> derivative_kappa_1;
  std::map<std::array<int, D + 1>, std::vector<Real>>
    residual_promotion_cache;

  void build_volume_data()
  {
    X_vol.assign((std::size_t)nq_vol * D, (Real)0);
    W_vol.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa.data(), X_vol.data(), W_vol.data());
    if (built != nq_vol) { throw std::runtime_error("RefSimplexPrecomp: volume quadrature build failed"); }

    std::vector<int> alpha, tail;
    std::vector<Real> invh;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha, tail, invh);

    V_vol.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n,
      alpha.data(), tail.data(), invh.data(), V_vol.data(), nq_vol, nullptr);

    std::vector<int> alpha_i, tail_i;
    std::vector<Real> invh_i;
    Basis<D,Real>::build_structures(n - 2, kappa.data(), alpha_i, tail_i, invh_i);

    V_vol_int.assign((std::size_t)nq_vol * m_int, (Real)0);
    Basis<D,Real>::eval_all(
      X_vol.data(), D, 1, nq_vol, kappa.data(), n - 2,
      alpha_i.data(), tail_i.data(), invh_i.data(), V_vol_int.data(), nq_vol, nullptr);
  }

  void build_residual_data()
  {
    X_res.assign((std::size_t)nq_vol * D, (Real)0);
    W_res.assign((std::size_t)nq_vol, (Real)0);
    const int built = QuadMapped<D,Real>::build_kappa(
      (unsigned int)q_vol, kappa_res.data(), X_res.data(), W_res.data());
    if (built != nq_vol)
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: residual quadrature build failed");
    }

    std::vector<int> alpha_res, tail_res;
    std::vector<Real> invh_res;
    Basis<D,Real>::build_structures(
      n, kappa_res.data(), alpha_res, tail_res, invh_res);

    V_res.assign((std::size_t)nq_vol * M, (Real)0);
    Basis<D,Real>::eval_all(
      X_res.data(), D, 1, nq_vol, kappa_res.data(), n,
      alpha_res.data(), tail_res.data(), invh_res.data(),
      V_res.data(), nq_vol, nullptr);
  }

  void build_face_data()
  {
    if constexpr (D == 1)
    {
      Y_face.clear();
      W_face.assign(1, (Real)1);
      V_face.assign(1, (Real)1);
    }
    else
    {
      Y_face.assign((std::size_t)nq_face * (D - 1), (Real)0);
      W_face.assign((std::size_t)nq_face, (Real)0);
      const int built = QuadMapped<D - 1,Real>::build_kappa(
        (unsigned int)q_face, kappa_face.data(), Y_face.data(), W_face.data());
      if (built != nq_face) { throw std::runtime_error("RefSimplexPrecomp: face quadrature build failed"); }

      std::vector<int> alpha_f, tail_f;
      std::vector<Real> invh_f;
      Basis<D - 1,Real>::build_structures(n, kappa_face.data(), alpha_f, tail_f, invh_f);

      V_face.assign((std::size_t)nq_face * kf, (Real)0);
      Basis<D - 1,Real>::eval_all(
        Y_face.data(), D - 1, 1, nq_face, kappa_face.data(), n,
        alpha_f.data(), tail_f.data(), invh_f.data(), V_face.data(), nq_face, nullptr);
    }

    face_ref_scale.assign((std::size_t)nface, (Real)0);
    for (int face_id = 0; face_id < nface; ++face_id)
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);
      face_ref_scale[(std::size_t)face_id] =
        dsimplex_reference_face_scale_from_vertex_ids<D,Real>(fv);
    }
  }

  static void copy_csc_raw_structure(
    int rows,
    int cols,
    const std::string& signature,
    int* colptr_raw,
    int* rowind_raw,
    std::size_t nnz,
    SharedOperatorStructure& out)
  {
    std::unique_ptr<int, decltype(&std::free)> colptr_guard(
      colptr_raw, &std::free);
    std::unique_ptr<int, decltype(&std::free)> rowind_guard(
      rowind_raw, &std::free);

    out.rows = rows;
    out.cols = cols;
    out.signature = signature;
    out.colptr.assign(
      colptr_raw,
      colptr_raw + static_cast<std::size_t>(cols) + 1);
    if (nnz > 0)
    {
      out.rowind.assign(rowind_raw, rowind_raw + nnz);
    }
  }

  void build_operator_stencil_data()
  {
    d_axis_structure.fill(-1);
    k_parameter_structure.fill(-1);
    d_structures.clear();
    k_structures.clear();

    std::map<std::string, int> d_signature_index;
    std::map<std::string, int> k_signature_index;

    const int d_rows = Basis<D,Real>::dim_Pi(n - 1);

    for (int axis = 0; axis < D; ++axis)
    {
      DMatStencil stencil{};
      DMat<D,Real>::load_or_discover_natural_stencil(
        static_cast<unsigned int>(q_vol),
        kappa.data(),
        axis,
        &stencil,
        stencil_folder);

      const std::string signature =
        DMat<D,Real>::stencil_signature(stencil);
      const auto found = d_signature_index.find(signature);
      if (found != d_signature_index.end())
      {
        d_axis_structure[static_cast<std::size_t>(axis)] = found->second;
        stencil.clear();
        continue;
      }

      int* colptr_raw = nullptr;
      int* rowind_raw = nullptr;
      const std::size_t nnz =
        DMat<D,Real>::build_natural_csc_pattern_from_stencil(
          n, stencil, &colptr_raw, &rowind_raw);

      SharedOperatorStructure structure;
      copy_csc_raw_structure(
        d_rows, M, signature,
        colptr_raw, rowind_raw, nnz, structure);

      const int index = static_cast<int>(d_structures.size());
      d_structures.push_back(std::move(structure));
      d_signature_index.emplace(signature, index);
      d_axis_structure[static_cast<std::size_t>(axis)] = index;
      stencil.clear();
    }

    for (int parameter = 0; parameter <= D; ++parameter)
    {
      KMatStencil stencil{};
      KMat<D,Real>::load_or_discover_natural_stencil(
        static_cast<unsigned int>(q_vol),
        kappa.data(),
        parameter,
        &stencil,
        stencil_folder);

      const std::string signature =
        KMat<D,Real>::stencil_signature(stencil);
      const auto found = k_signature_index.find(signature);
      if (found != k_signature_index.end())
      {
        k_parameter_structure[static_cast<std::size_t>(parameter)] =
          found->second;
        stencil.clear();
        continue;
      }

      int* colptr_raw = nullptr;
      int* rowind_raw = nullptr;
      const std::size_t nnz =
        KMat<D,Real>::build_natural_csc_pattern_from_stencil(
          n, stencil, &colptr_raw, &rowind_raw);

      SharedOperatorStructure structure;
      copy_csc_raw_structure(
        M, M, signature,
        colptr_raw, rowind_raw, nnz, structure);

      const int index = static_cast<int>(k_structures.size());
      k_structures.push_back(std::move(structure));
      k_signature_index.emplace(signature, index);
      k_parameter_structure[static_cast<std::size_t>(parameter)] = index;
      stencil.clear();
    }
  }

  const SharedOperatorStructure& d_structure(int axis) const
  {
    if (axis < 0 || axis >= D)
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: derivative-axis structure index");
    }
    const int index = d_axis_structure[static_cast<std::size_t>(axis)];
    if (index < 0 || index >= static_cast<int>(d_structures.size()))
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: derivative structure not initialized");
    }
    return d_structures[static_cast<std::size_t>(index)];
  }

  const SharedOperatorStructure& k_structure(int parameter) const
  {
    if (parameter < 0 || parameter > D)
    {
      throw std::out_of_range(
        "RefSimplexPrecomp: promotion-parameter structure index");
    }
    const int index =
      k_parameter_structure[static_cast<std::size_t>(parameter)];
    if (index < 0 || index >= static_cast<int>(k_structures.size()))
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: promotion structure not initialized");
    }
    return k_structures[static_cast<std::size_t>(index)];
  }

  std::vector<Real> build_d_factor_dense(
    const Real* kappa_src,
    int axis) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null derivative source kappa");
    }
    const SharedOperatorStructure& structure = d_structure(axis);
    std::vector<Real> values(structure.nnz(), Real(0));
    DMat<D,Real>::fill_tprod_natural_csc_values(
      n,
      static_cast<unsigned int>(q_vol),
      kappa_src,
      axis,
      structure.colptr.data(),
      structure.rowind.data(),
      values.data());

    // Preserve the legacy padded row-major M x M factor layout.
    std::vector<Real> dense(
      static_cast<std::size_t>(M) * M, Real(0));
    for (int col = 0; col < M; ++col)
    {
      for (int pos = structure.colptr[static_cast<std::size_t>(col)];
           pos < structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row = structure.rowind[static_cast<std::size_t>(pos)];
        dense[static_cast<std::size_t>(row) * M + col] =
          values[static_cast<std::size_t>(pos)];
      }
    }
    return dense;
  }

  std::vector<Real> build_k_factor_dense(
    const Real* kappa_src,
    int promoted_parameter) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null promotion source kappa");
    }
    const SharedOperatorStructure& structure =
      k_structure(promoted_parameter);
    std::vector<Real> values(structure.nnz(), Real(0));
    KMat<D,Real>::fill_tprod_natural_csc_values(
      n,
      static_cast<unsigned int>(q_vol),
      kappa_src,
      promoted_parameter,
      structure.colptr.data(),
      structure.rowind.data(),
      values.data());

    std::vector<Real> dense(
      static_cast<std::size_t>(M) * M, Real(0));
    for (int col = 0; col < M; ++col)
    {
      for (int pos = structure.colptr[static_cast<std::size_t>(col)];
           pos < structure.colptr[static_cast<std::size_t>(col + 1)];
           ++pos)
      {
        const int row = structure.rowind[static_cast<std::size_t>(pos)];
        dense[static_cast<std::size_t>(row) * M + col] =
          values[static_cast<std::size_t>(pos)];
      }
    }
    return dense;
  }

  void row_major_product(
    const std::vector<Real>& A,
    const std::vector<Real>& B,
    std::vector<Real>& C) const
  {
    const std::size_t count = static_cast<std::size_t>(M) * M;
    if (A.size() != count || B.size() != count)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: incompatible dense factor dimensions");
    }
    C.assign(count, Real(0));
    detail::BlasGemm<Real>::run(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      M,
      M,
      M,
      Real(1),
      A.data(),
      M,
      B.data(),
      M,
      Real(0),
      C.data(),
      M);
  }

  std::array<int, D + 1> residual_promotion_remaining(
    const Real* kappa_src) const
  {
    if (!kappa_src)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: null residual-promotion source kappa");
    }
    std::array<int, D + 1> remaining{};
    const Real tolerance = Real(1.0e-12);
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      const Real difference =
        kappa_res[static_cast<std::size_t>(parameter)] -
        kappa_src[parameter];
      const long long rounded = std::llround(
        static_cast<long double>(difference));
      if (rounded < 0 ||
          std::abs(difference - static_cast<Real>(rounded)) > tolerance)
      {
        throw std::invalid_argument(
          "RefSimplexPrecomp: residual promotion is not a nonnegative "
          "integer parameter shift");
      }
      remaining[static_cast<std::size_t>(parameter)] =
        static_cast<int>(rounded);
    }
    return remaining;
  }

  const std::vector<Real>& build_residual_promotion_from_remaining(
    const std::array<int, D + 1>& remaining)
  {
    const auto found = residual_promotion_cache.find(remaining);
    if (found != residual_promotion_cache.end())
    {
      return found->second;
    }

    bool identity = true;
    for (int parameter = 0; parameter <= D; ++parameter)
    {
      if (remaining[static_cast<std::size_t>(parameter)] != 0)
      {
        identity = false;
        break;
      }
    }
    if (identity)
    {
      std::vector<Real> I(
        static_cast<std::size_t>(M) * M, Real(0));
      for (int i = 0; i < M; ++i)
      {
        I[static_cast<std::size_t>(i) * M + i] = Real(1);
      }
      const auto inserted = residual_promotion_cache.emplace(
        remaining, std::move(I));
      return inserted.first->second;
    }

    int parameter = -1;
    for (int r = 0; r <= D; ++r)
    {
      if (remaining[static_cast<std::size_t>(r)] > 0)
      {
        parameter = r;
        break;
      }
    }
    if (parameter < 0)
    {
      throw std::runtime_error(
        "RefSimplexPrecomp: invalid residual-promotion DAG state");
    }

    std::array<Real, D + 1> source{};
    for (int r = 0; r <= D; ++r)
    {
      source[static_cast<std::size_t>(r)] =
        kappa_res[static_cast<std::size_t>(r)] -
        static_cast<Real>(remaining[static_cast<std::size_t>(r)]);
    }

    std::array<int, D + 1> next_remaining = remaining;
    --next_remaining[static_cast<std::size_t>(parameter)];

    const std::vector<Real> natural_factor =
      build_k_factor_dense(source.data(), parameter);
    const std::vector<Real>& tail =
      build_residual_promotion_from_remaining(next_remaining);

    std::vector<Real> result;
    row_major_product(tail, natural_factor, result);
    const auto inserted = residual_promotion_cache.emplace(
      remaining, std::move(result));
    return inserted.first->second;
  }

  const std::vector<Real>& build_residual_promotion(
    const Real* kappa_src)
  {
    return build_residual_promotion_from_remaining(
      residual_promotion_remaining(kappa_src));
  }

  void copy_row_major_to_column_major(
    const std::vector<Real>& source,
    Real* destination) const
  {
    if (!destination ||
        source.size() != static_cast<std::size_t>(M) * M)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: invalid dense layout conversion");
    }
    for (int col = 0; col < M; ++col)
    {
      for (int row = 0; row < M; ++row)
      {
        destination[static_cast<std::size_t>(row) +
                    static_cast<std::size_t>(M) * col] =
          source[static_cast<std::size_t>(row) * M + col];
      }
    }
  }

  void build_derivative_factor_data()
  {
    D1_factor_rm.assign(
      static_cast<std::size_t>(D) * M * M, Real(0));
    D2_factor_rm.assign(
      static_cast<std::size_t>(D) * D * M * M, Real(0));
    derivative_kappa_1.resize(static_cast<std::size_t>(D));

    for (int first_axis = 0; first_axis < D; ++first_axis)
    {
      Basis<D,Real>::derivative_kappa_shift(
        kappa.data(),
        first_axis,
        derivative_kappa_1[static_cast<std::size_t>(first_axis)].data());

      const std::vector<Real> first =
        build_d_factor_dense(kappa.data(), first_axis);
      std::copy(
        first.begin(),
        first.end(),
        D1_factor_rm.begin() +
          static_cast<std::size_t>(first_axis) * M * M);

      for (int second_axis = 0; second_axis < D; ++second_axis)
      {
        const std::vector<Real> second = build_d_factor_dense(
          derivative_kappa_1[static_cast<std::size_t>(first_axis)].data(),
          second_axis);
        std::copy(
          second.begin(),
          second.end(),
          D2_factor_rm.begin() + static_cast<std::size_t>(M) * M *
            (static_cast<std::size_t>(second_axis) +
             static_cast<std::size_t>(D) * first_axis));
      }
    }
  }

  void build_second_partials()
  {
    Lij_ref.assign(
      static_cast<std::size_t>(M) * M * D * D, Real(0));

    for (int first_axis = 0; first_axis < D; ++first_axis)
    {
      const Real* first = D1_factor_rm.data() +
        static_cast<std::size_t>(first_axis) * M * M;
      const std::vector<Real> first_matrix(
        first,
        first + static_cast<std::size_t>(M) * M);

      for (int second_axis = 0; second_axis < D; ++second_axis)
      {
        const Real* second = D2_factor_rm.data() +
          static_cast<std::size_t>(M) * M *
          (static_cast<std::size_t>(second_axis) +
           static_cast<std::size_t>(D) * first_axis);
        const std::vector<Real> second_matrix(
          second,
          second + static_cast<std::size_t>(M) * M);

        std::vector<Real> derivative_product;
        row_major_product(
          second_matrix,
          first_matrix,
          derivative_product);

        std::array<Real, D + 1> kappa_second{};
        Basis<D,Real>::derivative_kappa_shift(
          derivative_kappa_1[static_cast<std::size_t>(first_axis)].data(),
          second_axis,
          kappa_second.data());

        const std::vector<Real>& promotion =
          build_residual_promotion(kappa_second.data());
        std::vector<Real> result;
        row_major_product(promotion, derivative_product, result);

        Real* output = Lij_ref.data() +
          static_cast<std::size_t>(M) * M *
          (static_cast<std::size_t>(first_axis) +
           static_cast<std::size_t>(D) * second_axis);
        copy_row_major_to_column_major(result, output);
      }
    }
  }

  void build_first_partials()
  {
    Li_ref.assign(
      static_cast<std::size_t>(M) * M * D, Real(0));

    for (int axis = 0; axis < D; ++axis)
    {
      const Real* derivative = D1_factor_rm.data() +
        static_cast<std::size_t>(axis) * M * M;
      const std::vector<Real> derivative_matrix(
        derivative,
        derivative + static_cast<std::size_t>(M) * M);

      const std::vector<Real>& promotion = build_residual_promotion(
        derivative_kappa_1[static_cast<std::size_t>(axis)].data());
      std::vector<Real> result;
      row_major_product(promotion, derivative_matrix, result);

      copy_row_major_to_column_major(
        result,
        Li_ref.data() + static_cast<std::size_t>(M) * M * axis);
    }
  }

  void build_zero_partials()
  {
    L0_ref.assign(static_cast<std::size_t>(M) * M, Real(0));
    const std::vector<Real>& promotion =
      build_residual_promotion(kappa.data());
    copy_row_major_to_column_major(promotion, L0_ref.data());
  }

  void embed_face_points(int face_id, const int* sigma, std::vector<Real>& Xf) const
  {
    Xf.assign((std::size_t)nq_face * D, (Real)0);

    if constexpr (D == 1)
    {
      Xf[0] = (face_id == 0) ? (Real)1 : (Real)0;
      (void)sigma;
    }
    else
    {
      int fv[D];
      dsimplex_face_vertices<D>(face_id, fv);

      for (int q = 0; q < nq_face; ++q)
      {
        Real mu[D];
        Real sum_y = (Real)0;
        for (int a = 1; a < D; ++a)
        {
          const Real y = Y_face[(std::size_t)q * (D - 1) + (a - 1)];
          mu[a] = y;
          sum_y += y;
        }
        mu[0] = (Real)1 - sum_y;

        Real lambda[D + 1];
        for (int a = 0; a <= D; ++a) { lambda[a] = (Real)0; }

        // sigma is local-face-position -> canonical-face-position.
        for (int local = 0; local < D; ++local)
        {
          const int canonical = sigma[local];
          lambda[fv[local]] = mu[canonical];
        }

        for (int a = 0; a < D; ++a)
        {
          Xf[(std::size_t)q * D + a] = lambda[a + 1];
        }
      }
    }
  }

  void build_face_operator_data()
  {
    T_ref.assign((std::size_t)kf * M * nsigma * nface, (Real)0);
    Fgrad_ref.assign((std::size_t)kf * M * D * nsigma * nface, (Real)0);
    Mface_ref.assign((std::size_t)kf * kf * nface, (Real)0);

    std::vector<int> alpha_vol, tail_vol;
    std::vector<Real> invh_vol;
    Basis<D,Real>::build_structures(n, kappa.data(), alpha_vol, tail_vol, invh_vol);

    std::array<std::array<Real, D + 1>, D> k_deriv{};
    std::array<std::vector<int>, D> alpha_deriv;
    std::array<std::vector<int>, D> tail_deriv;
    std::array<std::vector<Real>, D> invh_deriv;
    for (int a = 0; a < D; ++a)
    {
      Basis<D,Real>::derivative_kappa_shift(kappa.data(), a, k_deriv[(std::size_t)a].data());
      Basis<D,Real>::build_structures(n, k_deriv[(std::size_t)a].data(),
                                      alpha_deriv[(std::size_t)a],
                                      tail_deriv[(std::size_t)a],
                                      invh_deriv[(std::size_t)a]);
    }

    for (int face_id = 0; face_id < nface; ++face_id)
    {
      const Real fscale = face_ref_scale[(std::size_t)face_id];

      // Face mass matrix for this embedded reference face:
      //   Mface_ref = V_face^T * diag(W_face * fscale) * V_face.
      std::vector<Real> WV_face((std::size_t)nq_face * kf, Real(0));
      for (int col = 0; col < kf; ++col)
      {
        for (int q = 0; q < nq_face; ++q)
        {
          WV_face[(std::size_t)q + (std::size_t)nq_face * col] =
            (W_face[(std::size_t)q] * fscale) *
            V_face[(std::size_t)q + (std::size_t)nq_face * col];
        }
      }
      detail::BlasGemm<Real>::run(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        kf,
        kf,
        nq_face,
        Real(1),
        V_face.data(),
        nq_face,
        WV_face.data(),
        nq_face,
        Real(0),
        &Mface_ref[(std::size_t)kf * kf * face_id],
        kf);

      for (int sigma_index = 0; sigma_index < nsigma; ++sigma_index)
      {
        int sigma[D];
        lehmer_index_to_perm<D>(sigma_index, sigma);

        std::vector<Real> Xf;
        embed_face_points(face_id, sigma, Xf);

        std::vector<Real> Vv((std::size_t)nq_face * M, (Real)0);
        Basis<D,Real>::eval_all(
          Xf.data(), D, 1, nq_face, kappa.data(), n,
          alpha_vol.data(), tail_vol.data(), invh_vol.data(), Vv.data(), nq_face, nullptr);

        // T_ref block:
        //   T_ref = V_face^T * diag(W_face * fscale) * Vv.
        std::vector<Real> WVv((std::size_t)nq_face * M, Real(0));
        for (int col = 0; col < M; ++col)
        {
          for (int q = 0; q < nq_face; ++q)
          {
            WVv[(std::size_t)q + (std::size_t)nq_face * col] =
              (W_face[(std::size_t)q] * fscale) *
              Vv[(std::size_t)q + (std::size_t)nq_face * col];
          }
        }
        detail::BlasGemm<Real>::run(
          CblasColMajor,
          CblasTrans,
          CblasNoTrans,
          kf,
          M,
          nq_face,
          Real(1),
          V_face.data(),
          nq_face,
          WVv.data(),
          nq_face,
          Real(0),
          &T_ref[idx_T(0, 0, sigma_index, face_id)],
          kf);

        for (int a = 0; a < D; ++a)
        {
          std::vector<Real> Vrng((std::size_t)nq_face * M, (Real)0);
          Basis<D,Real>::eval_all(
            Xf.data(), D, 1, nq_face, k_deriv[(std::size_t)a].data(), n,
            alpha_deriv[(std::size_t)a].data(),
            tail_deriv[(std::size_t)a].data(),
            invh_deriv[(std::size_t)a].data(),
            Vrng.data(), nq_face, nullptr);

          std::vector<Real> dV((std::size_t)nq_face * M, Real(0));
          // dV = Vrng * D_a.  Vrng is column-major nq_face x M.
          // D_a is stored row-major M x M; interpreted as column-major it is
          // D_a^T, so use transB = CblasTrans.
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasNoTrans,
            CblasTrans,
            nq_face,
            M,
            M,
            Real(1),
            Vrng.data(),
            nq_face,
            D1_factor_rm.data() + (std::size_t)a * M * M,
            M,
            Real(0),
            dV.data(),
            nq_face);

          // Fgrad_ref block:
          //   Fgrad_ref = V_face^T * diag(W_face * fscale) * dV.
          std::vector<Real> WdV((std::size_t)nq_face * M, Real(0));
          for (int col = 0; col < M; ++col)
          {
            for (int q = 0; q < nq_face; ++q)
            {
              WdV[(std::size_t)q + (std::size_t)nq_face * col] =
                (W_face[(std::size_t)q] * fscale) *
                dV[(std::size_t)q + (std::size_t)nq_face * col];
            }
          }
          detail::BlasGemm<Real>::run(
            CblasColMajor,
            CblasTrans,
            CblasNoTrans,
            kf,
            M,
            nq_face,
            Real(1),
            V_face.data(),
            nq_face,
            WdV.data(),
            nq_face,
            Real(0),
            &Fgrad_ref[idx_F(0, 0, a, sigma_index, face_id)],
            kf);
        }
      }
    }
  }

  CSCBlock make_pruned_csc_block(int rows,
                                 int cols,
                                 const Real* A_colmajor) const
  {
    if (rows < 0 || cols < 0 || !A_colmajor)
    {
      throw std::invalid_argument(
        "RefSimplexPrecomp: invalid dense block for CSC conversion");
    }

    long double norm2 = 0.0L;
    const std::size_t count = (std::size_t)rows * cols;
    for (std::size_t k = 0; k < count; ++k)
    {
      const long double v = static_cast<long double>(A_colmajor[k]);
      norm2 += v * v;
    }
    const Real threshold = boundary_csc_rel_prune *
      static_cast<Real>(std::sqrt(norm2));

    std::vector<Real> pruned(count, Real(0));
    for (std::size_t k = 0; k < count; ++k)
    {
      const Real v = A_colmajor[k];
      if (std::abs(v) > threshold) pruned[k] = v;
    }

    int* colptr_raw = nullptr;
    int* rowind_raw = nullptr;
    Real* values_raw = nullptr;
    const std::size_t nnz = detail::compress_dense_to_csc(
      rows,
      cols,
      pruned.data(),
      rows,
      true,
      &colptr_raw,
      &rowind_raw,
      &values_raw);

    std::unique_ptr<int, decltype(&std::free)> colptr_guard(
      colptr_raw, &std::free);
    std::unique_ptr<int, decltype(&std::free)> rowind_guard(
      rowind_raw, &std::free);
    std::unique_ptr<Real, decltype(&std::free)> values_guard(
      values_raw, &std::free);

    CSCBlock out;
    out.rows = rows;
    out.cols = cols;
    out.colptr.assign(colptr_raw, colptr_raw + (std::size_t)cols + 1);
    if (nnz > 0)
    {
      out.rowind.assign(rowind_raw, rowind_raw + nnz);
      out.values.assign(values_raw, values_raw + nnz);
    }
    return out;
  }

  void build_face_sparse_data()
  {
    T_ref_csc.resize((std::size_t)nsigma * nface);
    Fgrad_ref_csc.resize((std::size_t)D * nsigma * nface);
    Mface_ref_csc.resize((std::size_t)nface);

    for (int face = 0; face < nface; ++face)
    {
      Mface_ref_csc[(std::size_t)face] = make_pruned_csc_block(
        kf,
        kf,
        Mface_ref.data() + (std::size_t)kf * kf * face);

      for (int sigma = 0; sigma < nsigma; ++sigma)
      {
        T_ref_csc[(std::size_t)sigma + (std::size_t)nsigma * face] =
          make_pruned_csc_block(
            kf,
            M,
            T_ref.data() + idx_T(0, 0, sigma, face));

        for (int axis = 0; axis < D; ++axis)
        {
          Fgrad_ref_csc[(std::size_t)axis + (std::size_t)D *
            ((std::size_t)sigma + (std::size_t)nsigma * face)] =
            make_pruned_csc_block(
              kf,
              M,
              Fgrad_ref.data() + idx_F(0, 0, axis, sigma, face));
        }
      }
    }
  }

  std::size_t idx_T(int row, int col, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)sigma + (std::size_t)nsigma * face));
  }

  std::size_t idx_F(int row, int col, int axis, int sigma, int face) const
  {
    return (std::size_t)row + (std::size_t)kf *
      ((std::size_t)col + (std::size_t)M *
      ((std::size_t)axis + (std::size_t)D *
      ((std::size_t)sigma + (std::size_t)nsigma * face)));
  }
};

} // namespace jsimplex

#endif // JPRECOMP_HH
