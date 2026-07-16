#ifndef JMESH_HH
#define JMESH_HH

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <jgeom.hh>
#include <jperms.hh>

namespace jsimplex {

template<int D, class Real>
struct MeshVertex
{
  static_assert(D >= 1, "D must be positive");

  int id = -1;
  std::array<Real,D> x{};
};

template<int D>
struct FaceIncidence
{
  static_assert(D >= 1, "D must be positive");

  int element_id = -1;
  int local_face_id = -1;
};

template<int D, class Real>
struct MeshElementLocalData
{
  static_assert(D >= 1, "D must be positive");

  int element_id = -1;
  std::array<int,D + 1> global_vids{};

  // D x (D+1), column-major.
  std::array<Real,D * (D + 1)> V_phys{};

  DSimplexGeom<D,Real> geom{};
  std::array<Real,D * D> G{}; // B^{-1} B^{-T}, column-major.

  std::array<DSimplexFaceKey<D>,D + 1> face_keys{};
  std::array<int,D + 1> face_sigma_index{};
  std::array<Real,D + 1> face_ref_scale{};
  std::array<DSimplexFaceGeom<D,Real>,D + 1> face_geom{};
};

/*
  Minimal conforming D-simplex mesh for the HPS skeleton path.

  Input:
    vertices: global vertex id + coordinate in R^D
    simplices: (D+1)-tuples of global vertex ids

  build() computes element-local affine geometry, canonical face keys,
  sigma indices, face geometry, and a face-incidence map.
*/
template<int D, class Real = double>
class Mesh
{
public:
  static_assert(D >= 1, "Mesh requires D>=1");

  using Vertex = MeshVertex<D,Real>;
  using Simplex = std::array<int,D + 1>;
  using FaceKey = DSimplexFaceKey<D>;
  using Incidence = FaceIncidence<D>;
  using ElementData = MeshElementLocalData<D,Real>;
  using FaceIncidenceMap = std::map<FaceKey,std::vector<Incidence>>;

  std::vector<Vertex> vertices;
  std::vector<Simplex> simplices;
  std::vector<ElementData> elements;
  FaceIncidenceMap face_incidence;

  Mesh() = default;

  void clear()
  {
    vertices.clear();
    simplices.clear();
    elements.clear();
    face_incidence.clear();
  }

  int add_vertex(int global_id, const std::array<Real,D>& x)
  {
    vertices.push_back(Vertex{global_id, x});
    return static_cast<int>(vertices.size()) - 1;
  }

  int add_simplex(const Simplex& global_vids)
  {
    simplices.push_back(global_vids);
    return static_cast<int>(simplices.size()) - 1;
  }

  std::size_t num_vertices() const { return vertices.size(); }
  std::size_t num_elements() const { return elements.size(); }

  void build(bool require_manifold_faces = true)
  {
    elements.clear();
    face_incidence.clear();

    std::map<int,int> vertex_pos;
    for (int i = 0; i < static_cast<int>(vertices.size()); ++i)
    {
      const int gid = vertices[(std::size_t)i].id;
      if (vertex_pos.find(gid) != vertex_pos.end())
      {
        throw std::invalid_argument("Mesh::build: duplicate global vertex id");
      }
      vertex_pos[gid] = i;
    }

    elements.reserve(simplices.size());
    for (int e = 0; e < static_cast<int>(simplices.size()); ++e)
    {
      const Simplex& s = simplices[(std::size_t)e];
      check_distinct_simplex_vertices(s);

      ElementData elem{};
      elem.element_id = e;
      elem.global_vids = s;

      for (int a = 0; a < D + 1; ++a)
      {
        const auto it = vertex_pos.find(s[(std::size_t)a]);
        if (it == vertex_pos.end())
        {
          throw std::invalid_argument("Mesh::build: simplex references missing vertex id");
        }

        const Vertex& v = vertices[(std::size_t)it->second];
        for (int r = 0; r < D; ++r)
        {
          elem.V_phys[(std::size_t)r + (std::size_t)D * a] = v.x[(std::size_t)r];
        }
      }

      dsimplex_affine_from_verts<D,Real>(elem.V_phys.data(), elem.geom);
      if (!elem.geom.valid)
      {
        throw std::runtime_error("Mesh::build: degenerate affine simplex");
      }
      dsimplex_metric_from_geom<D,Real>(elem.geom, elem.G.data());

      for (int f = 0; f < D + 1; ++f)
      {
        elem.face_keys[(std::size_t)f] = dsimplex_face_key<D>(s.data(), f);
        elem.face_sigma_index[(std::size_t)f] = dsimplex_compute_face_sigma<D>(s.data(), f);

        int fv[D];
        dsimplex_face_vertices<D>(f, fv);
        elem.face_ref_scale[(std::size_t)f] =
          dsimplex_reference_face_scale_from_vertex_ids<D,Real>(fv);

        dsimplex_physical_face_geometry_colmajor<D,Real>(
          elem.V_phys.data(),
          f,
          elem.face_geom[(std::size_t)f]);
        if (!elem.face_geom[(std::size_t)f].valid)
        {
          throw std::runtime_error("Mesh::build: degenerate physical face");
        }

        face_incidence[elem.face_keys[(std::size_t)f]].push_back(Incidence{e, f});
      }

      elements.push_back(elem);
    }

    if (require_manifold_faces)
    {
      for (const auto& kv : face_incidence)
      {
        if (kv.second.empty() || kv.second.size() > 2)
        {
          throw std::runtime_error("Mesh::build: non-manifold face incidence count");
        }
      }
    }
  }

  const ElementData& element(int element_id) const
  {
    if (element_id < 0 || element_id >= static_cast<int>(elements.size()))
    {
      throw std::out_of_range("Mesh::element: element_id out of range");
    }
    return elements[(std::size_t)element_id];
  }

  const std::vector<Incidence>& incidences(const FaceKey& key) const
  {
    const auto it = face_incidence.find(key);
    if (it == face_incidence.end())
    {
      throw std::out_of_range("Mesh::incidences: unknown face key");
    }
    return it->second;
  }

  int incidence_count(const FaceKey& key) const
  {
    const auto it = face_incidence.find(key);
    if (it == face_incidence.end())
    {
      return 0;
    }
    return static_cast<int>(it->second.size());
  }

  bool is_boundary_face(const FaceKey& key) const
  {
    return incidence_count(key) == 1;
  }

  bool is_interior_face(const FaceKey& key) const
  {
    return incidence_count(key) == 2;
  }

  std::vector<FaceKey> boundary_face_keys() const
  {
    std::vector<FaceKey> out;
    for (const auto& kv : face_incidence)
    {
      if (kv.second.size() == 1)
      {
        out.push_back(kv.first);
      }
    }
    return out;
  }

  std::vector<FaceKey> interior_face_keys() const
  {
    std::vector<FaceKey> out;
    for (const auto& kv : face_incidence)
    {
      if (kv.second.size() == 2)
      {
        out.push_back(kv.first);
      }
    }
    return out;
  }

private:
  static void check_distinct_simplex_vertices(const Simplex& s)
  {
    std::set<int> ids;
    for (int i = 0; i < D + 1; ++i)
    {
      if (!ids.insert(s[(std::size_t)i]).second)
      {
        throw std::invalid_argument("Mesh: simplex has repeated vertex id");
      }
    }
  }
};

} // namespace jsimplex

#endif // JMESH_HH
