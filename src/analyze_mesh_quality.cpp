#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>

// If you have LAPACKE installed (likely already via your jpolyd setup):
#include <lapacke.h>

struct Mesh {
  int num_nodes = 0;
  int num_tets  = 0;
  std::vector<double> pts;  // size = 3 * num_nodes
  std::vector<int>    tets; // size = 4 * num_tets
};

// Simple VTK legacy loader for volume mesh (same as in assemble_cn_step)
Mesh load_volume_mesh_vtk(const std::string& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs) {
    throw std::runtime_error("Failed to open VTK file: " + filename);
  }

  Mesh mesh;
  std::string token;

  int num_pts   = 0;
  int num_cells = 0;
  std::vector<std::vector<int>> cells;
  std::vector<int> cell_types;

  while (ifs >> token) {
    if (token == "POINTS") {
      std::string type_str;
      ifs >> num_pts >> type_str;
      mesh.num_nodes = num_pts;
      mesh.pts.resize(3 * num_pts);
      for (int i = 0; i < 3 * num_pts; ++i) {
        ifs >> mesh.pts[i];
      }
    } else if (token == "CELLS") {
      int total_ints = 0;
      ifs >> num_cells >> total_ints;
      cells.resize(num_cells);
      for (int c = 0; c < num_cells; ++c) {
        int k;
        ifs >> k;
        cells[c].resize(k);
        for (int j = 0; j < k; ++j) {
          ifs >> cells[c][j];
        }
      }
    } else if (token == "CELL_TYPES") {
      int nct = 0;
      ifs >> nct;
      cell_types.resize(nct);
      for (int c = 0; c < nct; ++c) {
        ifs >> cell_types[c];
      }
    }
  }

  // Extract tetrahedra: VTK_TETRA has type=10
  for (int c = 0; c < num_cells; ++c) {
    if (cell_types[c] == 10 && (int)cells[c].size() == 4) {
      mesh.tets.push_back(cells[c][0]);
      mesh.tets.push_back(cells[c][1]);
      mesh.tets.push_back(cells[c][2]);
      mesh.tets.push_back(cells[c][3]);
    }
  }

  mesh.num_tets = (int)mesh.tets.size() / 4;
  if (mesh.num_tets == 0) {
    throw std::runtime_error("No tetrahedra found in VTK file: " + filename);
  }

  return mesh;
}

// Compute Jacobian J for tet K: J = [x1-x0, x2-x0, x3-x0] (3x3)
void tet_jacobian(const Mesh& mesh, int K, double J[9])
{
  const int* t = &mesh.tets[4 * K];
  int v0 = t[0];
  int v1 = t[1];
  int v2 = t[2];
  int v3 = t[3];

  const double* X = mesh.pts.data();
  const double* x0 = &X[3 * v0];
  const double* x1 = &X[3 * v1];
  const double* x2 = &X[3 * v2];
  const double* x3 = &X[3 * v3];

  // Columns: x1-x0, x2-x0, x3-x0
  J[0] = x1[0] - x0[0];  J[3] = x2[0] - x0[0];  J[6] = x3[0] - x0[0];
  J[1] = x1[1] - x0[1];  J[4] = x2[1] - x0[1];  J[7] = x3[1] - x0[1];
  J[2] = x1[2] - x0[2];  J[5] = x2[2] - x0[2];  J[8] = x3[2] - x0[2];
}

// Determinant of 3x3 matrix in column-major J
double det3(const double J[9])
{
  // J = [ j00 j03 j06
  //       j01 j04 j07
  //       j02 j05 j08 ]
  double j00 = J[0], j01 = J[1], j02 = J[2];
  double j10 = J[3], j11 = J[4], j12 = J[5];
  double j20 = J[6], j21 = J[7], j22 = J[8];

  return j00 * (j11 * j22 - j12 * j21)
       - j10 * (j01 * j22 - j02 * j21)
       + j20 * (j01 * j12 - j02 * j11);
}

// Build G = J^T J (3x3 symmetric), column-major
void JTJ(const double J[9], double G[9])
{
  // G = J^T J
  // G[i,j] = sum_k J[k,i] * J[k,j], i,j=0..2
  for (int i = 0; i < 9; ++i) G[i] = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        double Jki = J[i*3 + k]; // (row=k, col=i)
        double Jkj = J[j*3 + k]; // (row=k, col=j)
        sum += Jki * Jkj;
      }
      G[j*3 + i] = sum; // column-major
    }
  }
}

// Compute eigenvalues of symmetric 3x3 G using LAPACKE_dsyev
bool eigenvalues_sym3(double G[9], double evals[3])
{
  // LAPACK uses column-major; G is 3x3 column-major already.
  int n = 3;
  int lda = 3;
  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', n, G, lda, evals);
  return (info == 0);
}

// Compute all 6 edge lengths for tet K, and return (min, max)
void tet_edge_lengths(const Mesh& mesh, int K, double& minlen, double& maxlen)
{
  const int* t = &mesh.tets[4 * K];
  int v[4] = { t[0], t[1], t[2], t[3] };

  const double* X = mesh.pts.data();
  double minL = std::numeric_limits<double>::infinity();
  double maxL = 0.0;

  for (int i = 0; i < 4; ++i) {
    for (int j = i+1; j < 4; ++j) {
      const double* xi = &X[3 * v[i]];
      const double* xj = &X[3 * v[j]];
      double dx = xi[0] - xj[0];
      double dy = xi[1] - xj[1];
      double dz = xi[2] - xj[2];
      double L = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (L < minL) minL = L;
      if (L > maxL) maxL = L;
    }
  }

  minlen = minL;
  maxlen = maxL;
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " mesh.vtk\n";
    return EXIT_FAILURE;
  }

  std::string filename = argv[1];
  Mesh mesh;
  try {
    mesh = load_volume_mesh_vtk(filename);
  } catch (const std::exception& e) {
    std::cerr << "Error loading mesh: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Mesh: nodes = " << mesh.num_nodes
            << ", tets = " << mesh.num_tets << "\n";

  double min_det   =  std::numeric_limits<double>::infinity();
  double max_det   = -std::numeric_limits<double>::infinity();
  double sum_det   =  0.0;
  double sum_absdet=  0.0;

  double min_cond  =  std::numeric_limits<double>::infinity();
  double max_cond  =  0.0;
  double sum_cond  =  0.0;

  double min_ar    =  std::numeric_limits<double>::infinity(); // edge ratio
  double max_ar    =  0.0;
  double sum_ar    =  0.0;

  int n_bad_det    = 0;
  int n_bad_cond10 = 0;
  int n_bad_cond100= 0;
  int n_bad_ar10   = 0;

  for (int K = 0; K < mesh.num_tets; ++K) {
    double J[9];
    tet_jacobian(mesh, K, J);
    double det = det3(J);

    // Track det stats
    if (det < min_det) min_det = det;
    if (det > max_det) max_det = det;
    sum_det    += det;
    sum_absdet += std::fabs(det);
    if (std::fabs(det) < 1e-12) n_bad_det++;

    // Condition number via eigenvalues of G = J^T J
    double G[9];
    JTJ(J, G);
    double evals[3];
    bool ok = eigenvalues_sym3(G, evals);
    double condJ = 0.0;
    if (ok) {
      double lam_min = evals[0];
      double lam_max = evals[2];
      // guard against non-positive from numerical noise
      if (lam_min <= 0.0) {
        condJ = std::numeric_limits<double>::infinity();
        n_bad_cond100++;
      } else {
        condJ = std::sqrt(lam_max / lam_min);
      }
    } else {
      condJ = std::numeric_limits<double>::infinity();
      n_bad_cond100++;
    }

    if (condJ < min_cond) min_cond = condJ;
    if (condJ > max_cond) max_cond = condJ;
    if (std::isfinite(condJ)) {
      sum_cond += condJ;
    }

    if (condJ > 10.0)  n_bad_cond10++;
    if (condJ > 100.0) n_bad_cond100++;

    // Edge length ratio
    double minL, maxL;
    tet_edge_lengths(mesh, K, minL, maxL);
    double ar = maxL / (minL + 1e-300);
    if (ar < min_ar) min_ar = ar;
    if (ar > max_ar) max_ar = ar;
    sum_ar += ar;

    if (ar > 10.0) n_bad_ar10++;
  }

  int Nt = mesh.num_tets;
  double avg_det    = sum_det / Nt;
  double avg_absdet = sum_absdet / Nt;
  double avg_cond   = sum_cond / Nt;
  double avg_ar     = sum_ar / Nt;

  std::cout << "\nTet Jacobian det(J) stats:\n";
  std::cout << "  min detJ       = " << min_det << "\n";
  std::cout << "  max detJ       = " << max_det << "\n";
  std::cout << "  avg detJ       = " << avg_det << "\n";
  std::cout << "  avg |detJ|     = " << avg_absdet << "\n";
  std::cout << "  #|detJ|<1e-12  = " << n_bad_det << " / " << Nt << "\n";

  std::cout << "\nTet Jacobian condition number stats (approx):\n";
  std::cout << "  min cond(J)    = " << min_cond << "\n";
  std::cout << "  max cond(J)    = " << max_cond << "\n";
  std::cout << "  avg cond(J)    = " << avg_cond << "\n";
  std::cout << "  #cond(J)>10    = " << n_bad_cond10  << " / " << Nt << "\n";
  std::cout << "  #cond(J)>100   = " << n_bad_cond100 << " / " << Nt << "\n";

  std::cout << "\nTet edge length ratio (max edge / min edge):\n";
  std::cout << "  min ratio      = " << min_ar << "\n";
  std::cout << "  max ratio      = " << max_ar << "\n";
  std::cout << "  avg ratio      = " << avg_ar << "\n";
  std::cout << "  #ratio>10      = " << n_bad_ar10 << " / " << Nt << "\n";

  return EXIT_SUCCESS;
}

