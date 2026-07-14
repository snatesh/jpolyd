#include <jlaplace_c.h>

#include <jlaplace.hh>

using namespace jsimplex;

template<int D>
static inline int jlaplace_validate(int M,
                                    int m_int,
                                    const double* G,
                                    const double* Lij_ref,
                                    const double* L_int_out)
{
  if (!G || !Lij_ref || !L_int_out)
  {
    return 1;
  }
  if (M <= 0 || m_int <= 0 || m_int > M)
  {
    return 2;
  }
  return 0;
}

template<int D>
static inline int jlaplace_assemble_L_int_D(int M,
                                             int m_int,
                                             const double* G,
                                             double detBabs,
                                             const double* Lij_ref,
                                             double* L_int_out)
{
  const int ret = jlaplace_validate<D>(M, m_int, G, Lij_ref, L_int_out);
  if (ret != 0)
  {
    return ret;
  }

  jdsimplex_assemble_L_int<D,double>(
    M,
    m_int,
    G,
    detBabs,
    Lij_ref,
    L_int_out
  );

  return 0;
}

extern "C" {

int jlaplace_assemble_L_int(int D,
                             int M,
                             int m_int,
                             const double* G,
                             double detBabs,
                             const double* Lij_ref,
                             double* L_int_out)
{
  switch (D)
  {
    case 1: return jlaplace_assemble_L_int_D<1>(M, m_int, G, detBabs,
                                                 Lij_ref, L_int_out);
    case 2: return jlaplace_assemble_L_int_D<2>(M, m_int, G, detBabs,
                                                 Lij_ref, L_int_out);
    case 3: return jlaplace_assemble_L_int_D<3>(M, m_int, G, detBabs,
                                                 Lij_ref, L_int_out);
    case 4: return jlaplace_assemble_L_int_D<4>(M, m_int, G, detBabs,
                                                 Lij_ref, L_int_out);
    case 5: return jlaplace_assemble_L_int_D<5>(M, m_int, G, detBabs,
                                                 Lij_ref, L_int_out);
    default: return 3;
  }
}

} // extern "C"
