#ifndef JMAT_H
#define JMAT_H

#include <cstdlib>
#include <cmath>

/* Generate the Jacobi matrices encoding the 3-term
   recurrence relations in each dimension for the 
   d-simplex. 
   D=1 - we use analytical formulae to construct the
         Jacobi matrix
   D>1 - we use a tensor product quadrature rule
         from [0,1]^D to T^D the D-simplex to
         numerically evaluate inner-products needed
         for entries of the Jacobi matrix */

namespace jsimplex
{

/* Forward declaration so we can mention QuadMapped later,
   as needed for D>1 Jacobi matrices */
template<int D, class Real>
struct QuadMapped;

/* Primary template JMat<D,Real>, D >= 1.
   We *declare* it here; definition for D>1 will come later. */
template<int D, class Real>
struct JMat;

/* D = 1 specialization: classical Jacobi on [-1,1]. */
template<class Real>
struct JMat<1, Real>
{
  /* 1D recurrence builder - J is n x n*/
  static void build(unsigned int n, Real a, Real b, Real* J)
  {
    // zero out J
    for (unsigned int i = 0; i < n*n; ++i) { J[i] = Real(0.0); }

    Real* bvec    = (Real*) std::calloc(n, sizeof(Real));
    Real* avecON  = (Real*) std::calloc(n, sizeof(Real));
    Real av, bv, asq = a * a, bsq = b * b;
    bvec[0]   = -(Real(0.5) * (a - b)) / (Real(0.5) * (a + b) + Real(1.0));
    avecON[0] = (Real(2.0) / (a + b + Real(2.0))) * 
                std::sqrt( (a + Real(1.0)) * (b + Real(1.0)) / (a + b + Real(3.0)) );
    for (unsigned int i = 1; i < n; ++i)
    {
      av =
        (Real(2.0) * i + a + b + Real(1.0)) *
        (Real(2.0) * i + a + b + Real(2.0)) /
        ( Real(2.0) * (i + Real(1.0)) * (i + a + b + Real(1.0)) );

      bv =
        (asq - bsq) * (Real(2.0) * i + a + b + Real(1.0)) /
        ( Real(2.0) * (i + Real(1.0)) * (i + a + b + Real(1.0)) * (Real(2.0) * i + a + b) );

      bvec[i] = -bv / av;

      avecON[i] =
        Real(2.0) / (a + b + Real(2.0) * i + Real(2.0)) *
        std::sqrt(
                  (a + i + Real(1.0)) * (b + i + Real(1.0)) *
                  (i + Real(1.0)) * (a + b + i + Real(1.0)) /
                  ( (a + b + Real(2.0) * i + Real(1.0)) *
                    (a + b + Real(2.0) * i + Real(3.0)) )
                 );
    }

    for (unsigned int i = 0; i < n; ++i)
    {
      J[i + n*i] = bvec[i];

    }

    for (unsigned int i = 0; i < n-1; ++i)
    {
      J[i + n*(i+1)] = avecON[i];
      J[i+1 + n*i] = avecON[i];
    }
  
    std::free(bvec);
    std::free(avecON);  
  
  }
};


/* D>1 specialization using mapped quadrature from d-cube
   to d-simplex for entries of the Jacobi matrix */
template<int D, class Real>
struct JMat
{
  /* Gram matrix on the D-simplex using mapped quadrature. */
  static void build(const Real* kappa, unsigned int nquad,
                    const Real* basis_vals, unsigned int nbasis,
                    Real* G /* nbasis x nbasis */)
  {
    // implementation will use QuadMapped<D,Real>::integrate(...)
  }

};

} // namespace jsimplex

#endif // JMAT_H
