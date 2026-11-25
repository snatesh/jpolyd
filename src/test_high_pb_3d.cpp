#include <jdivergence.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main()
{
  constexpr int D=3;
  using Real=double;
  Real kappa[4]={0.5,0.5,0.5,0.5};
  jsimplex::RefSimplexPrecomp<D,Real> pre(4,2,9,9,kappa,"/tmp/jpolyd_high_pb_stencils");
  Real verts[12]={0,0,0,1,0,0,0,1,0,0,0,1};
  jsimplex::DSimplexGeom<D,Real> geom;
  jsimplex::dsimplex_affine_from_verts<D,Real>(verts,geom);
  const int mpb=jsimplex::Basis<D,Real>::dim_Pi(6);
  std::vector<Real> b(D*mpb,0);
  for(int r=0;r<D;r++) for(int j=0;j<mpb;j++) b[r*mpb+j]=0.02*std::sin(0.25*j+1.2*r);
  const int mpa=1;
  Real A[D*D*mpa]={1,0,0,0,1,0,0,0,1};
  auto expanded=jsimplex::expand_divergence_form_element<D,Real>(pre,geom,0,A,6,b.data(),-1,nullptr);
  std::vector<Real> X={0.2,0.15,0.10,0.4,0.12,0.20,0.12,0.25,0.35};
  const int nq=3;
  std::vector<int> ab,tb,ar,tr; std::vector<Real> hb,hr;
  jsimplex::Basis<D,Real>::build_structures(6,pre.kappa.data(),ab,tb,hb);
  jsimplex::Basis<D,Real>::build_structures(6,pre.kappa_res.data(),ar,tr,hr);
  std::vector<Real> Vb(nq*mpb),Vr(nq*mpb);
  jsimplex::Basis<D,Real>::eval_all(X.data(),D,1,nq,pre.kappa.data(),6,ab.data(),tb.data(),hb.data(),Vb.data(),nq,nullptr);
  jsimplex::Basis<D,Real>::eval_all(X.data(),D,1,nq,pre.kappa_res.data(),6,ar.data(),tr.data(),hr.data(),Vr.data(),nq,nullptr);
  double maxerr=0;
  for(int r=0;r<D;r++) for(int iq=0;iq<nq;iq++) {
    double src=0, dst=0;
    for(int j=0;j<mpb;j++) {src+=b[r*mpb+j]*Vb[iq+nq*j];dst+=expanded.b[r*mpb+j]*Vr[iq+nq*j];}
    maxerr=std::max(maxerr,std::abs(src-dst));
  }
  std::cout<<"max coefficient conversion error: "<<maxerr<<"\n";
  if(maxerr>1.e-9)return 1;
  jsimplex::EllipticDegreeSpec deg{0,6,-1};
  jsimplex::EllipticPlan<D,Real> plan(4,pre.kappa_res.data(),deg,true,jsimplex::EllipticResidualPolicy::TrialDegree,jsimplex::EllipticMultiplicationAssembler::Quadrature,false);
  const int MN=jsimplex::Basis<D,Real>::dim_Pi(3);
  std::vector<Real> Mq((std::size_t)plan.mR*MN);
  jsimplex::MultByQQuadratureWorkspace<Real> work;
  jsimplex::assemble_restricted_mult_quadrature<D,Real>(
    plan.quadrature_plan(),expanded.b.data(),6,3,4,work,Mq.data());
  const auto max_it=std::max_element(Mq.begin(),Mq.end(),
    [](Real a,Real b){return std::abs(a)<std::abs(b);});
  std::cout<<"R="<<plan.mR<<" input="<<MN<<" pb="<<plan.degrees.p1
           <<" quadrature="<<plan.q_mult<<" max|M|="<<std::abs(*max_it)<<"\n";
  for(Real x:Mq) if(!std::isfinite(x)) return 1;
  return plan.mR==jsimplex::Basis<D,Real>::dim_Pi(4)?0:1;
}
