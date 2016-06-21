/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "yashe.h"
#include "settings.h"
int Yashe::d = 0;
Polynomial Yashe::phi = Polynomial();
ZZ Yashe::q = ZZ(0);
bn_t Yashe::qDiv2;
cuyasheint_t Yashe::t = 0;
ZZ Yashe::delta = to_ZZ(0);
ZZ Yashe::w = ZZ(0);
int Yashe::lwq = 0;
Polynomial Yashe::h = Polynomial();
std::vector<Polynomial> Yashe::gamma;
Polynomial Yashe::f = Polynomial();
Polynomial Yashe::ff = Polynomial();
ZZ Yashe::WDMasking = ZZ(0);
std::vector<Polynomial> Yashe::P;


void Yashe::generate_keys(){
  #ifdef DEBUG
  std::cout << "generate_keys:" << std::endl;
  std::cout << "d: " << d << std::endl;
  std::cout << "phi: " << phi.to_string() << std::endl;
  std::cout << "q: " << Polynomial::global_mod << std::endl;
  std::cout << "t: " << t.get_value() << std::endl;
  std::cout << "w: " << w << std::endl;
  std::cout << "R: " << Polynomial::global_mod << std::endl;
  #endif

  Polynomial g = this->xkey.get_sample(phi.deg()-1);
  #ifdef DEBUG
  std::cout << "g = " << g.to_string() << std::endl;
  #endif

  // Computes a polynomial f with inverse
  Polynomial fInv;
  while(1==1){
    Polynomial fl = xkey.get_sample(phi.deg()-1);

    f = fl*t + 1;
    f.reduce();

    #ifdef DEBUG
    std::cout << "fl: " << fl.to_string() << std::endl;
    std::cout << "f: " << f.to_string() << std::endl;
    #endif
    try{
      // fInv = Polynomial::InvMod(f,phi);
      // fInv.normalize();
      fInv = f;

      break;
    } catch (exception& e)
    {
      #ifdef VERBOSE
      std::cout << "f has no modular inverse. " << e.what() << std::endl;
      #endif
    }
  }

  // Pre-computed value
  ff = f*f;
  ff.reduce();

  h = fInv*g*t;
  h.reduce();
  h.update_device_data();

  gamma.resize(lwq);
  for(int k = 0 ; k < lwq; k ++){
    gamma[k] = Polynomial(f);//Copy

    for(int j = 0; j < k;j ++){
      gamma[k] *= w;
    }

    Polynomial e = xerr.get_sample(phi.deg()-1);
    Polynomial s = xerr.get_sample(phi.deg()-1);

    Polynomial hs = h*s;
    hs.reduce();
    gamma[k] += e;
    gamma[k] += hs;
    gamma[k].reduce();
    gamma[k].update_crt_spacing(2*(phi.deg()-1));
    gamma[k].update_device_data();

    #ifdef DEBUG
    std::cout << "e = " << e.to_string() << std::endl;
    std::cout << "s = " << s.to_string() << std::endl;
    std::cout << "gamma[" << k << "] = " << gamma[k].to_string() << std::endl;
    #endif
  }

  // Word decomp mask
  WDMasking = NTL::LeftShift(ZZ(1),NumBits(Yashe::w))-1;

  //////////////////////////////////
  // Init static variables
  Yashe::ps.update_crt_spacing(2*phi.deg()-1);
  Yashe::e.update_crt_spacing(phi.deg()-1);
  //////////////////////////////////
  get_words(&qDiv2,q/2);
  //////////////////////////////////
  delta = (q/t); // q/t
  //////////////////////////////////
  bn_t *d_P;
  const int N = 2*Polynomial::global_phi->deg()-1;
  const int size = N*lwq;

  P.clear();
  P.resize(lwq,N);
  cudaError_t result = cudaMalloc((void**)&d_P,size*sizeof(bn_t));
  assert(result == cudaSuccess);

  bn_t *h_P;
  h_P = (bn_t*)malloc(size*sizeof(bn_t));

  // #pragma omp parallel for
  for(int i = 0; i < size; i++)
    get_words(&h_P[i],to_ZZ(0));

  result = cudaMemcpy(d_P,h_P,size*sizeof(bn_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

  for(int i = 0; i < lwq;i++){
    // cudaFree(P[i].d_bn_coefs);
    P[i].d_bn_coefs = d_P + i*N;
  }
  free(h_P);
  /////
  #ifdef VERBOSE
  std::cout << "Keys generated." << std::endl;
  #endif
}

Ciphertext Yashe::encrypt(Polynomial m){
  #ifdef DEBUG
  std::cout << "delta: "<< delta.get_value() <<std::endl;
  #endif
  /** 
   * ps will be used in a D degree multiplication, resulting in a 2*D degree polynomial
   * e will be used in a 2*D degree addition
   */
  xerr.get_sample(&ps,phi.deg()-1);
  xerr.get_sample(&e,phi.deg()-1);

  #ifdef DEBUG
  std::cout << "ps: "<< ps.to_string() <<std::endl;
  std::cout << "e: "<< e.to_string() <<std::endl;
  #endif
  
  Polynomial mdelta = m*delta;
  // ps *= h;
  // e += mdelta;
  // e += ps;
  assert(ps.get_crt_spacing() == h.get_crt_spacing());
  /////////////////
  // ps = ps + h //
  /////////////////
  #ifdef NTTMUL_TRANSFORM
  CUDAFunctions::callPolynomialMul( ps.get_device_crt_residues(),
                                    ps.get_device_crt_residues(),
                                    h.get_device_crt_residues(),
                                    (int)(ps.get_crt_spacing())*Polynomial::CRTPrimes.size(),
                                    ps.get_stream()
                                    );
  #else
  CUDAFunctions::executeCuFFTPolynomialMul(   ps.get_device_transf_residues(), 
                                              ps.get_device_transf_residues(), 
                                              h.get_device_transf_residues(), 
                                              ps.get_crt_spacing()*Polynomial::CRTPrimes.size(),
                                              ps.get_stream());
  #endif
  
  mdelta.update_crt_spacing(e.get_crt_spacing());

  assert(e.get_crt_spacing() == mdelta.get_crt_spacing());
  /////////////////////
  // e = e + m*delta //
  // e = e + ps //
  ////////////////
  #ifdef NTTMUL_TRANSFORM

  CUDAFunctions::callPolynomialAddSubInPlace( e.get_stream(),
                                              e.get_device_crt_residues(),
                                              mdelta.get_device_crt_residues(),
                                              (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                              ADD);

  CUDAFunctions::callPolynomialAddSubInPlace( e.get_stream(),
                                              e.get_device_crt_residues(),
                                              ps.get_device_crt_residues(),
                                              (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                              ADD);
  #else
  CUDAFunctions::callPolynomialcuFFTAddSubInPlace(  e.get_stream(),
                                                    e.get_device_transf_residues(),
                                                    mdelta.get_device_transf_residues(),
                                                    (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                    ADD);
  CUDAFunctions::callPolynomialcuFFTAddSubInPlace(  e.get_stream(),
                                                    e.get_device_transf_residues(),
                                                    ps.get_device_transf_residues(),
                                                    (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                    ADD);
  #endif
  e.reduce();

  Ciphertext c = e;
  return c;
}

Polynomial Yashe::decrypt(Ciphertext c){
  #ifdef VERBOSE
  std::cout << "Yashe decrypt" << std::endl;
  #endif
  // std::cout << "f " << f.to_string() << std::endl;
  // std::cout << "c " << c.to_string() << std::endl;
  // uint64_t start,end;

  Polynomial m;

  if(c.aftermul){
    #ifdef VERBOSE
    std::cout << "aftermul" << std::endl;
    #endif
    m = ff*c;    
    // std::cout << "f*f:" << g.to_string() << std::endl;
    // std::cout << "f*f*c:" << g.to_string() << std::endl;

  }else{
    #ifdef VERBOSE
    std::cout << "not  aftermul" << std::endl;
    #endif
    // f.set_crt_residues_computed(false);
    m = f*c;
  }
  m.reduce();

  #ifdef NTTMUL_TRANSFORM
  CUDAFunctions::callPolynomialOPIntegerInplace( MUL,
                                          m.get_stream(),
                                          m.get_device_crt_residues(),
                                          Yashe::t,
                                          m.get_crt_spacing(),
                                          Polynomial::CRTPrimes.size()
  );
  #else
  CUDAFunctions::callPolynomialcuFFTOPIntegerInplace( MUL,
                                                      m.get_stream(),
                                                      m.get_device_transf_residues(),
                                                      Yashe::t,
                                                      m.get_crt_spacing(),
                                                      Polynomial::CRTPrimes.size()
  );
  #endif
  m.icrt();
  callMersenneDiv(  m.d_bn_coefs, 
                    Yashe::q, 
                    m.get_crt_spacing(), 
                    m.get_stream());
  m.set_transf_computed(false);
  m.set_itransf_computed(false);
  m.set_crt_computed(false);
  m.set_icrt_computed(true);
  m.set_host_updated(false);
  return m;
}