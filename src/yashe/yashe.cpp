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

int Yashe::nphi;
int Yashe::nq;
bn_t Yashe::Q;
bn_t Yashe::UQ;
bn_t Yashe::qDiv2;
ZZ Yashe::q = ZZ(0);
cuyasheint_t Yashe::t = 0;
bn_t Yashe::delta;
ZZ Yashe::w = ZZ(0);
int Yashe::lwq = 0;
std::vector<poly_t> Yashe::gamma;
poly_t Yashe::h;
poly_t Yashe::f;
poly_t Yashe::ff;
poly_t Yashe::tf;
poly_t Yashe::tff;
poly_t Yashe::mdelta;
ZZ Yashe::WDMasking = ZZ(0);
std::vector<poly_t> Yashe::P;

// uint64_t get_cycles() {
//   unsigned int hi, lo;
//   asm (
//     "cpuid\n\t"/*serialize*/
//     "rdtsc\n\t"/*read the clock*/
//     "mov %%edx, %0\n\t"
//     "mov %%eax, %1\n\t"
//     : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
//   );
//   return ((uint64_t) lo) | (((uint64_t) hi) << 32);
// }

// #define BILLION  1000000000L
// #define MILLION  1000000L

// double compute_time_ms(struct timespec start,struct timespec stop){
//   return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
// }


void Yashe::generate_keys(){
  log_debug("generate_keys:");
  std::cout << "t: " << t << std::endl;
  /////////
  // q/t //
  /////////
  q = (NTL::power2_ZZ(nq)-1);
  get_words(&Yashe::Q,q);
  Yashe::UQ = get_reciprocal(q);
  get_words(&delta,q/to_ZZ(t));  
  std::cout << "delta: " << q/to_ZZ(t) << std::endl;  

  // Compute f and fInv
  poly_t fInv;
  while ( 1 == 1 ){
    xkey.get_sample(&fl, nphi-1);
    log_debug("fl: " + poly_print(&fl));

    // f = fl*t + 1
    poly_integer_mul(&f,&fl,t);
    
    poly_t one;
    poly_init(&one);
    poly_set_coeff(&one,0,to_ZZ(1));
    
    poly_add(&f,&f,&one);
    
    poly_reduce(&f, nphi, Yashe::Q, nq,Yashe::UQ); 

  try{
      log_debug("will try to compute fInv");
      poly_init(&fInv);
      // ZZ coeff = poly_get_coeff(&f,poly_get_deg(&f));
      // poly_set_coeff(&fInv,1,NTL::InvMod(coeff,q)),
      // std::cout << "coeff: " << coeff << std::endl;
      poly_invmod(&fInv,&f,nphi,nq);

      poly_t test;
      poly_init(&test);
      poly_mul(&test,&f,&fInv);
      poly_reduce(&test,nphi,Yashe::Q,nq,Yashe::UQ);
      log_debug("test: " + poly_print(&test));
      log_debug("fInv: " + poly_print(&fInv));
      std::cout << poly_get_coeff(&test,0) << std::endl;
      if(poly_get_deg(&test) != 0)
        throw std::runtime_error("wrong degree");
      if(poly_get_coeff(&test,0) != to_ZZ(1))
        throw std::runtime_error("0-coefficient different than 1");

      log_debug("fInv computed.");
      break;
    } catch (exception& e)
    {
      log_warn("f has no modular inverse: ");
      log_warn(e.what());
      std::cout << "f has no modular inverse: " << e.what()<< std::endl;
    }
  }
  ////////////
  // DEBUG  //
  ////////////
  // poly_clear(&f);
  // poly_init(&f);

  // poly_set_coeff(&f,0,to_ZZ(1));
  // poly_set_coeff(&f,1,to_ZZ(0));
  // poly_set_coeff(&f,2,to_ZZ(8174));
  // poly_set_coeff(&f,3,to_ZZ(1));
  log_debug("f: " + poly_print(&f));

  // ff = f*f
  poly_mul(&ff,&f,&f);
  poly_reduce(&ff,nphi,Yashe::Q,nq,Yashe::UQ);
    
  // tff = ff*t
  poly_integer_mul(&tff,&ff,t);
  poly_reduce(&tff,nphi,Yashe::Q,nq,Yashe::UQ);

  // Sample
  xkey.get_sample(&g, nphi-1);
  log_debug("g: " + poly_print(&g));
  // h = fInv*g*t
  poly_mul(&h, &fInv,&g);
  log_debug("fInv*g: " + poly_print(&h));
  poly_integer_mul(&h, &h, t);
  log_debug("fInv*gt*t: " + poly_print(&h));
  poly_reduce(&h, nphi, Yashe::Q,nq,Yashe::UQ);
  while(h.status != TRANSSTATE)
    poly_elevate(&h);

  log_debug("h: " + poly_print(&h));


  ///////////////////
  // Compute gamma //
  ///////////////////
  gamma.resize(lwq);

  ///////////
  // TO-DO //
  ///////////
}
poly_t Yashe::encrypt(poly_t m){
  log_notice("Encrypt");

  // uint64_t start,end,total_start,total_end;
  // total_start = get_cycles();

  // Sample
  // start = get_cycles();
  xerr.get_sample(&ps,nphi-1);
  xerr.get_sample(&e,nphi-1);
  // end = get_cycles();
  // std::cout << "Encrypt sampling in " + std::to_string(end-start) + " cycles" << std::endl;
  // poly_reduce(&e,nphi,nYashe::Q,nq,Yashe::UQ);
  // poly_reduce(&ps,nphi,nYashe::Q,nq,Yashe::UQ);
  log_debug("ps: " + poly_print(&ps));
  log_debug("e: " + poly_print(&e));
  
  // poly_init(&ps);
  // poly_init(&e);
  // poly_set_coeff(&ps,0,to_ZZ(8188));
  // poly_set_coeff(&ps,1,to_ZZ(0));
  // poly_set_coeff(&ps,2,to_ZZ(8190));
  // poly_set_coeff(&ps,3,to_ZZ(1));

  // poly_set_coeff(&e,0,to_ZZ(2));
  // poly_set_coeff(&e,1,to_ZZ(1));
  // poly_set_coeff(&e,2,to_ZZ(0));
  // poly_set_coeff(&e,3,to_ZZ(8190));

	// 
  // start = get_cycles();
	poly_biginteger_mul(&mdelta,&m,delta);

  // end = get_cycles();
  // std::cout << "poly_biginteger_mul in " + std::to_string(end-start) + " cycles" << std::endl;

  log_debug("m * delta: " + poly_print(&mdelta));

	// 
  // start = get_cycles();
	poly_add(&e,&e,&mdelta);

  // end = get_cycles();
  // std::cout << "poly_add in " + std::to_string(end-start) + " cycles" << std::endl;
  log_debug("mdelta + e: " + poly_print(&e));
	
  //
  // start = get_cycles();
	poly_mul(&ps,&ps,&h);

  // end = get_cycles();
  // std::cout << "poly_mul in " + std::to_string(end-start) + " cycles" << std::endl;
  log_debug("ps * h: " + poly_print(&ps));

	//
	poly_t c;
  // start = get_cycles();
  poly_init(&c);
  poly_add(&c,&e,&ps);
  log_debug("c: " + poly_print(&c));

  // end = get_cycles();
  // std::cout << "poly_add in " + std::to_string(end-start) + " cycles" << std::endl;

  //
  // start = get_cycles();
  poly_reduce(&c, nphi, Yashe::Q,nq,Yashe::UQ);

  // end = get_cycles();
  // std::cout << "poly_reduce in " + std::to_string(end-start) + " cycles" << std::endl;
  log_debug("c \\in R_q: " + poly_print(&c));
  
  // total_end = get_cycles(); 
  // std::cout << std::endl << "Yashe::encrypt in " + std::to_string(total_end-total_start) + " cycles" << std::endl<< std::endl;
  return c;
}

poly_t Yashe::decrypt(poly_t c){
  log_notice("Decrypt");
  // uint64_t start,end,total_start,total_end;
  // total_start = get_cycles();

  poly_t m;
  poly_init(&m);

  poly_mul(&m, &f, &c);
  log_debug("[c*f]: " + poly_print(&m));
  poly_reduce(&m, nphi, Yashe::Q,nq,Yashe::UQ);
  log_debug("[c*f]_q \\in R: " + poly_print(&m));

  poly_integer_mul(&m, &m, t);
  log_debug("[c*f]_q \\in R * t: " + poly_print(&m));

  // division by q to the nearest
   
  // start = get_cycles();
  for(int i = 0; i <= poly_get_deg(&m); i++){
    ZZ coeff = poly_get_coeff(&m,i);
    ZZ diff = coeff % q;
    if(2*diff > q)
     poly_set_coeff(&m,i,coeff/q +1);
    else
      poly_set_coeff(&m,i,coeff/q);
  }
  // poly_demote(&m); // CRT
  // poly_icrt(&m);
  // callCiphertextMulAux(m.d_bn_coefs, Yashe::Q, nq, CUDAFunctions::N, NULL);
  // callCRT(m.d_bn_coefs,
  //         CUDAFunctions::N,
  //         m.d_coefs,
  //         CUDAFunctions::N,
  //         CRTPrimes.size(),
  //         0x0
  //   );
  // m.status = CRTSTATE;
  // end = get_cycles();
  // std::cout << "decrypt last step in " + std::to_string(end-start) + " cycles" << std::endl;

  return m;
}
