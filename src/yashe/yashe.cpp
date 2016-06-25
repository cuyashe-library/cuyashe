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
ZZ Yashe::q = ZZ(0);
bn_t Yashe::qDiv2;
cuyasheint_t Yashe::t = 0;
ZZ Yashe::delta = to_ZZ(0);
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


void Yashe::generate_keys(){
  log_debug("generate_keys:");
  log_debug("nphi: " + nphi);
  log_debug("nq: " + nq);
  log_debug("t: " + t);
  // log_debug("w: " << w);

  // Compute f and fInv
  poly_t fInv;
  while ( 1 == 1 ){
    poly_t fl;
    xkey.get_sample(&fl, nphi-1);
    // log_notice("fl: " + poly_print(&fl));

    // f = fl*t + 1
    poly_integer_mul(&f,&fl,t);
    poly_integer_add(&f,&f,(cuyasheint_t)1);
    poly_reduce(&f, nphi, nq); 

  try{
      log_debug("will try to compute fInv");
      // poly_invmod(&fInv,&f,nphi, nq); 
      fInv = f;
      log_debug("fInv computed.");
      // log_notice("fInv: " + poly_print(&fInv));
      break;
    } catch (exception& e)
    {
      log_warn("f has no modular inverse.");
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
  // log_notice("f: " + poly_print(&f));

  // ff = f*f
  poly_mul(&ff,&f,&f);
  poly_reduce(&ff,nphi,nq);
    
  // tff = ff*t
  poly_integer_mul(&tff,&ff,t);
  poly_reduce(&tff,nphi,nq);

  // Sample
  poly_t g;
  xkey.get_sample(&g, nphi-1);
  // log_notice("g: " + poly_print(&g));
  // h = fInv*g*t
  poly_mul(&h, &fInv,&g);
  poly_integer_mul(&h, &h, t);
  poly_reduce(&h, nphi, nq);

  // log_notice("h: " + poly_print(&h));
  /////////
  // q/t //
  /////////
  q = (NTL::power2_ZZ(nq)-1);
  delta = q/t; 

  ///////////////////
  // Compute gamma //
  ///////////////////
  gamma.resize(lwq);

  ///////////
  // TO-DO //
  ///////////
}
poly_t Yashe::encrypt(poly_t m){
  // log_notice("Encrypt");

	// Sample
  poly_t ps,e; 
  xerr.get_sample(&ps,nphi-1);
  xerr.get_sample(&e,nphi-1);
  poly_reduce(&e,nphi,nq);
  poly_reduce(&ps,nphi,nq);
  // log_notice("ps: " + poly_print(&ps));
  // log_notice("e: " + poly_print(&e));
  
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
	poly_biginteger_mul(&mdelta,&m,delta);

  // log_notice("m * delta: " + poly_print(&mdelta));

	// 
	poly_add(&e,&e,&mdelta);
  // log_notice("mdelta + e: " + poly_print(&e));
	
  //
	poly_mul(&ps,&ps,&h);
  // log_notice("ps * h: " + poly_print(&ps));

	//
	poly_t c;
	poly_init(&c);
	poly_add(&c,&e,&ps);

  //
  poly_reduce(&c, nphi, nq);
  // log_notice("c: " + poly_print(&c));
	return c;
}

poly_t Yashe::decrypt(poly_t c){
  // log_notice("Decrypt");

	poly_t m;
	poly_init(&m);

	poly_mul(&m, &f, &c);
	poly_reduce(&m, nphi, nq);
  // log_notice("[c*f]_q \\in R: " + poly_print(&m));

  poly_integer_mul(&m, &m, t);
  // log_notice("[c*f]_q \\in R * t: " + poly_print(&m));

	// division by q to the nearest
	for(int i = 0; i <= poly_get_deg(&m); i++){
    ZZ coeff = poly_get_coeff(&m,i);
    ZZ diff = coeff % q;
	  if(2*diff > q)
	   poly_set_coeff(&m,i,coeff/q +1);
	  else
	    poly_set_coeff(&m,i,coeff/q);
  }

  return m;
}