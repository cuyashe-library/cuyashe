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
#include "distribution.h"
#include "../yashe/yashe.h"

/**
 * Generates an uniforme sample on the CRT domain
 * @param p      [description]
 * @param mod    [description]
 * @param degree [description]
 */
void Distribution::generate_sample(poly_t *p,int mod,int degree){
  callCuGetUniformSample(  p->d_bn_coefs, 
                           degree,
                           CRTPrimes.size(), 
                           mod);
  callCRT(p->d_bn_coefs,
      CUDAFunctions::N,
      p->d_coefs,
      CUDAFunctions::N,
      CRTPrimes.size(),
      0x0
  );
  p->status = CRTSTATE;
}

void Distribution::get_sample(poly_t *p, int degree){
  
  int mod;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      callCuGetNormalSample(p->d_bn_coefs, degree, gaussian_bound, gaussian_std_deviation, CRTPrimes.size());
      callCRT(p->d_bn_coefs,
          CUDAFunctions::N,
          p->d_coefs,
          CUDAFunctions::N,
          CRTPrimes.size(),
          0x0
        );
      p->status = CRTSTATE;
      return;
    break;
    case BINARY:
      mod = 2;
    break;
    case NARROW:
      mod = 2;
    break;
    default:
      mod = 100;
    break;
  }

  generate_sample(p,mod,degree);
}