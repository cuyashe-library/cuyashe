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

/**
 * Generates an uniforme sample on the CRT domain
 * @param p      [description]
 * @param mod    [description]
 * @param degree [description]
 */
void Distribution::generate_sample(poly_t *p,int mod,int degree){
  callCuGetUniformSample(  p->d_coefs, 
                            CRTPrimes.size(), 
                            mod);
  p->status = CRTSTATE;
}

void Distribution::get_sample(poly_t *p,int degree,int spacing){
  poly_init(p);
  int mod;
  // int rec;
  // int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
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
  return;
}


void Distribution::get_sample(poly_t *p, int degree){
  poly_init(p);

  int mod;
  // int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
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
  return ;
}