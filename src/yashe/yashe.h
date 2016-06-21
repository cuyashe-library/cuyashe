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
#ifndef YASHE_H
#define YASHE_H

#include <NTL/ZZ.h>
#include "../aritmetic/polynomial.h"
#include "../distribution/distribution.h"

class Yashe{
  private:
    Distribution xkey;
    Distribution xerr;

  public:
    static int nphi; // R_q degree
    static int nq; //
    static ZZ q; // 
    static bn_t qDiv2; // q/2
    static cuyasheint_t t; //
    static ZZ delta; // q/t
    static ZZ w; // 
    static std::vector<poly_t> gamma; //
    static poly_t h; // 
    static poly_t f; // 
    static poly_t ff; //
    static poly_t tf; //
    static poly_t tff; //
    static poly_t mdelta; //
    static int lwq; // log_w q
    static ZZ WDMasking;
    static std::vector<poly_t> P;
    Yashe(){
      const int sigma_err = 8;
      const float gaussian_std_deviation = sigma_err*0.4;
      const int gaussian_bound = sigma_err*6;
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);

      poly_init(&h); 
      poly_init(&f); 
      poly_init(&ff);
      poly_init(&tf);
      poly_init(&tff);
      poly_init(&mdelta);

    };
    Yashe(float gaussian_std_deviation, int gaussian_bound){
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    void generate_keys();
    poly_t encrypt(poly_t m);
    poly_t decrypt(poly_t c);
};

#endif
