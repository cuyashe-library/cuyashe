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
#include "polynomial.h"
#include "ciphertext.h"
#include "distribution.h"
#include "cuda_ciphertext.h"

class Yashe{
  private:
    Distribution xkey;
    Polynomial ps;
    Distribution xerr;
    Polynomial e;

  public:
    static int d;
    static Polynomial phi;
    static ZZ q;
    static bn_t qDiv2;
    static cuyasheint_t t;
    static ZZ delta;
    static ZZ w;
    static std::vector<Polynomial> gamma;
    static Polynomial h;
    static Polynomial f;
    static Polynomial ff;
    static int lwq;
    static ZZ WDMasking;
    static std::vector<Polynomial> P;
    Yashe(){
      const int sigma_err = 8;
      const float gaussian_std_deviation = sigma_err*0.4;
      const int gaussian_bound = sigma_err*6;
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    Yashe(float gaussian_std_deviation, int gaussian_bound){
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    void generate_keys();
    Ciphertext encrypt(Polynomial m);
    Polynomial decrypt(Ciphertext c);
};

#endif
