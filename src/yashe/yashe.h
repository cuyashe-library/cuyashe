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
#include "../settings.h"
#include "../aritmetic/polynomial.h"
#include "../distribution/distribution.h"
#include "../cuda/cuda_ciphertext.h"

struct ciphertext {
  poly_t p;
  int level = 0;
} typedef cipher_t;

#include "ciphertext.h"

class Yashe{
  private:
    Distribution xkey;
    Distribution xerr;
    poly_t ps;
    poly_t e;
    poly_t fl;
    poly_t g;

  public:
    static int nphi; // R_q degree
    static int nq; //
    static ZZ q; // 
    static bn_t Q; // 
    static bn_t UQ; // 
    static bn_t qDiv2; // q/2
    static poly_t t; //
    static poly_t delta; // q/t
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

      /**
       * Initialization of samples
       *
       * This reduces the time need to sampling but turns this code non-thread-safe
       */
      poly_init(&ps); 
      poly_init(&e);
      poly_init(&fl);
      poly_init(&g);
      //
      poly_init(&h); 
      poly_init(&f); 
      poly_init(&ff);
      poly_init(&tf);
      poly_init(&tff);
      poly_init(&delta);
      poly_init(&mdelta);
      poly_init(&t);

    };
    Yashe(float gaussian_std_deviation, int gaussian_bound){
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    void generate_keys();
    void encrypt(cipher_t *c, poly_t m);
    void decrypt(poly_t *m, cipher_t c);
};

#endif
