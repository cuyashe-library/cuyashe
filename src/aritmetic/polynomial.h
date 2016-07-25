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

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
#include <NTL/ZZ.h>
#include <NTL/ZZ_pEX.h>
#include <map>
#include <algorithm>
#include <sstream>
#include "../settings.h"
#include "../cuda/operators.h"
#include "../cuda/cuda_bn.h"
#include "../logging/logging.h"

NTL_CLIENT

// Stores the polynomial degree that should be used
extern int OP_DEGREE;

// Stores CRT primes
extern std::vector<cuyasheint_t> CRTPrimes;
extern ZZ CRTProduct;
extern std::vector<ZZ> CRTMpi;
extern std::vector<cuyasheint_t> CRTInvMpi;

// Three possible states:
// 
// 	* HOSTSTATE: data is updated on the host and stored in "coefs"
// 	* CRTSTATE: data is updated on the GPU and the residues are stored in "d_coefs"
// 	* TRANSSTATE: data is updated on the GPU and the transformed resides are stored in "d_coefs"
enum states {HOSTSTATE, CRTSTATE, TRANSSTATE};

struct polynomial {
	std::vector<ZZ> coefs;
	cuyasheint_t *d_coefs = NULL;
	int status = HOSTSTATE;
	bn_t *d_bn_coefs = NULL;
	#ifdef CUFFTMUL_TRANSFORM
	Complex *d_coefs_transf = NULL;
	#endif
} typedef poly_t;

/**
 * [poly_init description]
 * @param a [description]
 */
void poly_init(poly_t *a);

/**
 * [poly_free description]
 * @param a [description]
 */
void poly_free(poly_t *a);

/**
 * [poly_clear description]
 * @param a [description]
 */
void poly_clear(poly_t *a);

/**
 * [poly_copy_to_device description]
 * @param a [description]
 */
void poly_copy_to_device(poly_t *a);
/**
 * [poly_copy_to_host description]
 * @param a [description]
 */
void poly_copy_to_host(poly_t *a);

/**
 * [poly_demote description]
 * @param a [description]
 */
void poly_demote(poly_t *a);
/**
 * [poly_elevate description]
 * @param a [description]
 */
void poly_elevate(poly_t *a);

/**
 * [poly_crt description]
 * @param a [description]
 */
void poly_crt(poly_t *a);
/**
 * [poly_icrt description]
 * @param a [description]
 */
void poly_icrt(poly_t *a);

/**
 * returns the polynomial degree
 * @param  a [description]
 * @return   [description]
 */
int poly_get_deg(poly_t *a);

/**
 * polynomial addition
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */
void poly_add(poly_t *c, poly_t *a, poly_t *b);

/**
 * polynomial multiplication
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */

void poly_mul(poly_t *c, poly_t *a, poly_t *b);

/**
 * polynomial addition with an integer
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */
void poly_integer_add(poly_t *c, poly_t *a, cuyasheint_t b);

/**
 * polynomial multiplication with an integer
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */

void poly_integer_mul(poly_t *c, poly_t *a, cuyasheint_t b);

/**
 * [poly_biginteger_mul description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void poly_biginteger_mul(poly_t *c, poly_t *a, bn_t b);

/**
 * [poly_biginteger_mul description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void poly_biginteger_mul(poly_t *c, poly_t *a, ZZ b);

/**
 * [poly_reduce description]
 * @param f    [description]
 * @param nphi x^{nphi} - 1
 * @param nq   2^{nq} - 1
 */
void poly_reduce(poly_t *f, int nphi, bn_t Q, int nq);

/**
 * [poly_cyclotomic_reduction description]
 * @param f    [description]
 * @param nphi [description]
 */
void poly_cyclotomic_reduction(poly_t *f, int nphi);

/**
 * [poly_mersenne_reduction description]
 * @param a  [description]
 * @param q  [description]
 * @param nq [description]
 */
void poly_mersenne_reduction(poly_t *a, bn_t q, int nq);

/**
 * computes the polynomial inverse in R_q
 * @param fInv [description]
 * @param f    [description]
 * @param nphi x^{nphi} - 1
 * @param nq   2^{nq} - 1
 */
void poly_invmod(poly_t *fInv, poly_t *f, int nphi, int nq);

/**
 * print a polynomial
 * @param a [description]
 */
std::string poly_print(poly_t *a);

/**
 * generates a set of primes for CRT
 * @param q      [description]
 * @param degree [description]
 */
void gen_crt_primes(ZZ q,cuyasheint_t degree);

/**
 * convert a ZZ to bn_t
 * @param b [description]
 * @param a [description]
 */
void get_words(bn_t *b,ZZ a);

/**
 * convert a bn_t to ZZ
 * @param  a [description]
 * @return   [description]
 */
ZZ get_ZZ(bn_t *a);

void get_words_host(bn_t *b,ZZ a);

/**
 * computes the reciprocal of q
 * @param  q [description]
 * @return   [description]
 */
bn_t get_reciprocal(ZZ q);

void compute_reciprocal(ZZ q);

bn_t get_reciprocal(bn_t q);

/**
 * [poly_set_coeff description]
 * @param a [description]
 * @param index [description]
 * @param c [description]
 */
void poly_set_coeff(poly_t *a, int index, ZZ c);

/**
 * [poly_get_coeff description]
 * @param a     [description]
 * @param index [description]
 */
ZZ poly_get_coeff(poly_t *a, int index);


void poly_demote(poly_t *a);
void poly_elevate(poly_t *a);

void poly_set_nth_cyclotomic(poly_t *a, int n);
void poly_clear(poly_t *a);
#endif