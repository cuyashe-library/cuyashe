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
#ifndef CUDA_BN_H
#define CUDA_BN_H

#include <NTL/ZZ.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "../settings.h"
#include "operators.h"

NTL_CLIENT 


/**
 * Returns a bit mask to isolate the lowest part of a digit.
 *
 * @param[in] B			- the number of bits to isolate.
 */
#define MASK(B)				(((cuyasheint_t)1 << (B)) - 1)
#define MASK_32(B)				(((uint32_t)1 << (B)) - 1)
 

__host__  void bn_new(bn_t *a);
__host__ __device__ void bn_zero(bn_t *a);
__host__ __device__ void bn_zero_non_used(bn_t *a);
__host__ __device__ void bn_adjust_used(bn_t *a);
__host__ __device__ bool bn_is_zero(const bn_t* a);
__host__ __device__ void bn_set_dig(bn_t *a, cuyasheint_t digit);
__host__  void bn_free(bn_t *a);
__host__ void bn_grow(bn_t *a,const unsigned int new_size);
__host__ __device__ void bn_copy(bn_t *a, bn_t *b);
__host__ __device__ void bn_2_compl(bn_t *a);
__host__ __device__ int bn_count_bits(bn_t *a);
__host__ __device__ void bn_bitwise_and(bn_t *a, bn_t *b);
__host__ __device__ void bn_truncate(bn_t *a, int bits);
__host__ __device__ uint64_t bn_rshb_low(uint64_t *c, const uint64_t *a, int size, int bits);
__host__ __device__ uint32_t bn_rshb_low_32(uint32_t *c, const uint32_t *a, int size, int bits);
__host__ __device__ void bn_rshd_low(cuyasheint_t *c, const cuyasheint_t *a, int size, int digits);
__host__ __device__ void bn_lshd_low(cuyasheint_t *c, const cuyasheint_t *a, int size, int digits);	
__host__ __device__  cuyasheint_t bn_lshb_low(cuyasheint_t *c, const cuyasheint_t *a, int size, int bits);
__host__ __device__ uint64_t bn_addn_low(uint64_t *c, uint64_t *a, uint64_t *b, const int size );
__host__ __device__ uint32_t bn_addn_low_32(uint32_t *c, uint32_t *a, uint32_t *b, const int size );
__host__ __device__ uint64_t bn_subn_low(uint64_t * c, const uint64_t * a,const uint64_t * b, int size) ;
// __host__ __device__ uint64_t bn_subn_low(	uint64_t * c, const uint64_t * a, const uint64_t * b,  int size);
__host__ __device__ uint32_t bn_subn_low_32(	uint32_t * c, const uint32_t * a, const uint32_t * b,  int size);
__device__ void bn_muln_low(cuyasheint_t *c,
							const cuyasheint_t *a,
							const cuyasheint_t *b,
							int size 
						);
__device__ void bn_muld_low(cuyasheint_t * c, 
									const cuyasheint_t * a, 
									int sa,
									const cuyasheint_t * b, 
									int sb, 
									int l, 
									int h);
__device__ void bn_mod_barrt(	bn_t *C, 
								const bn_t A,
								const cuyasheint_t * m, 
								int sm, 
								const cuyasheint_t * u,
								int su);
__device__ void bn_divn_low( 	uint32_t *c, 
								uint32_t *d, 
								uint32_t *a, 
								int sa, 
								uint32_t *b, 
								int sb
							);
template<typename T>
__host__ __device__ int bn_cmpn_low(const T *a, const T *b, int size);
__host__ __device__ int bn_cmp_abs(	const bn_t *a,
									const bn_t *b);
__host__ __device__ uint64_t bn_add1_low(uint64_t *c, const uint64_t *a, uint64_t digit, int size);
__host__ __device__ uint32_t bn_add1_low_32(uint32_t *c, const uint32_t *a, uint32_t digit, int size);
__host__ __device__ uint64_t bn_sub1_low(	uint64_t *c, const uint64_t *a, uint64_t digit, int size);
__host__ __device__ uint32_t bn_sub1_low_32(	uint32_t *c, const uint32_t *a, uint32_t digit, int size);
__host__ __device__ uint64_t bn_mul1_low(uint64_t *c, const uint64_t *a, uint64_t digit, int size);
__host__ __device__ uint32_t bn_mul1_low(uint32_t *c, const uint32_t *a, uint32_t digit, int size);
__host__ __device__ cuyasheint_t bn_mod1_low(const cuyasheint_t *a,
											const int size,
											const uint32_t b);
__host__ void callCuModN(bn_t * c, bn_t * a,int NCoefs,
		const cuyasheint_t * m, int sm, const cuyasheint_t * u, int su,
		cudaStream_t stream);
__host__ int callBNGetDeg(bn_t *coefs, int N);
__host__ ZZ get_ZZ(bn_t *a);
__host__ __device__ int max_d(int a,int b);
__host__ __device__ int min_d(int a,int b);
__host__ __device__ uint64_t lessThan(uint64_t x, uint64_t y);
__host__ void callTestData(bn_t *coefs,int N);
__device__ int get_used_index(const cuyasheint_t *u,int alloc);
void callCRT(bn_t *coefs,const int used_coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);
void callICRT(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);




#endif
