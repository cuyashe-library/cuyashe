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
#include "cuda_ciphertext.h"

template <int WORDLENGTH>
/**
 * cuWordecomp computes de word decomposition of every coefficient in 32 bit words.
 * The coefficients are destroyed inside the kernel
 * @param P   [description]
 * @param a   [description]
 * @param lwq [description]
 * @param N   [description]
 */
__global__ void cuWordecomp(bn_t *P,bn_t *a,int lwq, int N){
  printf("Nothing to do");
}
/**
 * Computes WordDecomp for W = 2^32
 * @param P   A vector with N*(log_wq) elements
 * @param a   [description]
 * @param lwq [description]
 */
template<>
__global__ void cuWordecomp<32>(bn_t *P,bn_t *a,int lwq, int N){
	/**
	 * This kernel should be executed by lwq thread
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const int did = tid % N;// Decomposition id
	const int cid = tid / N;// Coefficient id

	if(tid < N){
		bn_zero(&P[tid]);
		P[tid].dp[cid] = (a[cid].dp[did/2] >> ((did%2)*32))&(4294967296-1); 
		P[tid].used = 1;
	}
}

/**
 * Computes WordDecomp for W = 2^64
 * @param P   [description]
 * @param a   [description]
 * @param lwq [description]
 */
// template<>
// __global__ void cuWordecomp<64>(bn_t **P,bn_t *a,int lwq, int N){	/**
// 	 * This kernel should be executed by N*lwq thread
// 	 */
// 	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
// 	const int did = tid % N;// Decomposition id
// 	const int cid = tid / N;// Coefficient id

// 	if(tid < N*lwq){
// 		P[tid].dp[cid] = (a[cid].dp[did/2] >> ((did%2)*32))&(4294967296-1); 
// 	}
// }


void callCuWordecomp(	dim3 gridDim, 
						dim3 blockDim, 
						cudaStream_t stream, 
						int WORDLENGTH, 
						bn_t *d_P, 
						bn_t *a, 
						int lwq, 
						int N){
	if(WORDLENGTH == 32)
		cuWordecomp<32><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	// else if(WORDLENGTH == 64)
	// 	cuWordecomp<64><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	else
		throw "Unknown WORDLENGTH";
	cudaError_t result = cudaGetLastError();
	assert(result == cudaSuccess);
}


__host__ __device__ int bn_count_bits( bn_t *a){
	if(a->used == 0)
		return 0;

	int count = 0;
	cuyasheint_t last_word = a->dp[a->used-1];
	while(last_word > 0){
		count++;
		last_word = (last_word>>1);
	}

	return count;		
}

/**
 * Computes x%q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ void mersenneDiv(	bn_t *x,
								bn_t *q,
								int q_bits){
	if(x->used == 0)
	return;


	bn_adjust_used(x);
	
	int check = (x->used-1)*WORD;
	cuyasheint_t last_word = x->dp[x->used-1];
	while(last_word > 0){
		check++;
		last_word = (last_word>>1);
	}

	assert(check < 2*q_bits);

	///////////////
	// a%q //
	///////////////
	check = bn_cmp_abs(x, q);
	if(check == CMP_LT)
		return;
	else if(check == CMP_EQ){
		x->used = 0;
		return;
	}

	int carry;
	// > q
	bn_t x_copy;
	cuyasheint_t dp[STD_BNT_WORDS_ALLOC];
	x_copy.alloc = x->alloc;
	x_copy.used = x->used;
	x_copy.sign = x->sign;
	x_copy.dp = dp;
	bn_copy(&x_copy, x);

	// x = x>>s
	bn_rshd_low(  x->dp,
				              x->dp,
				              x->used,
				              q_bits/WORD ); 
	x->used -= (q_bits/WORD>0)?q_bits/WORD:0;
	bn_rshb_low(  x->dp,
				              x->dp,
				              x->used,
				              q_bits%WORD );
	// x->dp[0] += carry;
	bn_adjust_used(x);
	bn_zero_non_used(x);
	// x_copy = x&q
	bn_bitwise_and(&x_copy, q);
	x_copy.used = q->used;
	bn_adjust_used(&x_copy);
	bn_zero_non_used(&x_copy);

	// x = (x>>s) + (x&q)
	int nwords = max_d(x->used,x_copy.used);
    carry = bn_addn_low(x->dp, x->dp, x_copy.dp,nwords);
    x->used = nwords;

    /* Equivalent to "If has a carry, add as last word" */
    x->dp[x->used] = carry;
    x->used += (carry > 0);

    // If x still bigger than q, x = x - q
	check = bn_cmp_abs(x, q);
	if(check == CMP_LT)
		return;
	else{
		bn_subn_low(x->dp,x->dp, q->dp, x->used);
		return;
	}
}

/**
 * Computes x/q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ void mersenneDivQ(	bn_t *x,
								bn_t *q,
								int q_bits){
	if(x->used == 0)
	return;


	bn_adjust_used(x);
	
	int check = (x->used-1)*WORD;
	cuyasheint_t last_word = x->dp[x->used-1];
	while(last_word > 0){
		check++;
		last_word = (last_word>>1);
	}

	assert(check < 2*q_bits);

	///////////////
	// a%q //
	///////////////
	check = bn_cmp_abs(x, q);
	if(check == CMP_LT)
		return;
	else if(check == CMP_EQ){
		x->used = 0;
		return;
	}

	// x = x>>s
	bn_rshd_low(  x->dp,
				              x->dp,
				              x->used,
				              q_bits/WORD ); 
	x->used -= (q_bits/WORD>0)?q_bits/WORD:0;
	bn_rshb_low(  x->dp,
				              x->dp,
				              x->used,
				              q_bits%WORD );
	bn_adjust_used(x);
	bn_zero_non_used(x);
}

/**
 * Computes g/q and g%q and set "output" according the result   
 * This function works inplace.
 * 
 * @param output [description]
 * @param g      [description]
 * @param q      [description]
 * @param N
 */
__global__ void cuCiphertextMulAux(	 
									bn_t *g, 
									bn_t q,
									int q_bits,
									bn_t qDiv2,
									int N){
	/**
	 * This kernel should be executed with N threads
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		bn_t *coef = &g[tid];

		bn_t coef_copy;
		cuyasheint_t dp[STD_BNT_WORDS_ALLOC];
		coef_copy.alloc = coef->alloc;
		coef_copy.used = coef->used;
		coef_copy.sign = coef->sign;
		coef_copy.dp = dp;
		bn_copy(&coef_copy, coef);

		// Div g by q
		mersenneDivQ(coef,&q,q_bits);

		// Modular reduction g by q
		mersenneDiv(&coef_copy,&q,q_bits);

		// Checks if g%q >= q/2.
		if(bn_cmp_abs(&coef_copy,&qDiv2) != CMP_LT){
			// If it is, add one.
			bn_add1_low(coef->dp, coef->dp, 1, coef->used);
		}

		g[tid] = *coef;
	}
}

/**
 * Divides a big integer g by a mersenne prime q
 * @param g      [description]
 * @param q_bits [description]
 * @param N      [description]
 */
__global__ void cuMersenneDiv( bn_t *g, 
	                           bn_t q,
								int q_bits,
								int N){
	/**
	 * This kernel should be executed with N threads
	 *
	 * It supposes that q is a mersenne prime
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		bn_t *coef = &g[tid];
		mersenneDiv(coef,&q,q_bits);
	}
}

/**
 * Computes the second part of a ciphertext multiplication
 * @param P      [description]
 * @param g      [description]
 * @param q      [description]
 * @param N      [description]
 * @param stream [description]
 */
__host__ void callCiphertextMulAux(bn_t *g, bn_t q, int nq,int N, cudaStream_t stream){
	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuCiphertextMulAux<<<gridDim, blockDim, 0, stream>>>(g, q, nq,Yashe::qDiv2, N);
	assert(cudaGetLastError() == cudaSuccess);
}

__host__ void callMersenneDiv(bn_t *g, bn_t q,int nq, int N, cudaStream_t stream){

	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuMersenneDiv<<<gridDim, blockDim,0, stream>>>(g, q, nq,N);
	assert(cudaGetLastError() == cudaSuccess);
}