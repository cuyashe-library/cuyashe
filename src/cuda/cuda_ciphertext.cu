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
  printf("cuWordecomp: Nothing to do");
}
/**
 * Computes WordDecomp for W = 2^32
 *
 * This method receives lwq arrays of coefficients concatenated and decomposes
 * each coefficient of a. Each coefficient of arrays in P stores a fraction of
 * the related coefficient in a.
 * 
 * @param P   A vector with N*(log_wq) elements
 * @param a   [description]
 * @param lwq [description]
 */
template<>
__global__ void cuWordecomp<32>(bn_t *P,bn_t *a,int lwq, int N){
	/**
	 * This kernel should be executed by N*lwq threads
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Coefficient
	const int cid = tid % N;
	// Decomposition to compute
	const int did = tid / N;

	if( tid < N*lwq ){
		bn_zero(&P[cid + did*N]);
		
		// Selects the first or second half of a 64 bits word 
		uint64_t half_word =  a [ cid ].dp [ did / 2 ];
		int i = ( did % 2 ) * 32;
		half_word >>= i;

		// Shift 
		P[cid + did*N].dp[0] = (uint32_t)( half_word );
		P[cid + did*N].used = 1;
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


void callCuWordecomp(	cudaStream_t stream, 
						int WORDLENGTH, 
						bn_t *d_P, 
						bn_t *a, 
						int lwq, 
						int N ){
	const int size = N*lwq;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	if(WORDLENGTH == 32)
		cuWordecomp<32><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	// else if(WORDLENGTH == 64)
	// 	cuWordecomp<64><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	else
		throw "Unknown WORDLENGTH";
	cudaError_t result = cudaGetLastError();
	assert(result == cudaSuccess);
}

/**
 * Computes x%q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ void mersenneMod(	bn_t *x,
								bn_t *q,
								int q_bits){
	bn_adjust_used(x);
	if(x->used == 0)
		return;
	
	///////////////
	// a%q //
	///////////////
	int check;
	int carry;	
	bn_t x_copy;
	cuyasheint_t dp[STD_BNT_WORDS_ALLOC];
	x_copy.alloc = x->alloc;
	x_copy.used = x->used;
	x_copy.sign = x->sign;
	x_copy.dp = dp;
	
	while(bn_cmp_abs(x,q) != CMP_LT){
		bn_copy(&x_copy, x);

		///////////
		// SHIFT //
		///////////
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

		/////////////////
		// BITWISE AND //
		/////////////////
		// x_copy = x&q
		bn_bitwise_and(&x_copy, q);
		x_copy.used = q->used;
		bn_adjust_used(&x_copy);
		bn_zero_non_used(&x_copy);

		///////////////////////////
		// REMAINDER COMPUTATION //
		///////////////////////////
		// x = (x>>s) + (x&q)
		int nwords = max_d(x->used,x_copy.used);
	    carry = bn_addn_low(x->dp, x->dp, x_copy.dp,nwords);
	    x->used = nwords;

	    /* Equivalent to "If has a carry, add as last word" */
	    x->dp[x->used] = carry;
	    x->used += (carry > 0);

	    // If x still bigger than q, x = x - q
		check = bn_cmp_abs(x, q);
		if(check != CMP_LT)
			bn_subn_low(x->dp,x->dp, q->dp, x->used);
	}

    if(x->sign == BN_NEG){
      /* Accumulate count! */
	  // carry = bn_add1_low(x->dp, x->dp, 1, x->used);
      /* Equivalent to "If has a carry, add as last word" */
      // quot->dp[quot->used] = carry;
	  // quot->used += (carry > 0);
      // q - ((a-b) % q)
      carry = bn_subn_low(  x->dp,
                            q->dp,
                            x->dp,
                            q->used );

      assert(carry == BN_POS);
      x->used = q->used;
      bn_adjust_used(x);

      x->sign = BN_POS;
    }

}

/**
 * Computes x/q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ void mersenneDiv(	bn_t *quot,
								bn_t *q,
								int q_bits){
	if(quot->used == 0)
		return;
	
	bn_adjust_used(quot);

	bn_t aux;
	cuyasheint_t dp_aux[STD_BNT_WORDS_ALLOC];
	aux.alloc = STD_BNT_WORDS_ALLOC;
	aux.used = 0;
	aux.sign = BN_POS;
	aux.dp = dp_aux;
	bn_zero_non_used(&aux);
	bn_copy(&aux,quot);
	quot->used = 0;
	bn_zero_non_used(quot);

	// Counter
	//  At end we add count to quot
	int count = -1;
	
	/**
	 * Iterates until aux < q
	 */
	int carry;
	while(aux.used > 0){
		count += 1;

		///////////
		// SHIFT //
		///////////
		// x = x>>s
		bn_rshd_low(  aux.dp,
		              aux.dp,
		              aux.used,
		              q_bits/WORD ); 
		aux.used -= (q_bits/WORD>0)?q_bits/WORD:0;
		bn_rshb_low(  aux.dp,
		              aux.dp,
		              aux.used,
		              q_bits%WORD );

		bn_adjust_used(&aux);
		bn_zero_non_used(&aux);

	    //////////////////////////
	    // QUOTIENT COMPUTATION //
	    //////////////////////////
	    /* Accumulate the quotient! */
		int nwords = max_d(aux.used,quot->used);
		if(quot->used >= aux.used)
		    carry = bn_addn_low(quot->dp, quot->dp, aux.dp, nwords);
		else
		    carry = bn_addn_low(quot->dp, aux.dp, quot->dp, nwords);
		    quot->used = nwords;

	    /* Equivalent to "If has a carry, add as last word" */
	    quot->dp[quot->used] = carry;
	    quot->used += (carry > 0);

	}	

	if(quot->sign == BN_NEG){
	//     /* Accumulate count! */
		carry = bn_add1_low(quot->dp, quot->dp, 1, quot->used);
	//      Equivalent to "If has a carry, add as last word" 
	    quot->dp[quot->used] = carry;
	    quot->used += (carry > 0);
	}
}

/**
 * Computes x/q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ void mersenneModDiv(	bn_t *quot,
								bn_t *rem,
								bn_t *q,
								int q_bits){
	if(quot->used == 0)
		return;
	
	bn_adjust_used(quot);

	bn_t aux;
	cuyasheint_t dp_aux[STD_BNT_WORDS_ALLOC];
	aux.alloc = STD_BNT_WORDS_ALLOC;
	aux.used = 0;
	aux.sign = BN_POS;
	aux.dp = dp_aux;
	bn_zero_non_used(&aux);

	// quot->sign = rem->sign;
	quot->used = 0;
	bn_zero_non_used(quot);
	bn_copy(&aux,rem);

	/**
	 * Iterates until aux < q
	 */
	int carry;
	int check = bn_cmp_abs(rem, q);
	while(check != CMP_LT){
		bn_copy(&aux,rem);

		///////////
		// SHIFT //
		///////////
		// x = x>>s
		bn_rshd_low(  rem->dp,
		              rem->dp,
		              rem->used,
		              q_bits/WORD ); 
		rem->used -= (q_bits/WORD>0)?q_bits/WORD:0;
		bn_rshb_low(  rem->dp,
		              rem->dp,
		              rem->used,
		              q_bits%WORD );

		bn_adjust_used(rem);
		bn_zero_non_used(rem);

	    //////////////////////////
	    // QUOTIENT COMPUTATION //
	    //////////////////////////
	    /* Accumulate the quotient! */
		int nwords = max_d(rem->used,quot->used);
		if(quot->used >= rem->used)
		    carry = bn_addn_low(quot->dp, quot->dp, rem->dp, nwords);
		else
		    carry = bn_addn_low(quot->dp, rem->dp, quot->dp, nwords);
		    quot->used = nwords;

	    /* Equivalent to "If has a carry, add as last word" */
	    quot->dp[quot->used] = carry;
	    quot->used += (carry > 0);
		
		/////////////////
		// BITWISE AND //
		/////////////////
		// aux = x&q
		bn_bitwise_and(&aux, q);
		aux.used = q->used;
		bn_adjust_used(&aux);
		bn_zero_non_used(&aux);

		///////////////////////////
		// REMAINDER COMPUTATION //
		///////////////////////////
		// x = (x>>s) + (x&q)
		nwords = max_d(rem->used,aux.used);
		if(rem->used >= aux.used)
	    	carry = bn_addn_low(rem->dp, rem->dp, aux.dp,nwords);
	    else
	    	carry = bn_addn_low(rem->dp, aux.dp, rem->dp, nwords);
	    rem->used = nwords;

	    /* Equivalent to "If has a carry, add as last word" */
	    rem->dp[rem->used] = carry;
	    rem->used += (carry > 0);

	    // If x still bigger than q, x = x - q
		check = bn_cmp_abs(rem, q);
		// if(check != CMP_LT)
			// bn_subn_low(rem->dp,rem->dp, q->dp, rem->used);
	}	

	if(quot->sign == BN_NEG){
		//////////////
		// Quotient //
		//////////////
		carry = bn_add1_low(quot->dp, quot->dp, 1, quot->used);
		/* Equivalent to "If has a carry, add as last word" */
	    quot->dp[quot->used] = carry;
	    quot->used += (carry > 0);

	    ///////////////
	    // Remainder //
	    ///////////////
		carry = bn_subn_low(  	rem->dp,
			                    q->dp,
			                    rem->dp,
			                    q->used );

		assert(carry == BN_POS);
		rem->used = q->used;
		bn_adjust_used(rem);

		rem->sign = BN_POS;
	}
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

		mersenneModDiv(coef, &coef_copy, &q, q_bits);

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
__global__ void cuMersenneMod( bn_t *g, 
	                           bn_t q,
								int q_bits,
								int N){
	/**
	 * This kernel should be executed with N threads
	 *
	 * It supposes that q is a mersenne prime
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N)
		mersenneMod(&g[tid],&q,q_bits);
	
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

__host__ void callMersenneMod(bn_t *g, bn_t q,int nq, int N, cudaStream_t stream){

	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuMersenneMod<<<gridDim, blockDim,0, stream>>>(g, q, nq,N);
	assert(cudaGetLastError() == cudaSuccess);
}
