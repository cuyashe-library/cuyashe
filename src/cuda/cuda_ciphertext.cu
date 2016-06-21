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

/**
 * Computes x/q, where x is a bit integer and q is a mersenne prime
 * @param  x      [description]
 * @param  q_bits [description]
 * @return        [description]
 */
__device__ __inline__ void mersenneDiv(	bn_t *x,
							int q_bits
						){


		// q_bits right shift
		uint64_t carry = bn_rshb_low(x->dp,
									x->dp,
									x->used,
									q_bits);
		if(carry)
			bn_rshb_low(x->dp,
						x->dp,
						x->used,
						q_bits);
		x->dp -= (q_bits / WORD) + (q_bits % WORD != 0);
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
									int q_bits,
									bn_t qDiv2,
									int N){
	/**
	 * This kernel should be executed with N threads
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		// Divides g by q
		bn_t *coef = &g[tid];
		mersenneDiv(coef,q_bits);

		// Checks if g%q >= q/2.
		if(bn_cmp_abs(coef,&qDiv2) != CMP_LT){
			// If it is, add one.
			bn_add1_low(coef->dp, coef->dp, 1, coef->used);
		}

	}
}

/**
 * Divides a big integer g by a mersenne prime q
 * @param g      [description]
 * @param q_bits [description]
 * @param N      [description]
 */
__global__ void cuMersenneDiv( bn_t *g, 
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
		mersenneDiv(coef,q_bits);
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
__host__ void callCiphertextMulAux(bn_t *g, ZZ q,int N, cudaStream_t stream){
	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuCiphertextMulAux<<<gridDim, blockDim, 0, stream>>>(g, NTL::NumBits(q),Yashe::qDiv2, N);
	assert(cudaGetLastError() == cudaSuccess);
}

__host__ void callMersenneDiv(bn_t *g, ZZ q,int N, cudaStream_t stream){

	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuMersenneDiv<<<gridDim, blockDim,0, stream>>>(g,NTL::NumBits(q),N);
	assert(cudaGetLastError() == cudaSuccess);
}