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
#include "cuda_distribution.h"

__global__ void setup_kernel ( curandState * states, unsigned long seed ){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed+tid*4, tid, 0, &states[tid] );
}

__host__ void Distribution::call_setup_kernel(){
	const int N = MAX_DEGREE;
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	setup_kernel<<<gridDim,blockDim,0>>>(states,SEED);
	assert(cudaGetLastError() == cudaSuccess);
}


__global__ void generate_narrow_random_numbers(	bn_t *coefs,
												curandState *states,
												int N,
												int spacing,
												int NPrimes,
												int mod ) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid <= N){	
    	// This is not a "narrow" distribution, as defined in [Bos et al. 2013],
    	// [-1,0,1], but a binary distribution. However, it is a pain to deal 
    	// with negative values on unsigned integers.
    	// Í'm not sure this is secure enough 
    	// 
    	int value = llrintf(curand_uniform(&states[tid])); 
		// This is our guarantee that the polynomial will assume the desired degree
		value += (tid == N && value == 0); 
    	coefs[tid].dp[0] = value;
    	coefs[tid].used = 1;
    	bn_zero_non_used(&coefs[tid]);
    }
        
}


__global__ void generate_normal_random_numbers(	bn_t *coefs,
												curandState *states,
												int N,
												int spacing,
												float mean, 
												float stddev,
												int NPrimes) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N){	
    	int value = llrintf(curand_normal (&states[tid])*stddev + mean); 
		// This is our guarantee that the polynomial will assume the desired degree
		value += (tid == N && value == 0); 
    	coefs[tid].dp[0] = value;
    	coefs[tid].used = 1;
    	bn_zero_non_used(&coefs[tid]);
    }
        
}

__host__  void Distribution::callCuGetUniformSample(	bn_t *coefs,
														int N,
														int NPrimes,
														int mod ){
	/**
	 * Generates N random integers
	 */
	
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	/** 
	 * Generate values
	 */
	assert(N <= MAX_DEGREE);
	generate_narrow_random_numbers<<<gridDim,blockDim,0,NULL>>>( 	coefs,
																	states,
																	N,
																	CUDAFunctions::N,
																	NPrimes,
																	mod );
	assert(cudaGetLastError() == cudaSuccess);
}

__host__ void Distribution::callCuGetNormalSample(	bn_t *coefs,
													int N,
													float mean,
													float stddev,
													int NPrimes){
		/**
	 * Generates N random integers
	 */
	
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	/** 
	 * Generate values
	 */
	assert(N <= MAX_DEGREE);
	generate_normal_random_numbers<<<gridDim,blockDim,0,NULL>>>( 	coefs,
																	states,
																	N,
																	CUDAFunctions::N,
																	mean,
																	stddev,
																	NPrimes );
	assert(cudaGetLastError() == cudaSuccess);
}
