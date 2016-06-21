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
#include "../settings.h"
#include "../distribution/distribution.h"

__global__ void setup_kernel ( curandState * states, unsigned long seed )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, tid, 0, &states[tid] );
}

__host__ void Distribution::call_setup_kernel(){
	const int N = MAX_DEGREE;
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	setup_kernel<<<gridDim,blockDim,0>>>(states,SEED);
	assert(cudaGetLastError() == cudaSuccess);
}


__global__ void generate_random_numbers(cuyasheint_t* coefs, curandState *states, int N, int mod) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) 
        coefs[tid] = (cuyasheint_t)(curand_uniform(&states[tid])*mod);
    
}

__host__  void Distribution::callCuGetUniformSample(cuyasheint_t *coefs, int N, int mod){
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
	generate_random_numbers<<<blockDim,gridDim>>>(coefs,states,N,mod);
	assert(cudaGetLastError() == cudaSuccess);
	cudaDeviceSynchronize();
}

__host__ void Distribution::callCuGetNormalSample(cuyasheint_t *array, int N, float mean, float stddev){
	///////////
	// To-do //
	///////////

	curandGenerateLogNormal( gen, 
							(float*)array, 
							N,
							mean,
							stddev);


}