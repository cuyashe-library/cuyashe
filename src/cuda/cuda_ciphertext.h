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
#ifndef CUDA_CIPHERTEXT_H
#define CUDA_CIPHERTEXT_H
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_bn.h"
#include "polynomial.h"
#include "yashe.h"

template <int WORDLENGTH = 32>
extern __global__ void cuWordecomp(bn_t **P,bn_t *a,int lwq, int N);
void callCuWordecomp(dim3 gridDim, dim3 blockDim, cudaStream_t stream, int WORDLENGTH, bn_t *d_P, bn_t *a, int lwq, int N);
__host__ __device__ void convert_64_to_32(uint32_t *a,uint64_t *b,int n);
__host__ __device__ void convert_32_to_64(uint64_t *a, uint32_t *b, int n);

template <int WORDLENGTH>
__host__ void callWordDecomp(	std::vector<Polynomial> *P,
								bn_t *a,
								int lwq,
								int N,
								cudaStream_t stream
							){

	/**
	 * P is a collection of lwq arrays of size (deg+1)
	 */

	const int size = N*lwq;

	// Worddecomp
	const int ADDGRIDXDIM = (size%32 == 0? size/32 : size/32 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(32);

	callCuWordecomp(gridDim,blockDim,stream,WORDLENGTH,(P->at(0).d_bn_coefs),a,lwq, N);

	for(int i = 0; i < lwq; i++){
		P->at(i).set_icrt_computed(true);
		P->at(i).set_crt_computed(false);
		P->at(i).set_itransf_computed(false);
		P->at(i).set_transf_computed(false);
		P->at(i).set_host_updated(false);
	}

	// result = cudaDeviceSynchronize();
	// assert(result == cudaSuccess);
}


__host__ void callCiphertextMulAux(	bn_t *g, 
									ZZ q,
									int N, 
									cudaStream_t stream);
__host__ void callMersenneDiv(bn_t *g, ZZ q,int N, cudaStream_t stream);

#endif