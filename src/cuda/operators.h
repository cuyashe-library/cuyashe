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
#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <NTL/ZZ.h>
#include "../settings.h"
#include "../cuda/cuda_bn.h"
#include "../aritmetic/polynomial.h"

NTL_CLIENT

#define MAX_PRIMES_ON_C_MEMORY 4096
typedef double2 Complex;
extern __constant__ cuyasheint_t CRTPrimesConstant[COPRIMES_BUCKET_SIZE];

__host__ bool is_power_of(uint64_t a, uint64_t b);
class CUDAFunctions{
  public:
    ////////////////////////
    // Mersenne operator  //
    ////////////////////////


  	static int N;
    static int std_bn_t_alloc;
    static int transform;
    //////////////////////////
    // CRT global variables //
    //////////////////////////
    static cuyasheint_t *d_inner_results;
    static cuyasheint_t *d_inner_results_used;

    /////////
    // NTT //
    /////////
    static cuyasheint_t wN;
    static cuyasheint_t *d_W;
    static cuyasheint_t *d_WInv;
    static cuyasheint_t *d_mulA;
    static cuyasheint_t *d_mulB;
    static cuyasheint_t *d_mulAux;
    ///////////
    ///////////
    // cuFFT //
    ///////////
    static cufftHandle plan;
    static Complex *d_mulComplexA;
    static Complex *d_mulComplexB;
    static Complex *d_mulComplexC;
    static cuyasheint_t* applyNTT(  cuyasheint_t *d_a,
                                    const int N,
                                    const int NPolis,
                                    int type,
                                    cudaStream_t stream);
    static void callPolynomialAddSub(   cuyasheint_t *c,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP,
                                            cudaStream_t stream);
    static void callPolynomialAddSubInPlace(cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP);
    static void callPolynomialcuFFTAddSub(Complex *c,
                                            Complex *a,
                                            Complex *b,
                                            int size,
                                            int OP,
                                            cudaStream_t stream);
    static void callPolynomialcuFFTAddSubInPlace(cudaStream_t stream,
                                            Complex *a,
                                            Complex *b,
                                            int size,
                                            int OP);

    static void executeNTTScale(    cuyasheint_t *a, 
                                    const int size, 
                                    const int N,
                                    cudaStream_t stream);
    static void executePolynomialMul(cuyasheint_t *c, 
                                    cuyasheint_t *a, 
                                    cuyasheint_t *b, 
                                    const int size, 
                                    cudaStream_t stream);
    static void executeCuFFTPolynomialMul( Complex *a, 
                                            Complex *b, 
                                            Complex *c, 
                                            int size, 
                                            cudaStream_t stream);
    static void executePolynomialAdd(cuyasheint_t *c, 
                                    cuyasheint_t *a, 
                                    cuyasheint_t *b, 
                                    const int size, 
                                    cudaStream_t stream);
    static void executeCopyIntegerToComplex(   Complex *d_a, 
                                                            cuyasheint_t *a,
                                                            const int size,
                                                            cudaStream_t stream);
    static void executeCopyAndNormalizeComplexRealPartToInteger(   cuyasheint_t *d_a, 
                                                                                cufftDoubleComplex *a,
                                                                                const int size,
                                                                                int N,
                                                                                cudaStream_t stream);
    static cuyasheint_t* callPolynomialMul(cuyasheint_t *output,
                                                        cuyasheint_t *a,
                                                        cuyasheint_t *b,
                                                        const int size,
                                                        cudaStream_t stream);
    static void callPolynomialOPInteger(   const int opcode,
                                                    cudaStream_t stream,
                                                    cuyasheint_t *b,
                                                    cuyasheint_t *a,
                                                    cuyasheint_t integer_array,
                                                    const int N,
                                                    const int NPolis);
    static void callPolynomialOPIntegerInplace(     const int opcode,
                                                    cudaStream_t stream,
                                                    cuyasheint_t *a,
                                                    cuyasheint_t integer,
                                                    const int N,
                                                    const int NPolis);
    static void callPolynomialcuFFTOPInteger(
                                                  const int opcode,
                                                  cudaStream_t stream,
                                                  Complex *b,
                                                  Complex *a,
                                                  cuyasheint_t integer,
                                                  const int N,
                                                  const int NPolis);
    static void callPolynomialcuFFTOPIntegerInplace(
                                                      const int opcode,
                                                      cudaStream_t stream,
                                                      Complex *a,
                                                      cuyasheint_t integer,
                                                      const int N,
                                                      const int NPolis);
    static void callPolynomialOPDigit( const int opcode,
                                            cudaStream_t stream,
                                            bn_t *b,
                                            bn_t *a,
                                            bn_t digit,
                                            const int N);
    static cuyasheint_t* callRealignCRTResidues(cudaStream_t stream,
                                            int oldSpacing,
                                            int newSpacing,
                                            cuyasheint_t *array,
                                            int residuesSize,
                                            int residuesQty);
    static void callNTT(const int N, const int NPolis,int RADIX, cuyasheint_t* dataI, cuyasheint_t* dataO,const int type);
    static void init(int N);
    static void write_crt_primes();
    
    static void callPolynomialReductionCoefs(   bn_t *a,
                                                const int half,
                                                const int N);
  private:
};
__device__ __host__ inline uint64_t s_rem (uint64_t a);
__device__ __host__  uint64_t s_mul(uint64_t a,
                                    uint64_t b);
#endif