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
#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../settings.h"
#include "../aritmetic/polynomial.h"
 
enum kind_t
{
  DISCRETE_GAUSSIAN,
  BINARY,
  NARROW,
  UNIFORMLY,
  KINDS_COUNT
};

#define MAX_DEGREE 16384
#define SEED (unsigned long long)(7226776555987513888)// We derandomize for debug purposes

class Distribution{
  private:
  int kind;
  float gaussian_std_deviation;
  int gaussian_bound;
  curandGenerator_t gen;
  curandState *states;

  public:
  Distribution(kind_t kind, float std_dev, int bound){
    assert(kind == DISCRETE_GAUSSIAN);
    this->kind = kind;
    this->gaussian_std_deviation = std_dev;
    this->gaussian_bound = bound;

    curandStatus_t resultRand = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    resultRand = curandSetPseudoRandomGeneratorSeed(gen, 
                SEED);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    /**
     * Setup
    */
    cudaError_t result = cudaMalloc((void**)&states,MAX_DEGREE*sizeof(curandState));
    assert(result == cudaSuccess);
    call_setup_kernel();


  }
  Distribution(kind_t kind){
    assert(kind != DISCRETE_GAUSSIAN);
    assert(kind < KINDS_COUNT);
    this->kind = kind;

    curandStatus_t resultRand = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    resultRand = curandSetPseudoRandomGeneratorSeed(gen, 
                SEED);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    /**
     * Setup
    */
    cudaError_t result = cudaMalloc((void**)&states,MAX_DEGREE*sizeof(curandState));
    assert(result == cudaSuccess);
    call_setup_kernel();

  }
  Distribution(){
    this->kind = UNIFORMLY;

    curandStatus_t resultRand = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    resultRand = curandSetPseudoRandomGeneratorSeed(gen, 
                SEED);
    assert(resultRand == CURAND_STATUS_SUCCESS);

    /**
     * Setup
    */
    cudaError_t result = cudaMalloc((void**)&states,MAX_DEGREE*sizeof(curandState));
    assert(result == cudaSuccess);
    call_setup_kernel();

  }
  void get_sample(poly_t *p, int degree);
  void generate_sample(poly_t *p,int mod,int degree);
private:
  void callCuGetUniformSample(bn_t *coefs,int N, int NPrimes, int mod);
  void callCuGetNormalSample(bn_t *array, int N, float mean, float stddev, int NPrimes);
__host__ void call_setup_kernel();

};
#endif
