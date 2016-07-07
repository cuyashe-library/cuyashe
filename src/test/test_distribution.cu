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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "../cuda/cuda_distribution.h"
#include "../distribution/distribution.h"
#include "../cuda/operators.h"
#include "../aritmetic/polynomial.h"

NTL_CLIENT

int main(void){
    cudaError_t result;
    //////////
    // Init //
    // 
    // Distribution
    const int degree = 4096;
    const int sigma_err = 8;
    const float gaussian_std_deviation = sigma_err*0.4;
    const int gaussian_bound = sigma_err*6;
    Distribution xkey = Distribution(NARROW);
    Distribution xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    
    // Poly
    ZZ q;
    bn_t Q;

    int mersenne_n = 5;
    q = NTL::power2_ZZ(mersenne_n) - 1;
    get_words(&Q,q);
    gen_crt_primes(q,degree);
    CUDAFunctions::init(degree);

    poly_t a;
    poly_init(&a);

    cuyasheint_t *h_array;
    h_array = (cuyasheint_t*) malloc (CRTPrimes.size()*degree*sizeof(cuyasheint_t));
    //////////

    xkey.get_sample( &a,
                      degree);

    result = cudaMemcpy(h_array,a.d_coefs, CRTPrimes.size()*degree*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    std::cout << "Narrow distribution: ";
    for(int i = 0 ; i < degree; i++)
      std::cout << h_array[i] << " ";
    std::cout << std::endl  << std::endl;

    result = cudaMemset(a.d_coefs, 0 , CRTPrimes.size()*degree*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    xerr.get_sample( &a,
                      degree);

    result = cudaMemcpy(h_array,a.d_coefs, CRTPrimes.size()*degree*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    std::cout << "Discrete gaussian distribution: ";
    for(int i = 0 ; i < degree; i++)
      std::cout << h_array[i] << " ";
    std::cout << std::endl  << std::endl;

    return 0;
}
