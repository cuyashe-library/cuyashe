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
#include <fstream>
#include <iterator>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <NTL/ZZ.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include "../settings.h"
#include "../aritmetic/polynomial.h"
#include "../logging/logging.h"
#include "../distribution/distribution.h"

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}


 double runInit(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a;

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_init(&a);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

 double runAdd(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a,b,c;
  poly_init(&a);
  poly_init(&b);
  poly_init(&c);
  dist.generate_sample(&a, 50, d);
  dist.generate_sample(&b, 50, d);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
  	poly_add(&c,&a,&b);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

 double runMul(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a,b,c;
  poly_init(&a);
  poly_init(&b);
  poly_init(&c);
  dist.generate_sample(&a, 50, d);
  dist.generate_sample(&b, 50, d);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_mul(&c,&a,&b);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

 double runReduce(int d,int nphi, int nq){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a,b,c;
  poly_init(&a);
  dist.generate_sample(&a, 50, d);
  ZZ q = NTL::power2_ZZ(nq)-1;
  bn_t Q;
  get_words(&Q,q);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_reduce(&a,nphi,Q,nq);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

double runCRT(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a,b,c;
  poly_init(&a);
  dist.generate_sample(&a, 50, d);
  poly_copy_to_device(&a);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_crt(&a);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

  double runICRT(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a,b,c;
  poly_init(&a);
  dist.generate_sample(&a, 50, d);
  poly_elevate(&a);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_icrt(&a);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

  double runSamplingUniform(int d){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(UNIFORMLY);

  // Init
  poly_t a;
  poly_init(&a);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    dist.generate_sample(&a, 50, d);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }


double runSamplingDiscreteGaussian(int d, float gaussian_std_deviation, int gaussian_bound){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);

  // Init
  poly_t a;
  poly_init(&a);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    dist.generate_sample(&a, 50, d);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

double runBigIntegerMulZZ(int d, float gaussian_std_deviation, int gaussian_bound){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);

  // Init
  poly_t a;
  poly_init(&a);

  dist.generate_sample(&a, 50, d);
  poly_elevate(&a);

  ZZ v = to_ZZ(42);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_biginteger_mul(&a,&a,v); 
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

double runBigIntegerMulBNT(int d, float gaussian_std_deviation, int gaussian_bound){
  struct timespec start, stop;
  Distribution dist;
  dist = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);

  // Init
  poly_t a;
  poly_init(&a);

  dist.generate_sample(&a, 50, d);
  poly_elevate(&a);

  ZZ v = to_ZZ(42);
  bn_t V;
  get_words(&V,v);

  // Exec
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    poly_biginteger_mul(&a,&a,V); 
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
 }

int main(int argc, char* argv[]){
     // Log
    log_init("benchmark.log");
    double diff;

    // Output precision
    cout << fixed;
    cout.precision(2);

    // Init
    ZZ q = NTL::power2_ZZ(127) - 1;
    poly_t phi;
    ZZ_pX NTL_Phi;

    for(int d = 512; d < 8192; d*=2){
    	gen_crt_primes(q,d);
	    CUDAFunctions::init(d);
    	poly_init(&phi);
    	poly_set_nth_cyclotomic(&phi,2*d);
	    std::cout << "Degree: " << poly_get_deg(&phi) << std::endl;
    	ZZ_p::init(q);
    	
	    // Init NTL
	    for(int i = 0; i <= poly_get_deg(&phi);i++)
	      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(poly_get_coeff(&phi,i)));
	    ZZ_pE::init(NTL_Phi);

      diff = runInit(d);
      std::cout << d << " - Initialization) " << diff << " ms" << std::endl;
      diff = runAdd(d);
      std::cout << d << " - Addition) " << diff << " ms" << std::endl;
      diff = runMul(d);
      std::cout << d << " - Multiplication) " << diff << " ms" << std::endl;
      diff = runReduce(d,d,127);
      std::cout << d << " - Reduction) " << diff << " ms" << std::endl;
      diff = runCRT(d);
      std::cout << d << " - CRT) " << diff << " ms" << std::endl;
      diff = runICRT(d);
      std::cout << d << " - ICRT) " << diff << " ms" << std::endl;
      diff = runSamplingUniform(d);
      std::cout << d << " - SamplingUniform) " << diff << " ms" << std::endl;
      diff = runSamplingDiscreteGaussian(d, 8*0.4, 8*6);
      std::cout << d << " - SamplingDiscreteGaussian) " << diff << " ms" << std::endl;
      diff = runBigIntegerMulZZ(d, 8*0.4, 8*6);
      std::cout << d << " - Big-Integer multiplication ZZ) " << diff << " ms" << std::endl;
      diff = runBigIntegerMulBNT(d, 8*0.4, 8*6);
      std::cout << d << " - Big-Integer multiplication BNT) " << diff << " ms" << std::endl;
    }

}