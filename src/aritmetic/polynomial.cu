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

#include "polynomial.h"

int OP_DEGREE = 4096;

std::vector<cuyasheint_t> CRTPrimes;
ZZ CRTProduct;
std::vector<ZZ> CRTMpi;
std::vector<cuyasheint_t> CRTInvMpi;
extern __host__ void callMersenneMod(bn_t *g, bn_t q,int nq, int N, cudaStream_t stream);



std::map<ZZ, std::pair<cuyasheint_t*,int>> reciprocals;

/** 
 * polynomial initialization
 * @param a [description]
 */
void poly_init(poly_t *a){
	assert(CUDAFunctions::N);

	// Memory allocation on the GPU is synchronous. So, first we allocate all memory that 
	// we need and then call asynchronous functions.
	cudaError_t result = cudaMalloc((void**)&a->d_coefs,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));
	assert( result == cudaSuccess);
	
	// Big-number array
	bn_t *h_bn_coefs;
	h_bn_coefs = (bn_t*)malloc(CUDAFunctions::N*sizeof(bn_t));
	cuyasheint_t *d_bn_coefs_dp;
	result = cudaMalloc((void**)&d_bn_coefs_dp,CUDAFunctions::N*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
	assert( result == cudaSuccess);
	

	// CRT-NTT array
	result = cudaMalloc((void**)&a->d_bn_coefs,CUDAFunctions::N*sizeof(bn_t));
	assert( result == cudaSuccess);
	
	#ifdef CUFFTMUL_TRANSFORM
	// If CUFFTMUL mode
	result = cudaMalloc((void**)&a->d_coefs_transf,CUDAFunctions::N*CRTPrimes.size()*sizeof(Complex));
	assert( result == cudaSuccess);
	#endif

	// Host vector
	a->coefs.resize(CUDAFunctions::N);

	// Init on host
	for(int i = 0; i < CUDAFunctions::N; i++){
		h_bn_coefs[i].alloc = STD_BNT_WORDS_ALLOC;
		h_bn_coefs[i].used = 0;
		h_bn_coefs[i].sign = BN_POS;
		h_bn_coefs[i].dp = d_bn_coefs_dp + i*STD_BNT_WORDS_ALLOC;
	}

	// Copy to device	
	result = cudaMemsetAsync(a->d_coefs,0,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));
	assert( result == cudaSuccess);
	result = cudaMemcpyAsync(a->d_bn_coefs,h_bn_coefs,CUDAFunctions::N*sizeof(bn_t),cudaMemcpyHostToDevice);
	assert( result == cudaSuccess);


}

/**
 * [poly_free description]
 * @param a [description]
 */
void poly_free(poly_t *a){
	// Coefficients and CRT residues
	// CRT residues
	cudaError_t result;
	a->coefs.clear();
	result = cudaFree(a->d_coefs);
	assert(result == cudaSuccess);

	// BN_T
	bn_t d_first;
	result = cudaMemcpy(&d_first,a->d_bn_coefs,sizeof(bn_t),cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaFree(d_first.dp);
	assert(result == cudaSuccess);
	result = cudaFree(a->d_bn_coefs);
	assert(result == cudaSuccess);

	// FFT residues
	#ifdef CUFFTMUL_TRANSFORM
	result = cudaFree(a->d_coefs_transf);
	assert(result == cudaSuccess);
	#endif

}


/**
 * [poly_clear description]
 * @param a [description]
 */
void poly_clear(poly_t *a){
	// Coefficients and CRT residues
	cudaError_t result;
	a->coefs.clear();
	a->coefs.resize(CUDAFunctions::N);
	result = cudaMemsetAsync(a->d_coefs,0,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));
	assert(result == cudaSuccess);

	// BN_T
	bn_t d_first;
	result = cudaMemcpy(&d_first,a->d_bn_coefs,sizeof(bn_t),cudaMemcpyDeviceToHost);
	result = cudaMemsetAsync(d_first.dp,0,CUDAFunctions::N*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
	assert(result == cudaSuccess);

	// FFT residues
	#ifdef CUFFTMUL_TRANSFORM
	result = cudaMemsetAsync(a->d_coefs_transf,0,CUDAFunctions::N*CRTPrimes.size()*sizeof(Complex));
	assert(result == cudaSuccess);
	#endif
	
}

/**
 * [poly_get_deg description]
 * @param  a [description]
 * @return   [description]
 */
int poly_get_deg(poly_t *a){
	while(a->status != HOSTSTATE)
		poly_demote(a);
	for( int i = a->coefs.size()-1 ; i >= 0 ; i--)
		if(a->coefs[i] != to_ZZ(0))
			return i;
	return -1;
}

/**
 * polynomial addition
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */
void poly_add(poly_t *c, poly_t *a, poly_t *b){
	while(a->status != TRANSSTATE)
		poly_elevate(a);
	while(b->status != TRANSSTATE)
		poly_elevate(b);

	#ifdef NTTMUL_TRANSFORM
	CUDAFunctions::callPolynomialAddSub(	c->d_coefs,
											a->d_coefs,
											b->d_coefs,
											CUDAFunctions::N*CRTPrimes.size(),
											NULL);

	#else
	CUDAFunctions::callPolynomialcuFFTAddSub(	c->d_coefs_transf,
												a->d_coefs_transf,
												b->d_coefs_transf,
												CUDAFunctions::N*CRTPrimes.size(),
												ADD,
												NULL);
	#endif

	c->status = TRANSSTATE;
}
/**
 * polynomial multiplication
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */

void poly_mul(poly_t *c, poly_t *a, poly_t *b){
	while(a->status != TRANSSTATE)
		poly_elevate(a);
	while(b->status != TRANSSTATE)
		poly_elevate(b);

	#ifdef NTTMUL_TRANSFORM
	CUDAFunctions::callPolynomialMul(  	c->d_coefs,
										a->d_coefs,
										b->d_coefs,
										CUDAFunctions::N*CRTPrimes.size(),
										NULL);
	#else
	CUDAFunctions::executeCuFFTPolynomialMul( 	c->d_coefs_transf, 
	                                            a->d_coefs_transf, 
	                                            b->d_coefs_transf, 
	                                            CUDAFunctions::N*CRTPrimes.size(),
	                                            NULL);
	#endif

	c->status = TRANSSTATE;
}

/**
 * polynomial addition with an integer
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */
void poly_integer_add(poly_t *c, poly_t *a, cuyasheint_t b){
	while(a->status != TRANSSTATE)
		poly_elevate(a);

  	#ifdef NTTMUL_TRANSFORM
	CUDAFunctions::callPolynomialOPInteger (
	                                          ADD,
	                                          NULL,
	                                          c->d_coefs,
	                                          a->d_coefs,
	                                          b,
	                                          CUDAFunctions::N,
	                                          CRTPrimes.size()
                                          );
	#else
	CUDAFunctions::callPolynomialcuFFTOPInteger (
	                                          ADD,
	                                          NULL,
	                                          c->d_coefs_transf,
	                                          a->d_coefs_transf,
	                                          b,
	                                          CUDAFunctions::N,
	                                          CRTPrimes.size()
                                          );
	#endif

	c->status = TRANSSTATE;
}

/**
 * polynomial multiplication with an integer
 * @param c [output]
 * @param a [input]
 * @param b [input]
 */

void poly_integer_mul(poly_t *c, poly_t *a, cuyasheint_t b){
	while(a->status != TRANSSTATE)
		poly_elevate(a);

  	#ifdef NTTMUL_TRANSFORM
	CUDAFunctions::callPolynomialOPInteger (
	                                          MUL,
	                                          NULL,
	                                          c->d_coefs,
	                                          a->d_coefs,
	                                          b,
	                                          CUDAFunctions::N,
	                                          CRTPrimes.size()
                                          );
	#else
	CUDAFunctions::callPolynomialcuFFTOPInteger (
	                                          MUL,
	                                          NULL,
	                                          c->d_coefs_transf,
	                                          a->d_coefs_transf,
	                                          b,
	                                          CUDAFunctions::N,
	                                          CRTPrimes.size()
                                          );
	#endif

	c->status = TRANSSTATE;
}

/**
 * [poly_biginteger_mul description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void poly_biginteger_mul(poly_t *c, poly_t *a, bn_t b){
	if(a->status != HOSTSTATE)
		poly_elevate(a);
	else{
		// TRANSTATE
		poly_demote(a);
		poly_icrt(a);
	}

	/////////////////////////////
	// Big integers on the GPU //
	/////////////////////////////

    CUDAFunctions::callPolynomialOPDigit( MUL,
                                          NULL,
                                          c->d_bn_coefs,
                                          a->d_bn_coefs,
                                          b,
                                          CUDAFunctions::N);

    // Back to CRTSTATE
    callCRT(  c->d_bn_coefs,
	          CUDAFunctions::N,
	          c->d_coefs,
	          CUDAFunctions::N,
	          CRTPrimes.size(),
	          0x0
    );

	c->status = CRTSTATE;
}

/**
 * [poly_biginteger_mul description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void poly_biginteger_mul(poly_t *c, poly_t *a, ZZ b){
  bn_t B;
  get_words(&B,b);
  poly_biginteger_mul(c,a,B);
}

/**
 * Reduces a polynomial a by the 2*nphi-th cyclotomic polynomial on Rq
 * 
 * @param a    [description]
 * @param nphi [description]
 * @param q    [description]
 * @param nq   [description]
 */
void poly_reduce(poly_t *a, int nphi, bn_t q, int nq){
	const unsigned int half = nphi-1;     
	
	///////////
	// To-do //
	///////////
	// log_notice("reducing on HOST");

	// // #pragma omp parallel for
	// for(int i = 0; (i <= half) && (i + half + 1 <= poly_get_deg(a)); i++){
	// 	poly_set_coeff(a, i, poly_get_coeff(a, i) - poly_get_coeff(a, i+half+1) );
	// 	poly_set_coeff(a, i+half+1, to_ZZ(0));
	// }

	// ZZ q = NTL::power2_ZZ(nq)-1;
	// for(int i = 0; i <= poly_get_deg(a); i++)
	// 	poly_set_coeff(a, i, poly_get_coeff(a, i) % q);

	// poly_elevate(a);
	// poly_elevate(a);
	
	// log_notice("reducing on GPU/COEFS")
	if(a->status == TRANSSTATE){
		poly_demote(a);
		callICRT(a->d_bn_coefs,
		      a->d_coefs,
		      CUDAFunctions::N,
		      CRTPrimes.size(),
		      NULL
	    );
	}else if(a->status == HOSTSTATE)
		poly_elevate(a);

	CUDAFunctions::callPolynomialReductionCoefs(a->d_bn_coefs, half, CUDAFunctions::N);
	callMersenneMod(a->d_bn_coefs , q, nq, CUDAFunctions::N, NULL);
    
    callCRT(a->d_bn_coefs,
          CUDAFunctions::N,
          a->d_coefs,
          CUDAFunctions::N,
          CRTPrimes.size(),
          0x0
    );


	// poly_mersenne_reduction(a,q,nq);
  	a->status = CRTSTATE;
}

/**
 * [poly_cyclotomic_reduction description]
 * @param f    [description]
 * @param nphi [description]
 */
void poly_cyclotomic_reduction(poly_t *a, int nphi){
	const unsigned int half = nphi-1;     
	if(a->status == TRANSSTATE){
		poly_demote(a);
		callICRT(a->d_bn_coefs,
		      a->d_coefs,
		      CUDAFunctions::N,
		      CRTPrimes.size(),
		      NULL
	    );
	}else if(a->status == HOSTSTATE)
		poly_elevate(a);

	CUDAFunctions::callPolynomialReductionCoefs(a->d_bn_coefs, half, CUDAFunctions::N);

   //  callCRT(a->d_bn_coefs,
   //        CUDAFunctions::N,
   //        a->d_coefs,
   //        CUDAFunctions::N,
   //        CRTPrimes.size(),
   //        0x0
   //  );

  	// a->status = CRTSTATE;
}
	
/**
 * a % q
 * @param a  [description]
 * @param q  [description]
 * @param nq [description]
 */
void poly_mersenne_reduction(poly_t *a, bn_t q, int nq){
	if(a->status == TRANSSTATE){
		poly_demote(a);
		callICRT(a->d_bn_coefs,
		      a->d_coefs,
		      CUDAFunctions::N,
		      CRTPrimes.size(),
		      NULL
	    );
	}else if(a->status == HOSTSTATE)
		poly_elevate(a);

	callMersenneMod(a->d_bn_coefs , q, nq, CUDAFunctions::N, NULL);

    callCRT(a->d_bn_coefs,
          CUDAFunctions::N,
          a->d_coefs,
          CUDAFunctions::N,
          CRTPrimes.size(),
          0x0
    );

  	a->status = CRTSTATE;
}


/**
 * computes the polynomial inverse in R_q
 * @param fInv [description]
 * @param f    [description]
 * @param nphi x^{nphi} - 1
 * @param nq   2^{nq} - 1
 */
void poly_invmod(poly_t *fInv, poly_t *f, int nphi, int nq){
	///////////////////////////////
	// This is a very ugly hack. //
	///////////////////////////////

	//
	ZZ_pEX ntl_f;
	for(int i = 0; i <= poly_get_deg(f); i++)
		NTL::SetCoeff(ntl_f,i,conv<ZZ_p>(poly_get_coeff(f,i)));
	ZZ_pEX ntl_phi;
	NTL::SetCoeff(ntl_phi,0,1);
	NTL::SetCoeff(ntl_phi,nphi,1);

	ZZ_pEX inv_f_ntl =  NTL::InvMod(ntl_f, ntl_phi);

	poly_init(fInv);
	for(int i = 0; i <= nphi;i++){
		ZZ ntl_value;
		if( NTL::IsZero(NTL::coeff(inv_f_ntl,i)) )
			// Without this, NTL raises an exception when we call rep()
			ntl_value = 0L;
		else
			ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(inv_f_ntl,i))[0]);

			poly_set_coeff(fInv,i,ntl_value);
	}

	fInv->status = HOSTSTATE;

}

/**
 * Step to the next polynomial status
 * @param a [description]
 */
void poly_elevate(poly_t *a){

	if(a->status ==HOSTSTATE){
		// Copy to the GPU and compute CRT
		log_notice("Elevating from HOST to CRT.");
		poly_copy_to_device(a);
		poly_crt(a);
		a->status = CRTSTATE;
	}else if(a->status ==CRTSTATE){
		// Apply the transform
		log_notice("Elevating from CRT to TRANS.");
	  	#ifdef NTTMUL_TRANSFORM
		// NTT mul
		CUDAFunctions::applyNTT( a->d_coefs, CUDAFunctions::N, CRTPrimes.size(), FORWARD, NULL);
		#else
		// FFT mul
		int size = CUDAFunctions::N*CRTPrimes.size();

		CUDAFunctions::executeCopyIntegerToComplex(a->d_coefs_transf,a->d_coefs,size,NULL);
		assert(cudaGetLastError() == cudaSuccess);

		cufftExecZ2Z( CUDAFunctions::plan,
		            (cufftDoubleComplex *)(a->d_coefs_transf),
		            (cufftDoubleComplex *)(a->d_coefs_transf),
		            CUFFT_FORWARD
		          );
		#endif
		a->status = TRANSSTATE;
	}else if(a->status ==TRANSSTATE){
		// Do nothing
		log_notice("There is no need to elevate the polynomial.");
	}else{
		log_error("Inconsistent state!");
	}
}

// Step back
// Step to the next polynomial status
void poly_demote(poly_t *a){

	if(a->status == HOSTSTATE){
		// Do nothing
		log_notice("There is no need to demote the polynomial.");
	}else if(a->status == CRTSTATE){
		// Copy back to the HOST
		log_notice("Demoting from CRT to HOST");
		poly_icrt(a);
		poly_copy_to_host(a);
		a->status = HOSTSTATE;
	}else if(a->status == TRANSSTATE){
		log_notice("Demoting from TRANS to CRT");
	  	#ifdef NTTMUL_TRANSFORM
		// NTT mul
		CUDAFunctions::applyNTT( a->d_coefs, CUDAFunctions::N, CRTPrimes.size(), INVERSE, NULL);
		#else
		// FFT mul
		int size = CUDAFunctions::N*CRTPrimes.size();

		cufftExecZ2Z( CUDAFunctions::plan,
		            (cufftDoubleComplex *)(a->d_coefs_transf),
		            (cufftDoubleComplex *)(a->d_coefs_transf),
		            CUFFT_INVERSE
		          );

		CUDAFunctions::executeCopyAndNormalizeComplexRealPartToInteger(a->d_coefs,(cufftDoubleComplex *)a->d_coefs_transf,size,CUDAFunctions::N,NULL);
		assert(cudaGetLastError() == cudaSuccess);
		#endif
		a->status = CRTSTATE;
	}else{
		log_error("Inconsistent state!");
	}
}

void poly_crt(poly_t *a){
  if(a->status == CRTSTATE || a->status == TRANSSTATE)
  	return;
  // So, a->status == HOSTSTATE

  /**
   * To run CRT we need a copy of the coeficients on device's memory
   */
   poly_copy_to_device(a);

  callCRT(a->d_bn_coefs,
          CUDAFunctions::N,
          a->d_coefs,
          CUDAFunctions::N,
          CRTPrimes.size(),
          0x0
    );

  a->status = CRTSTATE;
}

void poly_icrt(poly_t *a){
  if(a->status == HOSTSTATE){
  	poly_copy_to_device(a);
  	return;
  }else if (a->status == TRANSSTATE)
  	poly_demote(a);
  

  // So, a->status == CRTSTATE
  
  callICRT(a->d_bn_coefs,
          a->d_coefs,
          CUDAFunctions::N,
          CRTPrimes.size(),
          NULL
    );

}

std::string poly_print(poly_t *a){
	while(a->status != HOSTSTATE)
		poly_demote(a);

	std::ostringstream oss;
	for(int i = 0; i < a->coefs.size(); i++)
		oss << a->coefs[i] << ", ";
	return oss.str();
	// for(int i = 0; i < a->coefs.size(); i++)
	// 	std::cout << a->coefs[i] << ", ";
	// std::cout << std::endl;
}

/**
 * get_words converts a NTL big integer
 * in our bn_t format
 * @param b output: word representation
 * @param a input: operand
 */
void get_words(bn_t *b,ZZ a){
  cudaError_t result;
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  assert(alloc > 0);
  
  h_dp = (cuyasheint_t *) calloc (alloc,sizeof(cuyasheint_t));

  /**
   * Compute the decomposition of a
   */
  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    if(used >= alloc){
      h_dp = (cuyasheint_t*)realloc(h_dp,alloc+STD_BNT_WORDS_ALLOC);
      alloc += STD_BNT_WORDS_ALLOC;
      log_warn("get_words realloc! This is bad for performance.");
    }

    h_dp[used] = conv<uint64_t>(x);
  
  }

  /** 
   * Asserts that there is enough space allocated on the GPU
   */
  if(b->alloc < alloc){
	if(b->alloc > 0 && b->dp != 0x0){
		// If b->dp was allocated with less data than we need
		result = cudaFree(b->dp);
		assert(result == cudaSuccess); 
	}

	result = cudaMalloc((void**)&b->dp,alloc*sizeof(cuyasheint_t));
	assert(result == cudaSuccess);  

  }

  /*
   * Copy new data to device memory
   * This call shouldn't be async because it can mess with that free in the end
   */
  result = cudaMemcpy(b->dp,h_dp,alloc*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);

  free(h_dp);
}

void get_words_allocatted(bn_t *b,ZZ a,cuyasheint_t *h_data, cuyasheint_t *d_data, int index,cudaStream_t stream){
  /**
   * Uses previously allocated memory
   */
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  h_dp = h_data + index*alloc;

  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    assert(used < alloc);
    h_dp[used] = conv<uint64_t>(x);
  }

  /*
   * Copy new data to device memory
   */

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);
  b->dp = d_data + index*alloc;
}

void get_words_host(bn_t *b,ZZ a){
  /**
   * Compute
   */
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  h_dp = (cuyasheint_t *) calloc (alloc,sizeof(cuyasheint_t));

  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    if(used >= alloc){
      h_dp = (cuyasheint_t*)realloc(h_dp,alloc+STD_BNT_WORDS_ALLOC);
      alloc += STD_BNT_WORDS_ALLOC;
      std::cout << "get_words realloc!" << std::endl;
    }
    h_dp[used] = conv<uint64_t>(x);
  }

  // if(b->alloc != alloc && alloc > 0){
    // cudaError_t result;
    if(b->alloc != alloc || b->dp == 0x0)
      // If b->dp was allocated with less data than we need
      if(b->alloc != 0)
        free(b->dp);

  /*
   * Copy new data to device memory
   */

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);
  b->dp = h_dp;
}

/**
 * receives a bn_t (on CPU memory) and returns the related ZZ
 * @param  a [description]
 * @return   [description]
 */
ZZ get_ZZ(bn_t *a){
  ZZ b = conv<ZZ>(0);
  for(int i = a->used-1; i >= 0;i--)
      b = (b<<WORD) | to_ZZ(a->dp[i]);
  return b;
}

void poly_copy_to_device(poly_t *a){
	if(a->status != HOSTSTATE)
		return;

	cudaError_t result;

	// Alloc memory
	if(!a->d_coefs){
		result = cudaMalloc((void**)&a->d_coefs,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));
		assert(result == cudaSuccess);
	}
	if(!a->d_bn_coefs){
		result = cudaMalloc((void**)&a->d_bn_coefs,CUDAFunctions::N*sizeof(bn_t));
		assert(result == cudaSuccess);
	}

	bn_t *h_bn_coefs = (bn_t*)malloc(CUDAFunctions::N*sizeof(bn_t));
	assert(h_bn_coefs);

	// Allocate all memory all at once
	cuyasheint_t *d_dp;
	result = cudaMalloc((void**)&d_dp,CUDAFunctions::N*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
	assert(result == cudaSuccess);
	cuyasheint_t *h_dp = (cuyasheint_t*)calloc(CUDAFunctions::N*STD_BNT_WORDS_ALLOC,sizeof(cuyasheint_t));
	assert(h_dp);

	// Copy coefs to the device
	for(int i = 0; i < CUDAFunctions::N; i++)
		get_words_allocatted(&h_bn_coefs[i],a->coefs[i],h_dp,d_dp,i,NULL);

	// Copy the references
	result = cudaMemcpy(d_dp,h_dp,CUDAFunctions::N*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
	result = cudaMemcpy(a->d_bn_coefs,h_bn_coefs,CUDAFunctions::N*sizeof(bn_t),cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

	free(h_bn_coefs);
}

void poly_copy_to_host(poly_t *a){
	if(a->status == HOSTSTATE)
		return;

	cudaError_t result;

	// Alloc memory
	bn_t *h_bn_coefs = (bn_t*)malloc(CUDAFunctions::N*sizeof(bn_t));
	assert(h_bn_coefs);


	// Copy to host
	result = cudaMemcpy(h_bn_coefs,a->d_bn_coefs,CUDAFunctions::N*sizeof(bn_t),cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	// 
	a->coefs.clear();
	a->coefs.resize(CUDAFunctions::N);
	std::vector<ZZ>::iterator it;
	it = a->coefs.begin();
	for(int i = 0; i < CUDAFunctions::N; i++){
	  // Recover the coef
      bn_t bn_coef;
      bn_coef.used = h_bn_coefs[i].used;
      if(bn_coef.used == 0)
      	continue;
      bn_coef.alloc = h_bn_coefs[i].alloc;
      bn_coef.sign = h_bn_coefs[i].sign;
      bn_coef.dp = (cuyasheint_t*)malloc(bn_coef.used*sizeof(cuyasheint_t));

      result = cudaMemcpy(bn_coef.dp,h_bn_coefs[i].dp,bn_coef.used*sizeof(cuyasheint_t),cudaMemcpyDeviceToHost);
      
      // Build the ZZ
      ZZ coef = get_ZZ(&bn_coef);
      
      // Inserts
      a->coefs.insert(a->coefs.begin()+i,coef);		
	}

	a->status = HOSTSTATE;
}


bn_t get_reciprocal(ZZ q){
      std::pair<cuyasheint_t*,int> pair = reciprocals[q];
      cuyasheint_t *reciprocal = std::get<0>(pair);
      int su = std::get<1>(pair);

      if( reciprocal == NULL){
        /** 
         * Not computed yet
         */
        compute_reciprocal(q);
        pair = reciprocals[q];
        reciprocal = std::get<0>(pair);
        su = std::get<1>(pair);
      }

      bn_t result;
      result.used = su;
      result.alloc = su;
      result.sign = BN_POS;
      result.dp = reciprocal;

      return result;
    }
bn_t get_reciprocal(bn_t q){
	/**
	 * The reciprocal is computed on the first time this function() is called.
	 * After that, the result is reused.
 	*/
	ZZ q_ZZ = get_ZZ(&q);
	return get_reciprocal(q_ZZ);
    }
void compute_reciprocal(ZZ q){
      ZZ u_ZZ;

      int nwords = NTL::NumBits(q)/WORD + (NTL::NumBits(q)%WORD != 0);

      ZZ x = power2_ZZ(2*WORD*nwords);
      u_ZZ = x/q;
      // std::cout << "The reciprocal of " << q << " is " << u_ZZ << std::endl;

      bn_t *h_u;
      h_u = (bn_t*) malloc (sizeof(bn_t));
      h_u->alloc = 0;

      get_words(h_u,u_ZZ);  

      //////////
      // Copy //
      //////////
      cudaError_t result;
      
      // Copy words
      cuyasheint_t *d_u;        
      
      // Copy to device
      result = cudaMalloc((void**)&d_u,h_u->alloc*sizeof(cuyasheint_t));
      assert(result == cudaSuccess);
      
      result = cudaMemcpy(d_u,h_u->dp,h_u->alloc*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
      assert(result == cudaSuccess);

      reciprocals[q] = std::pair<cuyasheint_t*,int>(d_u,h_u->used);
      
      free(h_u);
    }

void gen_crt_primes(ZZ q,cuyasheint_t degree){
	ZZ M = ZZ(1);
	std::vector<cuyasheint_t> P;
	std::vector<ZZ> Mpi;
	std::vector<cuyasheint_t> InvMpi;

	cuyasheint_t n;

	// Get primes
	// std::cout << "Primes: " << std::endl;
	int count = 0;
	while( (M < (2*degree)*q*q) ){
		n = COPRIMES_BUCKET[count];
		count++;
		P.push_back(n);
		M *=(n);
	}
	// std::cout << std::endl;
	// Compute M/pi and it's inverse
	for(unsigned int i = 0; i < P.size();i++){
		ZZ pi = to_ZZ(P[i]);
		Mpi.push_back(M/pi);
		InvMpi.push_back(conv<cuyasheint_t>(NTL::InvMod(Mpi[i]%pi,pi)));
	}

	#ifndef CUFFTMUL_TRANSFORM
	compute_reciprocal(M);
	#endif

	CRTProduct = M;
	CRTPrimes = P;
	CRTMpi = Mpi;
	CRTInvMpi = InvMpi;

	#ifdef VERBOSE
	log_notice("Primes size: " << CRTPRIMESIZE);
	log_notice("Primes set - M:" << Polynomial::CRTProduct);
	log_notice(P.size() << " primes generated.");
	#endif

	// Send primes to GPU
	CUDAFunctions::write_crt_primes();
}


/**
 * [poly_set_coeff description]
 * @param a [description]
 * @param index [description]
 * @param c [description]
 */
void poly_set_coeff(poly_t *a, int index, ZZ c){
	while(a->status != HOSTSTATE)
		poly_demote(a);
	if(a->coefs.size() <= index)
		a->coefs.resize(index+1);
	a->coefs[index] = c;
}

/**
 * [poly_get_coeff description]
 * @param a     [description]
 * @param index [description]
 */
ZZ poly_get_coeff(poly_t *a, int index){
	while(a->status != HOSTSTATE)
		poly_demote(a);
	return a->coefs[index];
}

bool is_power_of_two(int n){
	return (n & (n - 1)) == 0;
}

/**
 * [poly_set_nth_cyclotomic description]
 * @param a [description]
 * @param n [description]
 */
void poly_set_nth_cyclotomic(poly_t *a, int n){
	assert(is_power_of_two(n));
	poly_clear(a);
	poly_init(a);
	poly_set_coeff(a,0,to_ZZ(1));
	poly_set_coeff(a,n/2,to_ZZ(1));
}
