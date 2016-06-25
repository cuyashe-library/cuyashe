#include "polynomial.h"

int OP_DEGREE = 4096;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                                                                                                                             //
// 29 bits                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        //                                                                                                                                                                                                                                                                                                             //
#if CRTPRIMESIZE == 29
  const uint32_t PRIMES_BUCKET[] = {536870909, 536870879, 536870869, 536870849, 536870839, 536870837, 536870819, 536870813, 536870791, 536870779, 536870767, 536870743, 536870729, 536870723, 536870717, 536870701, 536870683, 536870657, 536870641, 536870627, 536870611, 536870603, 536870599, 536870573, 536870569, 536870563, 536870561, 536870513, 536870501, 536870497, 536870473, 536870401, 536870363, 536870317, 536870303, 536870297, 536870273, 536870267, 536870239, 536870233, 536870219, 536870171, 536870167, 536870153, 536870123, 536870063, 536870057, 536870041, 536870027, 536869999, 536869951, 536869943, 536869937, 536869919, 536869901, 536869891, 536869831, 536869829, 536869793, 536869787, 536869777, 536869771, 536869769, 536869747, 536869693, 536869679, 536869651, 536869637, 536869633, 536869631, 536869607, 536869603, 536869589, 536869583, 536869573, 536869559, 536869549, 536869523, 536869483, 536869471, 536869447, 536869423, 536869409, 536869387, 536869331, 536869283, 536869247, 536869217, 536869189, 536869159, 536869153, 536869117, 536869097, 536869043, 536868979, 536868977, 536868973, 536868961, 536868953, 536868901}; //                                                                                                                                                                                                                                                                                                             //
#else
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                                                                                                                             //
  // 15 bits
  #if CRTPRIMESIZE == 15
   const uint32_t PRIMES_BUCKET[] = {32749, 32719, 32717, 32713, 32707, 32693, 32687, 32653, 32647, 32633, 32621, 32611, 32609, 32603, 32587, 32579, 32573, 32569, 32563, 32561, 32537, 32533, 32531, 32507, 32503, 32497, 32491, 32479, 32467, 32443, 32441, 32429, 32423, 32413, 32411, 32401, 32381, 32377, 32371, 32369, 32363, 32359, 32353, 32341, 32327, 32323, 32321, 32309, 32303, 32299, 32297, 32261, 32257, 32251, 32237, 32233, 32213, 32203, 32191, 32189, 32183, 32173, 32159, 32143, 32141, 32119, 32117, 32099, 32089, 32083, 32077, 32069, 32063, 32059, 32057, 32051, 32029, 32027, 32009, 32003, 31991, 31981, 31973, 31963, 31957, 31907, 31891, 31883, 31873, 31859, 31849, 31847, 31817, 31799, 31793, 31771, 31769, 31751, 31741, 31729, 31727, 31723, 31721, 31699, 31687, 31667, 31663, 31657, 31649, 31643, 31627, 31607, 31601, 31583, 31573, 31567, 31547, 31543, 31541, 31531, 31517, 31513, 31511, 31489, 31481, 31477, 31469, 31397, 31393, 31391, 31387, 31379, 31357, 31337, 31333, 31327, 31321, 31319, 31307, 31277, 31271, 31267, 31259, 31253, 31249, 31247, 31237, 31231, 31223, 31219, 31193, 31189, 31183, 31181, 31177, 31159, 31153, 31151, 31147, 31139, 31123, 31121, 31091, 31081, 31079, 31069, 31063, 31051, 31039, 31033, 31019, 31013, 30983, 30977, 30971, 30949, 30941, 30937, 30931, 30911, 30893, 30881, 30871, 30869, 30859, 30853, 30851, 30841, 30839, 30829, 30817, 30809, 30803, 30781, 30773, 30763, 30757, 30727, 30713, 30707}; // //
  #else
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 10 bits
    #if CRTPRIMESIZE == 10
     const uint32_t PRIMES_BUCKET[] = {1021, 1019, 1013, 1009, 997, 991, 983, 977, 971, 967, 953, 947, 941, 937, 929, 919, 911, 907, 887, 883, 881, 877, 863, 859, 857, 853, 839, 829, 827, 823, 821, 811, 809, 797, 787, 773, 769, 761, 757, 751, 743, 739, 733, 727, 719, 709, 701, 691, 683, 677, 673, 661, 659, 653, 647, 643, 641, 631, 619, 617, 613, 607, 601, 599, 593, 587, 577, 571, 569, 563, 557, 547, 541, 523, 521}; ///
    #else
      #if CRTPRIMESIZE == 9
        const uint32_t PRIMES_BUCKET[] = {67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509};
      #endif
    #endif
  #endif
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<cuyasheint_t> CRTPrimes;
ZZ CRTProduct;
std::vector<ZZ> CRTMpi;
std::vector<cuyasheint_t> CRTInvMpi;



std::map<ZZ, std::pair<cuyasheint_t*,int>> reciprocals;

/** 
 * polynomial initialization
 * @param a [description]
 */
void poly_init(poly_t *a){
	assert(CUDAFunctions::N);

	// Host vector
	a->coefs.resize(CUDAFunctions::N);
	// CRT-NTT array
	cudaError_t result = cudaMalloc((void**)&a->d_coefs,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));
	assert( result == cudaSuccess);
	result = cudaMemset(a->d_coefs,0,CUDAFunctions::N*CRTPrimes.size()*sizeof(cuyasheint_t));

	// Big-number array
	// Init on host
	bn_t *h_bn_coefs;
	h_bn_coefs = (bn_t*)malloc(CUDAFunctions::N*sizeof(bn_t));
	for(int i = 0; i < CUDAFunctions::N; i++)
		bn_new(&h_bn_coefs[i]);
	// Copy to device	
	result = cudaMalloc((void**)&a->d_bn_coefs,CUDAFunctions::N*sizeof(bn_t));
	assert( result == cudaSuccess);
	result = cudaMemcpy(a->d_bn_coefs,h_bn_coefs,CUDAFunctions::N*sizeof(bn_t),cudaMemcpyHostToDevice);
	assert( result == cudaSuccess);

	#ifdef CUFFTMUL_TRANSFORM
	// If CUFFTMUL mode
	result = cudaMalloc((void**)&a->d_coefs_transf,CUDAFunctions::N*CRTPrimes.size()*sizeof(Complex));
	assert( result == cudaSuccess);
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
	else
		// TRANSTATE
		poly_demote(a);
	
	// Big integers on the GPU
	poly_icrt(a);

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

void poly_reduce(poly_t *a, int nphi, int nq){
	const unsigned int half = nphi-1;     
	
	///////////
	// To-do //
	///////////
	log_notice("reducing on HOST");

	// #pragma omp parallel for
	for(int i = 0; (i <= half) && (i + half + 1 <= poly_get_deg(a)); i++){
		poly_set_coeff(a, i, poly_get_coeff(a, i) - poly_get_coeff(a, i+half+1) );
		poly_set_coeff(a, i+half+1, to_ZZ(0));
	}

	ZZ q = NTL::power2_ZZ(nq)-1;
	for(int i = 0; i <= poly_get_deg(a); i++)
		poly_set_coeff(a, i, poly_get_coeff(a, i) % q);

	poly_elevate(a);
	poly_elevate(a);
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
	NTL::SetCoeff(ntl_phi,0,conv<ZZ_p>(1));
	NTL::SetCoeff(ntl_phi,nphi,conv<ZZ_p>(1));

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
	 * The reciprocal is computed in the first time this function() is called
	 * After that, the result is reused
	 */
	 ZZ q_ZZ = get_ZZ(&q);
      std::pair<cuyasheint_t*,int> pair = reciprocals[q_ZZ];
      cuyasheint_t *reciprocal = std::get<0>(pair);
      int su = std::get<1>(pair);

      if( reciprocal != NULL){
        /** 
         * Not computed yet
         */
        compute_reciprocal(q_ZZ);
        pair = reciprocals[q_ZZ];
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
	#ifdef CUFFTMUL_TRANSFORM
			int primes_size = CRTPRIMESIZE;
		#else
			unsigned int count = 0;
	#endif
			
	while( (M < (2*degree)*q*q) ){

		#ifdef CUFFTMUL_TRANSFORM
		n = NTL::GenPrime_long(primes_size);
		#else
		n = PRIMES_BUCKET[count];
		count++;
		#endif

		// if( std::find(P.begin(), P.end(), n) == P.end()){
			// Does not contains
			// std::cout << n << std::endl;
			P.push_back(n);
			M *=(n);
		// }
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
	std::cout << "Primes size: " << primes_size << std::endl;
	std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
	std::cout << P.size() << " primes generated." << std::endl;
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

/**
 * release all memory used by this polynomial
 * @param a [description]
 */
void poly_clear(poly_t *a){
	a->coefs.clear();

	cudaError_t result;
	// CRT-NTT array
	result = cudaFree(a->d_coefs);
	assert(result == cudaSuccess);
	result = cudaFree(a->d_bn_coefs);
	assert(result == cudaSuccess);
	#ifdef CUFFTMUL_TRANSFORM
	// If CUFFTMUL mode
	result = cudaFree(a->d_coefs_transf);
	assert(result == cudaSuccess);
	#endif
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