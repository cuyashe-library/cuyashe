#include "ciphertext.h"

void cipher_keyswitch(cipher_t *cmul, cipher_t c);

///////////////////////////////////////////////////
///
void cipher_init(cipher_t *a){
	poly_init(&a->p);
	a->level = 0;
}

/**
 * [cipher_init_keyswitch description]
 * @param a [description]
 */
void cipher_init_keyswitch(cipher_t *a){
	cudaError_t result;
	int size = CUDAFunctions::N*Yashe::lwq;

	// Big-number array
	bn_t *h_bn_coefs;
	h_bn_coefs = (bn_t*)malloc(size*sizeof(bn_t));
	cuyasheint_t *d_bn_coefs_dp;
	result = cudaMalloc((void**)&d_bn_coefs_dp,size*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
	assert( result == cudaSuccess);
	

	// Init on host
	for(int i = 0; i < size; i++){
		h_bn_coefs[i].alloc = STD_BNT_WORDS_ALLOC;
		h_bn_coefs[i].used = 0;
		h_bn_coefs[i].sign = BN_POS;
		h_bn_coefs[i].dp = d_bn_coefs_dp + i*STD_BNT_WORDS_ALLOC;
	}

	// Copy to device	
	result = cudaMalloc((void**)&a->d_bn_coefs,size*sizeof(bn_t));
	assert ( result == cudaSuccess);
	result = cudaMemcpyAsync(a->d_bn_coefs,h_bn_coefs,size*sizeof(bn_t),cudaMemcpyHostToDevice);
	assert( result == cudaSuccess);

	// Allocates lwq polynomials
	// a->P = (poly_t*) malloc (Yashe::lwq*sizeof(poly_t));
	a->P.resize(Yashe::lwq);
	for(int i = 0; i < Yashe::lwq; i++){
		// After initialization, it frees d_bn_coefs and sets it again with consecutive
		// memory addresses
		// 
		poly_init(&a->P[i]);
		cudaError_t result = cudaFree(a->P[i].d_bn_coefs);
		assert(result == cudaSuccess);

		a->P[i].d_bn_coefs = a->d_bn_coefs + i*CUDAFunctions::N;
	}

}

void cipher_free(cipher_t *a){
	poly_free(&a->p);
}

void cipher_add(cipher_t *c, cipher_t *a,cipher_t *b){
	poly_add(&c->p, &a->p, &b->p);
	c->level += 1;
}

void cipher_mul(cipher_t *c,cipher_t *a,cipher_t *b){

	// g = c1*c2
	poly_mul(&c->p, &a->p, &b->p);
	// poly_mersenne(&c->p,Yashe::Q,Yashe::nq);
	log_debug("c1*c2: " + poly_print(&c->p));
	// g = t*c1*c2
	poly_mul(&c->p,&c->p,&Yashe::t);
	log_debug("t*c1*c2: "+poly_print(&c->p));
	// log_debug("t*c1*c2 in R: "+poly_print(&c->p));

	// g = approx( g/q )
	poly_icrt(&c->p);
	poly_cyclotomic_reduction(&c->p, Yashe::nphi);
	callCiphertextMulAux(	c->p.d_bn_coefs,
							Yashe::Q,
							Yashe::nq,
							CUDAFunctions::N,
							NULL );
    poly_reduce(&c->p, Yashe::nphi, Yashe::Q, Yashe::nq); 
	
	// cipher_keyswitch(c, *c);

	callCRT(c->p.d_bn_coefs,
			CUDAFunctions::N,
			c->p.d_coefs,
			CUDAFunctions::N,
			CRTPrimes.size(),
			0x0	);
	
	c->p.status = CRTSTATE;
	c->level = std::max(a->level,b->level) + 1;	

	log_debug("c_mul: "+poly_print(&c->p));
}

void cipher_keyswitch(cipher_t *cmul, cipher_t c){

	// keyswitch auxilia variable not initialized
	if(c.P.size() == 0)
		cipher_init_keyswitch(&c);

	// WordDecomp
	callCuWordecomp(	NULL,
						Yashe::w,
						c.d_bn_coefs, // Array of lwq polynomial
						c.p.d_bn_coefs, // operand
						Yashe::lwq,
						CUDAFunctions::N);
	
	// Each polynomial in c.P will be multiplied with a polynomial in evk and
	// added to cmul
	
	for ( int i = 0; i < Yashe::lwq; i++){
		poly_mul(&c.P.at(i), &c.P.at(i), &Yashe::gamma.at(i));
		poly_add(&cmul->p,&cmul->p,&c.P.at(i));
	}

}