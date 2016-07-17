#include "ciphertext.h"

void cipher_init(cipher_t *a){
	poly_init(&a->p);
	a->level = 0;
}

void cipher_free(cipher_t *a){
	poly_free(&a->p);
}

void cipher_add(cipher_t *c, cipher_t *a,cipher_t *b){
	poly_add(&c->p, &a->p, &b->p);
	c->level += 1;
}

void cipher_mul(cipher_t *c,cipher_t *a,cipher_t *b){

	// g = t*c1*c2
	poly_mul(&c->p, &a->p, &b->p);
	poly_mersenne(&c->p,Yashe::Q,Yashe::nq);
	log_debug("c1*c2: " + poly_print(&c->p));
	poly_mul(&c->p,&c->p,&Yashe::t);
	log_debug("t*c1*c2: "+poly_print(&c->p));
	poly_reduce(&c->p, Yashe::nphi, Yashe::Q, Yashe::nq);

	// g = approx( g/q )
	poly_icrt(&c->p);
	callCiphertextMulAux(	c->p.d_bn_coefs,
							Yashe::Q,
							Yashe::nq,
							CUDAFunctions::N,
							NULL );
	callCRT(c->p.d_bn_coefs,
			CUDAFunctions::N,
			c->p.d_coefs,
			CUDAFunctions::N,
			CRTPrimes.size(),
			0x0	);
	c->p.status = CRTSTATE;

	c->level += 1;	
}

void cipher_convert(cipher_t *c, cipher_t *a,cipher_t *b){
	//to-do
}