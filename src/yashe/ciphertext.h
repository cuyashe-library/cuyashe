#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "../settings.h"
#include "../aritmetic/polynomial.h"
#include "../yashe/yashe.h"
#include "../cuda/operators.h"
#include "../cuda/cuda_ciphertext.h"
#include "../logging/logging.h"

/**
 * [cipher_init description]
 * @param a [description]
 */
void cipher_init(cipher_t *a);

/**
 * [cipher_free description]
 * @param a [description]
 */
void cipher_free(cipher_t *a);

/**
 * [cipher_add description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void cipher_add(cipher_t *c, cipher_t *a,cipher_t *b);

/**
 * [cipher_mul description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void cipher_mul(cipher_t *c,cipher_t *a,cipher_t *b);

/**
 * [cipher_convert description]
 * @param c [description]
 * @param a [description]
 * @param b [description]
 */
void cipher_convert(cipher_t *c, cipher_t *a,cipher_t *b);

#endif