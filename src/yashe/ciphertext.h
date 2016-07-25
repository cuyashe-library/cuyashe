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
 * [cipher_init_keyswitch description]
 * @param a [description]
 */
void cipher_init_keyswitch(cipher_t *a);

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