#ifndef SETTINGS_H
#define SETTINGS_H

#include <cstdint>
#include <cuda_runtime.h>

///////////////////////
// cuYASHE's integer //
///////////////////////
typedef uint64_t cuyasheint_t;
typedef double2 Complex;

//////////////////////
// Big number's stuff //
//////////////////////

enum BN_SIGN {BN_POS,BN_NEG};
enum BN_CMP {CMP_LT,CMP_GT,CMP_EQ};
typedef struct bn_st{
	int alloc = 0;
	int used = 0;
	int sign = BN_POS;
	cuyasheint_t *dp = NULL;
} bn_t;

// default block x size
#define ADDBLOCKXDIM 32

// This define the default transform for polynomial multiplication
// #define NTTMUL_TRANSFORM
#define CUFFTMUL_TRANSFORM


#ifdef CUFFTMUL_TRANSFORM
#define CRTPRIMESIZE 10 
#define COPRIMES_BUCKET_SIZE 200                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     //
extern const uint32_t COPRIMES_BUCKET[];
#else
#define CRTPRIMESIZE 10 
#define COPRIMES_BUCKET_SIZE 200                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           //
extern const uint32_t COPRIMES_BUCKET[];
#endif

// CRT cannot use primes bigger than WORD/2 bits
#define WORD 64
#define BN_DIGIT WORD

// Standard number of words to allocate
// #define STD_BNT_WORDS_ALLOC 32 // Up to 1024 bits big integers
#define STD_BNT_WORDS_ALLOC 10 // Up to  bits big integers
#define DSTD_BNT_WORDS_ALLOC 20 // Up to  bits big integers

enum add_mode_t {ADD,SUB,MUL,DIV,MOD};
enum transforms {NTTMUL, CUFFTMUL};
enum ntt_mode_t {INVERSE,FORWARD};

#include <time.h>

extern double compute_time_ms(struct timespec start,struct timespec stop);

extern uint64_t get_cycles();

#endif