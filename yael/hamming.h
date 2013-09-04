/* Hamming distances. The binary vector length should be a power of 8 */
#ifndef __hamming_h
#define __hamming_h

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;


/* matching elements (those returned) */
typedef struct hammatch_s {
  int qid;        /* query id */
  int bid;        /* base id */
  uint16 score;   /* Hamming distance */
} hammatch_t;


/* Define individual Hamming distance for various sizes.
   The generic one is slow while optimization is available for specific sizes */

uint16 hamming_generic (const uint8 *bs1, const uint8 * bs2, int ncodes);


/* Define prototype if no SSE4.2 available, otherwise use the specific processor instructions */
#ifndef __SSE4_2__
uint16 hamming_32 (const uint32 * bs1, const uint32 * bs2);
uint16 hamming_64 (const uint64 * bs1, const uint64 * bs2);

#else  /* Use SSE 4.2 */
#include <nmmintrin.h>
#ifndef hamming_32
#define hamming_32(pa,pb) _mm_popcnt_u32((*((uint32 *) (pa)) ^ *((uint32 *) (pb))))
#endif
#ifndef hamming_64
#define hamming_64(pa,pb) _mm_popcnt_u64((*((uint64 *) (pa)) ^ *((uint64 *) (pb))))
#endif
#endif


#ifndef BITVECSIZE
#warning "# BITVECSIZE UNDEFINED. SET TO 128 BY DEFAULT." 
#define BITVECSIZE 128
#elif BITVECSIZE%8 != 0
#error "Only power of 8 are possible for BITVECSIZE"
#endif

#define BITVECBYTE (BITVECSIZE/8)

/* Define the Hamming distance by selecting the most appropriate function,
   using the generic version as a backup */
#if BITVECSIZE==32
#define hamming(a,b)  hamming_32((uint32*) (a), (uint32*) (b))

#elif BITVECSIZE==64
#define hamming(a,b)  hamming_64((uint64*) (a), (uint64*) (b))

#elif BITVECSIZE==128
#define hamming(a,b)  (hamming_64(a,b)+hamming_64(((uint64 *) (a)) + 1, ((uint64 *) (b)) + 1))

#else
#define hamming(a,b) hamming_generic((uint8*) (a), (uint8*) (b), BITVECBYTE);
#endif



/* Compute a set of Hamming distances between na and nb binary vectors */
void compute_hamming (uint16 * dis, const uint8 * a, const uint8 * b, int na, int nb);

/* The same but with a generic function */
void compute_hamming_generic (uint16 * dis, const uint8 * a, const uint8 * b, 
                              int na, int nb, int ncodes);

/* Threaded versions, when OpenMP is available */
#ifdef _OPENMP
void compute_hamming_thread (uint16 * dis, const uint8 * a, const uint8 * b, int na, int nb);
#endif /* _OPENMP */

/* Compute hamming distance and report those below a given threshold in a structure array */
void match_hamming_thres (const uint8 * qbs, const uint8 * dbs, int nb, int ht,
                          int bufsize, hammatch_t ** hmptr, int * nptr);

void match_hamming_thres_generic (const uint8 * qbs, const uint8 * dbs, int nb, int ht,
                                  int bufsize, hammatch_t ** hmptr, int * nptr, int ncodes);

/* Compute all cross-distances between two sets of binary vectors */
void crossmatch_he (const uint8 * dbs, int n, int ht,
                    int bufsize, hammatch_t ** hmptr, int * nptr);



#endif /* __hamming_h */
