/* This code was written by Herve Jegou. Contact: herve.jegou@inria.fr  */
/* Last change: June 1st, 2010                                          */
/* This software is governed by the CeCILL license under French law and */
/* abiding by the rules of distribution of free software.               */
/* See http://www.cecill.info/licences.en.html                          */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hamming.h"

/* the slice size is set to avoid testing the buffer size too often */
#define HAMMATCH_SLICESIZE 16

/* geometric re-allocation: add a constant size plus a relative 50% of additional memory */
#define HAMMATCH_REALLOC_NEWSIZE(oldsize) (HAMMATCH_SLICESIZE+((oldsize * 5) / 4))



static uint16 uint8_nbones[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};



/* Elementary Hamming distance computation */

uint16 hamming_generic (const uint8 *bs1, const uint8 * bs2, int ncodes)
{
  int i;
  uint16 ham = 0;

  for (i = 0; i < ncodes ; i++) {
    ham += uint8_nbones[*bs1 ^ *bs2];
    bs1++;
    bs2++;
  }

  return ham;
}


#ifndef __SSE4_2__
#warning "SSE4.2 NOT USED FOR HAMMING DISTANCE COMPUTATION. Consider adding -msse4 in makefile.inc!"

uint16 hamming_32 (const uint32 * bs1, const uint32 * bs2)
{
  uint16 ham = 0;
  uint32 diff = ((*bs1) ^ (*bs2));

  ham = uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff];
  return ham;
}


uint16 hamming_64 (const uint64 * bs1, const uint64 * bs2)
{
  uint16 ham = 0;
  uint64 diff = ((*bs1) ^ (*bs2));

  ham = uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff & 255];
  diff >>= 8;
  ham += uint8_nbones[diff];

  return ham;
}

#endif



void compute_hamming (uint16 * dis, const uint8 * a, const uint8 * b, int na, int nb)
{
  int i, j;
  const uint8 * pb = b;
  for (j = 0 ; j < nb ; j++) {
    const uint8 * pa = a;
    for (i = 0 ; i < na ; i++) {
      *dis = hamming (pa, pb);
      pa += BITVECBYTE;
      dis++;
    }
    pb += BITVECBYTE;
  }
}


void compute_hamming_generic (uint16 * dis, const uint8 * a, const uint8 * b, 
                              int na, int nb, int ncodes)
{
  if (ncodes == BITVECBYTE) {
    compute_hamming (dis, a, b, na, nb);
    return;
  }
  fprintf (stderr, "# Warning: non-optimized version of the Hamming distance\n");  
     
  int i, j;
  const uint8 * pb = b;
  for (j = 0 ; j < nb ; j++) {
    const uint8 * pa = a;
    for (i = 0 ; i < na ; i++) {
      *dis = hamming_generic (pa, pb, ncodes);
      pa += ncodes;
      dis++;
    }
    pb += ncodes;
  }
}


/* Compute hamming distance and report those below a given threshold in a structure array */
hammatch_t * hammatch_new (int n)
{
  return (hammatch_t *) malloc (n * sizeof (hammatch_t));
}


hammatch_t * hammatch_realloc (hammatch_t * m, int n)
{
  return (hammatch_t *) realloc (m, n * sizeof (hammatch_t));
}



void match_hamming_thres (const uint8 * qbs, const uint8 * dbs, int nb, int ht,
                          size_t bufsize, hammatch_t ** hmptr, size_t * nptr)
{
  size_t j, posm = 0;
  uint16 h;
  *hmptr = hammatch_new (bufsize);
  hammatch_t * hm = *hmptr;
  
  for (j = 0 ; j < nb ; j++) {
    
    /* Here perform the real work of computing the distance */
    h = hamming (qbs, dbs);
            
    /* collect the match only if this satisfies the threshold */
    if (h <= ht) {
      /* Enough space to store another match ? */
      if (posm >= bufsize) {
          bufsize = HAMMATCH_REALLOC_NEWSIZE (bufsize);
          *hmptr = hammatch_realloc (*hmptr, bufsize);
          assert (*hmptr != NULL);
          hm = (*hmptr) + posm;
      }
      
      hm->bid = j;
      hm->score = h;
      hm++;
      posm++;
    }
    dbs += BITVECBYTE;  /* next signature */
  }
  
  *nptr = posm;
}


void match_hamming_thres_generic (const uint8 * qbs, const uint8 * dbs, 
                                  int nb, int ht, size_t bufsize, 
                                  hammatch_t ** hmptr, size_t * nptr, size_t ncodes)
{
  size_t j, posm = 0;
  uint16 h;
  *hmptr = hammatch_new (bufsize);
  hammatch_t * hm = *hmptr;
  
  for (j = 0 ; j < nb ; j++) {
    
    /* Here perform the real work of computing the distance */
    h = hamming_generic (qbs, dbs, ncodes);
    
    /* collect the match only if this satisfies the threshold */
    if (h <= ht) {
      /* Enough space to store another match ? */
      if (posm >= bufsize) {
        bufsize = HAMMATCH_REALLOC_NEWSIZE (bufsize);
        *hmptr = hammatch_realloc (*hmptr, bufsize);
        assert (*hmptr != NULL);
        hm = (*hmptr) + posm;
      }
      
      hm->bid = j;
      hm->score = h;
      hm++;
      posm++;
    }
    dbs += ncodes;  /* next signature */
  }
  
  *nptr = posm;
}


void crossmatch_he (const uint8 * dbs, long n, int ht,
                    long bufsize, hammatch_t ** hmptr, size_t * nptr)
{
  size_t i, j, posm = 0;
  uint16 h;
  *hmptr = hammatch_new (bufsize);
  hammatch_t * hm = *hmptr;
  const uint8 * bs1 = dbs;
  
  for (i = 0 ; i < n ; i++) {
    const uint8 * bs2 = bs1 + BITVECBYTE;
    
    for (j = i + 1 ; j < n ; j++) {
      
      /* Here perform the real work of computing the distance */
      h = hamming (bs1, bs2);
      
      /* collect the match only if this satisfies the threshold */
      if (h <= ht) {
        /* Enough space to store another match ? */
        if (posm >= bufsize) {
          bufsize = HAMMATCH_REALLOC_NEWSIZE (bufsize);
          *hmptr = hammatch_realloc (*hmptr, bufsize);
          assert (*hmptr != NULL);
          hm = (*hmptr) + posm;
        }
        
        hm->qid = i;
        hm->bid = j;
        hm->score = h;
        hm++;
        posm++;
      }
      bs2 += BITVECBYTE;
    }
    bs1  += BITVECBYTE;  /* next signature */
  }
  
  *nptr = posm;
}



void crossmatch_he_count (const uint8 * dbs, int n, int ht, size_t * nptr)
{
  size_t i, j, posm = 0;
  const uint8 * bs1 = dbs;
  
  for (i = 0 ; i < n ; i++) {
    const uint8 * bs2 = bs1 + BITVECBYTE;
    
    for (j = i + 1 ; j < n ; j++) {
      
      /* collect the match only if this satisfies the threshold */
      if (hamming (bs1, bs2) <= ht) 
        posm++;
      
      bs2 += BITVECBYTE;
    }
    bs1  += BITVECBYTE;  /* next signature */
  }
  
  *nptr = posm;
}




/*-------------------------------------------*/
/* Threaded versions, if OpenMP is available */
#ifdef _OPENMP
void compute_hamming_thread (uint16 * dis, const uint8 * a, const uint8 * b, 
                             int na, int nb)
{
  long i, j;
#pragma omp parallel shared (dis, a, b, na, nb) private (i, j)
    {
#pragma omp for 
      for (j = 0 ; j < nb ; j++)
	      for (i = 0 ; i < na ; i++)
	        dis[j * na + i] = hamming (a + i * BITVECBYTE, b + j * BITVECBYTE);
    }
}

#endif /* _OPENMP */


