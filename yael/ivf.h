/* Generic inverted file    */

#ifndef __ivfc_h
#define __ivfc_h

#include <stdio.h>

#define KEYNULL (2147483647)
#define IVFCHECKSUM (0xf0f0f0fa)

#define DEFAULT_SEG_SIZE 128

/* A structure representing the inverted file (loaded in memory)) */
typedef struct ivf_s {
  int checksum;
  int k;                 /* number of distinct key values */
  int n;                 /* number of vectors stored */
  int elem_size;         /* size of the additional info, per element */
  int *nbelems;          /* the number of elements stored for a given vw */
  int *seg_size;         /* the amount (number of elements) of data allocated for a given vw */
  int **ids;             /* the ids of the vectors (for each key, all the ids) */
  unsigned char ** adat; /* additional info stored with vectors */
} ivf_t;


/* matching elements (those returned) */
typedef struct ivfmatch_s {
  int qid;               /* query id */
  int bid;               /* base id */
  float score;           /* matching score */
} ivfmatch_t;



/*-------------------- Ivf interface --------------------*/

/* Check if the inverted file looks Ok */
int ivf_checksum (const ivf_t * ivf);

/* create a void inverted file in memory. 
   l: number of distinct inverted lists
   elemsize: size of the elementary element (0 for regular BOF)
   segsize: default if =0, otherwise specified number of elements
*/
ivf_t *ivf_new (int k, int elemsize, int segsize);


/* suppress the inverted file */
void ivf_delete (ivf_t * ivf);

/* Add n entries in the structure */
void ivf_addn (ivf_t * ivf, const int * id, const int * keys, 
	       const unsigned char * val, int n);

void ivf_add (ivf_t * ivf, int id, int key, const unsigned char * val);

/* returns the size of the list associated with a given key entry */
int ivf_get_nb_elems (const ivf_t * ivf, int key);

/* returns the set of vector identifiers associated with a given key value */
int * ivf_get_ids (const ivf_t * ivf, int key);

/* returns the set of values with a given key value */
unsigned char * ivf_get_vals (const ivf_t * ivf, int key);

/* returns the set of values with a given key value and given ids */
unsigned char * ivf_find_vals (const ivf_t * ivf, int * keys, int * ids, int n);

/* display the contents of an inverted file  */
void ivf_display (const ivf_t * ivf);

/* Count the total number of elements (descriptors) in the inverted file */
int ivf_count_nbelems (const ivf_t * ivf);

/* compute the imbalance factor */
double ivf_imbalance_factor (const ivf_t * ivf);

/* query the inverted file */
ivfmatch_t * ivf_hequery (const ivf_t * ivf, const int * qids, const int * keys, 
                          const unsigned char * adat, int nq,
                          int * buffer_size, int ht);

/* Alternate implementations, which weights the scores */
ivfmatch_t * ivf_hequeryw (const ivf_t * ivf, const int * qids, const int * keys,
                           const unsigned char * adat, int nq, int ht, int * nm, 
                           const float * score_map_, const float * list_w_);


/* Inverted file I/O */
int ivf_save (const char * fname, const ivf_t * ivf);

ivf_t * ivf_load (const char * fname);

ivfmatch_t * ivfmatch_new (int n);


#endif
