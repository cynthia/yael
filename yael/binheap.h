#ifndef __binheap_h
#define __binheap_h

#include <stdlib.h>

/*********************************************************************
 * Binary heap used as a maxheap. May be used to find the maxk smallest
 * elements of a possibly unsized stream of values. 
 * Element (label[1],val[1]) always contains the maximum value of the binheap.
 *********************************************************************/

typedef struct fbinheap_t {
  float * val;     /* valid values are val[1] to val[k-1] */
  int * label;     /* idem for labels */
  int k;           /* number of elements stored  */
  int maxk;        /* maximum number of elements */
} fbinheap_t;


/* create (allocate) the maxheap structure for n elements (maximum) */
fbinheap_t * fbinheap_new (int maxk);

/* a binheap can be stored in an externally allocated memory area of
   fbinheap_sizeof(maxk) bytes. It must then be initialized with
   fbinheap_init() */
size_t fbinheap_sizeof(int maxk); 
void fbinheap_init(fbinheap_t *bh,int maxk);

/* free allocated memory */
void fbinheap_delete (fbinheap_t * bh);

/* insert an element on the heap */
void fbinheap_add (fbinheap_t * bh, int label, float val);

/* remove largest value from binheap (low-level access!) */
void fbinheap_pop (fbinheap_t * bh);

/* add n elements on the heap */
void fbinheap_addn (fbinheap_t * bh, int n, const int * label, const float * v);

/* add n elements on the heap, using the set of label starting at label0  */
void fbinheap_addn_label_range (fbinheap_t * bh, int n, int label0, const float * v);

/* output the labels in increasing order of associated values */
void fbinheap_sort_labels (fbinheap_t * bh, int * labels);

/* output the sorted values */
void fbinheap_sort_values (fbinheap_t * bh, float * v);

/* output both sorted results: labels and corresponding values  */
void fbinheap_sort (fbinheap_t * bh, int * labels, float *v);


/* sort by increasing labels, ouptput sorted labels & associated values */
void fbinheap_sort_per_labels (fbinheap_t * bh, int * labels, float *v);




/* show the heap content */
void fbinheap_display (fbinheap_t * bh);



#endif
