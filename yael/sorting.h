/*
Copyright © INRIA 2010. 
Authors: Matthijs Douze & Herve Jegou 
Contact: matthijs.douze@inria.fr  herve.jegou@inria.fr

This software is a computer program whose purpose is to provide 
efficient tools for basic yet computationally demanding tasks, 
such as find k-nearest neighbors using exhaustive search 
and kmeans clustering. 

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
*/


#ifndef SORTING_H_INCLUDED
#define SORTING_H_INCLUDED

/*---------------------------------------------------------------------------*/
/*! @addtogroup sorting
 *  @{  */

/* Various sorting functions + a few simple array functions that can
   be called from python efficiently */

/*! 
 tab is a n-element table
 fills maxes such that

 tab[maxes[0]] >= tab[maxes[1]] >= ... >= tab[maxes[k-1]] >= tab[i] 

 for all i not in maxes.
 

*/
void fvec_find_k_max(const float *tab,int n,
		     int *maxes, int k);


void fvec_find_k_min(const float *tab, int n,
		     int *maxes, int k);


/*! finds the ranks of vals[i] for i=0..nval-1 in tab if it was sorted
 * by *decreasing* order
 * minranks[i]-1 is the highest index of values > vals[i]
 * maxranks[i] is the lowest index of values < vals[i]
 * both may be NULL if you're not interested
 */ 
void fvec_ranks_of(const float *tab,int n,
                     const float *vals,int nval,
                     int *minranks,int *maxranks);

/* idem but ranks in increasing array */
void fvec_ranks_inc_of(const float *tab, int n,
                         const float *vals, int nval,
                         int *minranks, int *maxranks);


/*---------------------------------------------------------------------------*/
/* Simple index functions (useful to call from C)                            */
/*---------------------------------------------------------------------------*/

/*! @brief Replace ilabels[i] with the location of ilabels[i] in the table labels.
 *
 *  on input: labels[nres],ilabels[nilabels]\n
 *  on output: labels[ilabels_out[i]]=ilabels[i] for 0<=i<nilabels or -1 if there is none
*/
void find_labels (int *labels, int nres, int *ilabels, int nilabels);

/*! count nb of 0s in array */
int fvec_count_0(const float *val,int n); 

float fvec_min(const float *f, long n);
float fvec_max(const float *f, long n);
int fvec_min_index(const float *f, long n);
int fvec_max_index(const float *f, long n);


/*! computes the median of a float array. Array modified on output! */
float fvec_median (float *f,int n);

/* idem, without modifying array */
float fvec_median_const (const float *f,int n);

/*! computes the arg min of a float array */
int fvec_arg_min (const float *f, int n);



/*! find quantile so that q elements are <= this quantile. On ouput
  the 0..q-1 elements of f are below this quantile */
float fvec_quantile(float *f,int n,int q);


/*! in-place sort */
void ivec_sort(int *tab, int n);

/*! return permutation to sort an array. Is stable. */
void ivec_sort_index (const int *tab, int n, int *perm);

/* fill-in iperm so that iperm[perm[i]]=i for i=0..n-1 */
void ivec_invert_perm(const int *perm, int n, int *iperm); 


/*! in-place sort */
void fvec_sort(float *tab, int n);

/*! return permutation to sort an array. Is stable. */
void fvec_sort_index (const float *tab, int n, int *perm);

/*! sort according to the input permutation. The permutation is 
   typically generated using the ivec_sort_index function. In that 
   case the function perform the sort accordingly. 
*/
void ivec_sort_by_permutation (int * v, const int * order, int n);



/*! count occurrences of val in sorted vector */
int ivec_sorted_count_occurrences(const int *v,int n,int val);

/*! find index of highest value <= val (=-1 if all values are > val) */
int ivec_sorted_find(const int *v,int n,int val);

/*! count unique occurrences  */
int ivec_sorted_count_unique(const int *v,int n);

/*! count nb of occurrences of several values */
int ivec_sorted_count_occurrences_multiple (const int *v,int n,
                                            const int *vals,int nval);


/* merge k ordered sets defined by 
 * 
 * [(lists[i][j],vals[i][j]),j=0..sizes[i]-1]
 * 
 * for i=0..k-1 
 * 
 * the individual lists are supposes to be ordered already.
 * 
 * returns total number of elements (=sum(sizes[i],i=0..k-1))
 */
int merge_ordered_sets (const int **labels, const float **vals,
                        const int *sizes, int k,
                        int **labels_out, float **vals_out); 


/* finds the smallest value m of vals, compresses array labels by
removing labels[i] for which vals[i] < m * ratio returns new size
of labels array */
int compress_labels_by_disratio (int *labels, const float *vals, int n, float ratio); 


/*! @} */
#endif
