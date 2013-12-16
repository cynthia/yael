/* This code was written by Herve Jegou. Contact: herve.jegou@inria.fr  */
/* Last change: June 1st, 2010                                          */
/* This software is governed by the CeCILL license under French law and */
/* abiding by the rules of distribution of free software.               */
/* See http://www.cecill.info/licences.en.html                          */

#include <assert.h>
#include "mex.h"
#include "../yael/hamming.h"

void usage (const char * msg) 
{
  char msgtot[1024];
  const char * msgusg = 
    "There are two modes, depending on whether a threshold is given or not\n\n"
    "H = yael_hamming (X, Y);\n\n"
    "       X and Y are set of compact bit vectors (uint8), 1 per column\n"
    "       H is the set of all Hamming distances, in uint16 format\n\n"
    "[ids, hdis] = yael_hamming (X, Y, thres);\n"
    "       ids: matching elements, thres: hamming threshold\n";
  
  sprintf (msgtot, "%s\n\n%s\n", msg, msgusg);
  mexErrMsgTxt (msgtot);
}


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  int mode_thres = 0;
  if (nrhs != 2 && nrhs != 3) 
    mexErrMsgTxt ("This function requires either 2 input arguments.");
  
  if (nrhs == 3) mode_thres = 1; 
  
  int d = mxGetM (prhs[0]);   /* d is the number of codes, i.e., 8 times the number of bits */
  int na = mxGetN (prhs[0]);
  int nb = mxGetN (prhs[1]);

  if (mxGetM (prhs[1]) != d) 
      usage ("Dimension of binary vectors are not consistent");

  if (mxGetClassID(prhs[0]) != mxUINT8_CLASS)
    	usage ("first argument should be uint 8 type"); 

  if (mxGetClassID(prhs[1]) != mxUINT8_CLASS)
      usage ("second argument should be uint8 type"); 

  uint8 * a = (uint8*) mxGetPr (prhs[0]);
  uint8 * b = (uint8*) mxGetPr (prhs[1]);


  /* Just compute all Hamming distances */
  if (mode_thres == 0) {
    if (nlhs > 1)
      usage ("This syntax expects only exactly one output argument");

    /* ouptut: distances */
    plhs[0] = mxCreateNumericMatrix (na, nb, mxUINT16_CLASS, mxREAL);
    uint16 *dis = (uint16*) mxGetPr (plhs[0]);

    compute_hamming (dis, a, b, na, nb, d);
  }
  
  /* Return only the Hamming distances below a given threshold */
  else {
    if (nlhs != 2)
      usage ("This syntax expects only exactly two output arguments");
    int ht = (int) mxGetScalar (prhs[2]);
    size_t totmatches;
    
    match_he_count (a, b, na, nb, ht, d, &totmatches);
    
    plhs[0] = mxCreateNumericMatrix (2, totmatches, mxINT32_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix (1, totmatches, mxUINT16_CLASS, mxREAL);
    
    int * keys = (int *) mxGetPr(plhs[0]);
    uint16 *dis = (uint16*) mxGetPr (plhs[1]);
    
    size_t ret = match_hamming_thres_prealloc (a, b, na, nb, ht, d, keys, dis);
    assert (ret == totmatches);

  long i;
  for (i = 0 ; i < 2*totmatches ; i++)
      keys[i] = keys[i] + 1;
  }
            
}

//%-------------------------------------------
///* Count the number of matches */
//int ht = (int) mxGetScalar (prhs[1]);
//
//size_t * nmatches = (size_t *) malloc (sizeof(*nmatches) * ivf->k);
//ivf_he_count_crossmatches2 (ivf, ht, nmatches);
//
///* compute the cumulative number of matches */
//size_t * cumnmatches = (size_t *) malloc (sizeof (*cumnmatches) * (ivf->k+1));
//cumnmatches[0] = 0;
//for (i = 0 ; i < ivf->k ; i++) 
//cumnmatches[i+1] = nmatches[i] + cumnmatches[i];
//
//size_t totmatches = cumnmatches[ivf->k];
//
//
//plhs[0] = mxCreateNumericMatrix (2, totmatches, mxINT32_CLASS, mxREAL);
//plhs[1] = mxCreateNumericMatrix (1, totmatches, mxUINT16_CLASS, mxREAL);
//plhs[2] = mxCreateNumericMatrix (1, ivf->k-1, mxINT64_CLASS, mxREAL);
//
//ivf_he_crossmatches_prealloc2 (ivf, ht, (int *) mxGetPr(plhs[0]), 
//                               (uint16 *) mxGetPr(plhs[1]), cumnmatches);
//
//memcpy (mxGetPr(plhs[2]), nmatches + off, sizeof (*nmatches) * (ivf->k-1)); 
//
//if (nlhs == 4) {
//  plhs[3] = mxCreateNumericMatrix (1, totmatches, mxINT32_CLASS, mxREAL);
//  int * key = (int *) mxGetPr(plhs[3]);
//  long j;
//  for (i = 0 ; i < ivf->k ; i++)
//    for (j = cumnmatches[i] ; j < cumnmatches[i+1] ; j++) {
//      key[j] = i;
//    }
//}
