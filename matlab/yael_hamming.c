/* This code was written by Herve Jegou. Contact: herve.jegou@inria.fr  */
/* Last change: June 1st, 2010                                          */
/* This software is governed by the CeCILL license under French law and */
/* abiding by the rules of distribution of free software.               */
/* See http://www.cecill.info/licences.en.html                          */

#include "mex.h"
#include "../yael/hamming.h"


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs != 2) 
    mexErrMsgTxt ("This function requires exactly 2 input arguments");
  
  if (nlhs > 1)
    mexErrMsgTxt ("This function output exactly 1 argument");

  int d = mxGetM (prhs[0]);   /* d is the number of codes, i.e., 8 times the number of bits */
  int na = mxGetN (prhs[0]);
  int nb = mxGetN (prhs[1]);

  if (mxGetM (prhs[1]) != d)
      mexErrMsgTxt("Dimension of binary vectors are not consistent");

  if (mxGetClassID(prhs[0]) != mxUINT8_CLASS)
    mexErrMsgTxt ("first argument should be uint 8 type"); 

  if (mxGetClassID(prhs[1]) != mxUINT8_CLASS)
    mexErrMsgTxt ("second argument should be uint8 type"); 

  uint8 * a = (uint8*) mxGetPr (prhs[0]);
  uint8 * b = (uint8*) mxGetPr (prhs[1]);

  /* ouptut: distances */
  plhs[0] = mxCreateNumericMatrix (na, nb, mxUINT16_CLASS, mxREAL);
  uint16 *dis = (uint16*) mxGetPr (plhs[0]);

  if (BITVECBYTE == d) 
    compute_hamming (dis, a, b, na, nb);
  else
    compute_hamming_generic (dis, a, b, na, nb, d); 
}
