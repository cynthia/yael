#include "mex.h"
#include <yael/vector.h>

void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 1 || nrhs > 4) 
    mexErrMsgTxt ("Invalid number of input arguments");
  
  if (nlhs < 1 || nlhs > 2)
    mexErrMsgTxt ("1 o 2 output arguments are required");

  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("First parameter must be a single precision matrix"); 


  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  int norm = 2; 
  int i;

  if (nrhs >= 2)
    norm = mxGetScalar (prhs[1]);

  /* the set of vectors to be normalized */
  float * v = (float*) mxGetPr (prhs[0]);  

  plhs[0] = mxCreateNumericMatrix (d, n, mxSINGLE_CLASS, mxREAL);
  float * vo = (float*) mxGetPr (plhs[0]);
  
  /* norm should be double precision */
  plhs[1] = mxCreateNumericMatrix (n, 1, mxDOUBLE_CLASS, mxREAL);
  double * vnr = (double*) mxGetPr (plhs[1]);


  for (i = 0 ; i < n ; i++) {
    double nrtmp = fvec_norm (v + i * d, d, norm);
    vnr[i] = nrtmp;
    nrtmp = 1 / nrtmp;

    long j = 0;
    for (j = 0 ; j < d ; j++)
      vo[i*d+j] = v[i*d+j] * nrtmp;
  }
}
