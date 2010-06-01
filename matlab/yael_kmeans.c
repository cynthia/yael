#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/kmeans.h>
#include <yael/machinedeps.h>

#include "mex.h"


/* 
compile with yael4matlab.sh
*/


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 2 || nrhs % 2 != 0) 
    mexErrMsgTxt("even nb of input arguments required.");
  else if (nlhs != 3) 
    mexErrMsgTxt("3 output arguments required.");

  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  
  if(mxGetClassID(prhs[0])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  float *v = (float*) mxGetPr (prhs[0]);
  int k = (int) mxGetScalar (prhs[1]);

  int niter = 50, redo = 1, nt = 1;

  {
    int i;
    for(i = 2 ; i < nrhs ; i += 2) {
      char varname[256];
      if (mxGetClassID(prhs[i]) != mxCHAR_CLASS) 
        mexErrMsgTxt ("variable name required");         

      if (mxGetString (prhs[i], varname, 256) != 0)
        mexErrMsgTxt ("Could not convert string data");

      if (!strcmp(varname, "niter")) 
        niter = (int) mxGetScalar (prhs[i+1]);

      else if (!strcmp(varname, "nt")) 
        nt = (int) mxGetScalar (prhs[i+1]);
      
      else if (!strcmp(varname,"redo")) 
        redo = (int) mxGetScalar (prhs[i+1]);

      else 
        mexErrMsgTxt("unknown variable name");  
    }
  }
  
  /* default: use all the processor cores */
  if (nt == 0)
    nt = count_cpu();

  printf("input array of %d*%d k=%d niter=%d nt=%d ar=[%g %g ... ; %g %g... ]\n",
         n, d, k, niter, nt, v[0], v[d], v[1], v[d+1]); 

  if(n < k) {
    mexErrMsgTxt("fewer points than centroids");    
  }


  /* ouptut: centroids, assignment, distances */

  plhs[0] = mxCreateNumericMatrix (d, k, mxSINGLE_CLASS, mxREAL);
  
  float *centroids=(float*)mxGetPr(plhs[0]);

  plhs[1] = mxCreateNumericMatrix (n, 1, mxSINGLE_CLASS, mxREAL);
  
  float *dis = (float*) mxGetPr (plhs[1]);

  plhs[2] = mxCreateNumericMatrix (n, 1, mxINT32_CLASS, mxREAL);
  
  int *assign = (int*) mxGetPr (plhs[2]);
  
  kmeans (d, n, k, niter, v, nt, 0, redo, centroids, dis, assign, NULL);
}
