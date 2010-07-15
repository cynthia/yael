#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/kmeans.h>
#include <yael/machinedeps.h>

#include "mex.h"


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 3 || nrhs % 2 != 1) 
    mexErrMsgTxt ("odd nb of input arguments required.");
  else if (nlhs != 1) 
    mexErrMsgTxt ("1 output argument are expected (the centroids).");

  int flags = KMEANS_INIT_USER;
  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  long seed = 0L;
  
  if(mxGetClassID(prhs[0])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  float *v = (float*) mxGetPr (prhs[0]);
  float *centroids0 = (float *) mxGetPr (prhs[1]);
  int k = (int) mxGetScalar (prhs[2]);

  int niter = 50, redo = 1, nt = 1, verbose = 1;

  {
    int i;
    for(i = 3 ; i < nrhs ; i += 2) {
      char varname[256];
      if (mxGetClassID(prhs[i]) != mxCHAR_CLASS) 
        mexErrMsgTxt ("variable name required");         

      if (mxGetString (prhs[i], varname, 256) != 0)
        mexErrMsgTxt ("Could not convert string data");

      if (!strcmp(varname, "niter")) 
        niter = (int) mxGetScalar (prhs[i+1]);

      else if (!strcmp(varname, "nt"))  
	/* !!! Normally, use nt=1 for multi-threading in Matlab: 
	   Blas is already multi-threaded. 
	   Explicit call with nt>1 causes memory leaks */
	nt = (int) mxGetScalar (prhs[i+1]); 
      
      else if (!strcmp(varname,"redo")) 
        redo = (int) mxGetScalar (prhs[i+1]);

      else if (!strcmp(varname,"seed")) 
        seed = (int) mxGetScalar (prhs[i+1]);

      else if (!strcmp(varname,"verbose")) 
        verbose = (int) mxGetScalar (prhs[i+1]);

      else if (!strcmp(varname,"init")) {
	int init_type = (int) mxGetScalar (prhs[i+1]);
	if (init_type == 0)  /* default: Berkeley */
	  ;
	else if (init_type == 1) /* random vectors */
	  flags = flags | KMEANS_INIT_RANDOM;
      }

      else 
        mexErrMsgTxt("unknown variable name");  
    }
  }
  
  /* default: use all the processor cores */
  if (nt == 0)
    nt = 1;

  flags |= nt;

  if (verbose > 0)
    printf("Input: %d vectors of dimension %d\nk=%d niter=%d nt=%d "
	   "redo=%d verbose=%d seed=%d v1=[%g %g ...], v2=[%g %g... ]\n",
	   n, d, k, niter, nt, redo, verbose, seed, v[0], v[1], v[d], v[d+1]); 
  else
    flags |= KMEANS_QUIET;
  

  if(n < k) {
    mexErrMsgTxt("fewer points than centroids");    
  }


  /* ouptut: centroids, assignment, distances */

  plhs[0] = mxCreateNumericMatrix (d, k, mxSINGLE_CLASS, mxREAL);
  float *centroids=(float*)mxGetPr(plhs[0]);

  kmeans (d, n, k, niter, v, flags, seed, 
	  redo, centroids, centroids0, NULL, NULL);

}
