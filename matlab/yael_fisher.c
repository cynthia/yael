#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/gmm.h>
#include <yael/machinedeps.h>

#include "mex.h"


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 4) 
    mexErrMsgTxt("At least 4 arguments are required even nb of input arguments required.");
  else if (nrhs != 1) 
    mexErrMsgTxt("yael_fishier produces exactly 1 output argument.");

  int flags = GMM_FLAGS_MU;
  
  if(mxGetClassID(prhs[0])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(prhs[1])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(prhs[2])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(prhs[2])!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  float *v = (float*) mxGetPr (prhs[0]);
  float *w = (float*) mxGetPr (prhs[1]);
  float *mu = (float*) mxGetPr (prhs[2]);
  float *sigma = (float*) mxGetPr (prhs[3]);

  fprintf (stderr, "v     -> %d x %d\n", mxGetM (prhs[0]), mxGetN (prhs[0]));
  fprintf (stderr, "w     -> %d x %d\n", mxGetM (prhs[1]), mxGetN (prhs[1]));
  fprintf (stderr, "mu    -> %d x %d\n", mxGetM (prhs[2]), mxGetN (prhs[2]));
  fprintf (stderr, "sigma -> %d x %d\n", mxGetM (prhs[3]), mxGetN (prhs[3]));
  
  int d = mxGetM (prhs[0]);  /* vector dimensionality */
  int n = mxGetN (prhs[0]);  /* number of fisher vector to produce */
  int k = mxGetN (prhs[1]);  /* number of gaussian */

  if (mxGetM (prhs[2]) != d || mxGetM (prhs[3]) != d || mxGetN (prhs[2]) !=k 
      || mxGetN (prhs[3]) != k || mxGetN (prhs[1]) != 1)
    mexErrMsgTxt("Invalid input dimensionalities.");

  {
    int i;
    for(i = 4 ; i < nrhs ; i += 1) {
      char varname[256];
      if (mxGetClassID(prhs[i]) != mxCHAR_CLASS) 
        mexErrMsgTxt ("variable name required");         

      if (mxGetString (prhs[i], varname, 256) != 0)
        mexErrMsgTxt ("Could not convert string data");

      if (!strcmp(varname, "sigma")) 
	flags |= GMM_FLAGS_SIGMA;
      
      else if (!strcmp(varname,"weights")) 
        flags |= GMM_FLAGS_W;

      else if (!strcmp(varname,"nomu")) 
        flags ^= GMM_FLAGS_MU;

      else 
        mexErrMsgTxt("unknown variable name");  
    }
  }
  

  /* ouptut: GMM, i.e., weights, mu and variances */
  gmm_t g = {d, k, w, mu, sigma};
  int dout = gmm_fisher_sizeof (&g, flags); 
  fprintf (stderr, "Size of the fisher vector = %d\n", dout);

  plhs[0] = mxCreateNumericMatrix (dout, k, mxSINGLE_CLASS, mxREAL);
  float * vf = (float *) mxGetPr (plhs[0]);
  gmm_fisher (n, v, &g, flags, vf);
}
