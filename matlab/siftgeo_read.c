#include "mex.h"

#define SIFTGEO_SIZE      168
#define SIFTGEO_DIM_DES   128
#define SIFTGEO_DIM_META  9

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

#ifdef HAVE_OCTAVE
#define mwSize long
#endif

  /* For the filename */
  mwSize buflen;
  
  char *fsiftgeo_name;
  FILE * fsiftgeo;
  long n;
  int i, j, ret;
    
  /* SIFT descriptors and associated meta-data */
  float * v;
  float * meta;
  unsigned char * fbuffer;
    
  /* check for proper number of arguments */
  if(nrhs<1 || nrhs>3) 
    mexErrMsgTxt("Usage: [v,g]=siftgeo_read(filename, nmax).");
  else if(nlhs > 2) 
    mexErrMsgTxt("Too many output arguments.");

  /* input must be a string */
  if (mxIsChar(prhs[0]) != 1)
    mexErrMsgTxt("Input must be a string.");
  
  /* input must be a row vector */
  if (mxGetM(prhs[0])!=1)
    mexErrMsgTxt("Input must be a row vector.");
  
  /* get the length of the input string */
  buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;
  
  /* copy the string data from prhs[0] into a C string input_ buf.    */
  fsiftgeo_name = mxArrayToString(prhs[0]);
  
  if(fsiftgeo_name == NULL) {
    mxFree(fsiftgeo_name);
    mexErrMsgTxt("Could not convert input to string.");
  }

  /* open the file for reading and retrieve it size */
  fsiftgeo = fopen (fsiftgeo_name, "r");
  if (!fsiftgeo) 
    mexErrMsgTxt("Could not open the input file");
  mxFree(fsiftgeo_name);
  
  fseek (fsiftgeo, 0, SEEK_END);
  n = ftell (fsiftgeo) / SIFTGEO_SIZE;
  fseek (fsiftgeo, 0, SEEK_SET);


  /* optionally read another argument to read a maximum number of input */
  if (nrhs == 2) 
    n = (int) mxGetScalar (prhs[1]);
  
  /* or directly specify the start and the end */
  if (nrhs == 3) {
    int posstart = (int) mxGetScalar (prhs[1]);
    int posend = (int) mxGetScalar (prhs[2]);
    if (posend < posstart || posstart < 0 || posend < 0) 
      mexErrMsgTxt("Invalid boundaries");
    n = posend - posstart + 1;
    fseek (fsiftgeo, posstart * SIFTGEO_SIZE, SEEK_SET);
  }

   /* Read all the data using a single read function, and close the file */
  fbuffer = malloc (n * SIFTGEO_SIZE);
  ret = fread (fbuffer, sizeof (*fbuffer), n * SIFTGEO_SIZE, fsiftgeo);
  if (ret != n * SIFTGEO_SIZE)
    mexErrMsgTxt("Unable to read correctly from the input file");
  fclose (fsiftgeo);

  const mwSize dimv[2] = {SIFTGEO_DIM_DES, n};
  const mwSize dimmeta[2] = {SIFTGEO_DIM_META, n};

  /* Allocate the output matrices */
  plhs[0] = mxCreateNumericArray(2, dimv, mxSINGLE_CLASS, mxREAL);
  v = (float *) mxGetPr(plhs[0]);
  
  if (nlhs > 1) {
    plhs[1] = mxCreateNumericArray(2, dimmeta, mxSINGLE_CLASS, mxREAL);
    meta = (float *) mxGetPr(plhs[1]);
  }

  /* Copy the data from the buffer into these variables */
  for (i = 0 ; i < n ; i++) {
    
    for (j = 0 ; j < SIFTGEO_DIM_DES ; j++)
      v[j + SIFTGEO_DIM_DES * i] = fbuffer[i * SIFTGEO_SIZE + j
	      + SIFTGEO_DIM_META * sizeof (float) + sizeof (int)];

    if (nlhs > 1) {
      float * fbuf = (float *) (fbuffer + i * SIFTGEO_SIZE);
      for (j = 0 ; j < SIFTGEO_DIM_META ; j++)
        meta[j + SIFTGEO_DIM_META * i] = fbuf[j];
    }
  }
  free (fbuffer);
}
