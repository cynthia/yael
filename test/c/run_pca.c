#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <yael/vector.h>
#include <yael/matrix.h>


void usage (const char * cmd)
{
  printf ("Usage: %s [-v] -n # -d # -fi fvin [-favg favg] [-fevec fevec] [-feval feval] [-fo fvout]\n", cmd);
  
  printf ("Output: all output file are optional (produced only if option is specified)\n"
	  "  Input\n"
	  "    -v                verbose output\n"
	  "    -n #              number of vectors\n"
          "    -d #              dimension of the vectors\n"
          "    -dout #           dimension of the output vectors\n"
          "    -powerlaw #       pre-process vector using power-law component-wise normalization\n"
          "    -L2               pre-normalization of input vector (may be after powerlaw)\n\n"
	  "    -fi filename      file of input vectors (raw format)\n"
	  "    -fo filename      file of PCA-transformed output vectors (raw format)\n"
	  "    -favg filename    raw file containing the mean values\n"
	  "    -fevec filename   raw file containing the eigenvectors\n"
	  "    -feval filename   raw file containing the eigenvalues\n"
	  );
  exit (0);
}


/* write output file in raw file format */
void write_raw_floats (const char * fname, const float * v, int d)
{
  int ret;

  FILE * f = fopen (fname, "w");
  if (!f) { 
    fprintf (stderr, "Unable to open file %s for writing\n", fname);
    exit (1);
  }

  ret = fwrite (v, sizeof (*v), d, f);
  if (ret != d) {
    fprintf (stderr, "Unable to write %d floats in file %s\n", d, fname);
    exit (2);
  }
  fclose (f);
}


int main (int argc, char **argv)
{
  int i, ret;
  int verbose = 0;
  int d = -1;
  int dout = -1;
  int n = -1;

  const char * vec_fname = NULL;     /* input vector file */
  const char * ovec_fname = NULL;    /* output vector file */
  const char * avg_fname = NULL;   
  const char * evec_fname = NULL; 
  const char * eval_fname = NULL; 

  for (i = 1 ; i < argc ; i++) {
    char *a = argv[i];

    if (!strcmp (a, "-h") || !strcmp (a, "--help"))
      usage (argv[0]);
    else if (!strcmp (a, "-verbose") || !strcmp (a, "-v")) {
      verbose = 2;
    }
    else if (!strcmp (a, "-n") && i+1 < argc) {
      ret = sscanf (argv[++i], "%d", &n);
      assert (ret);
    }
    else if (!strcmp (a, "-d") && i+1 < argc) {
      ret = sscanf (argv[++i], "%d", &d);
      assert (ret);
    }
    else if (!strcmp (a, "-dout") && i+1 < argc) {
      ret = sscanf (argv[++i], "%d", &dout);
      assert (ret);
    }
    else if (!strcmp (a, "-fi") && i+1 < argc) {
      vec_fname = argv[++i];
    }
    else if (!strcmp (a, "-fo") && i+1 < argc) {
      ovec_fname = argv[++i];
    }
    else if (!strcmp (a, "-favg") && i+1 < argc) {
      avg_fname = argv[++i];
    }
    else if (!strcmp (a, "-fevec") && i+1 < argc) {
      evec_fname = argv[++i];
    }
    else if (!strcmp (a, "-feval") && i+1 < argc) {
      eval_fname = argv[++i];
    }
    else {
      fprintf (stderr, "Unknown argument: %s\nAborting...\n", a);
      exit (4);
    }
  }

  if (verbose) {
    printf ("d=%d\nn=%d\nvec=%s\navg=%s\nevec=%s\neval=%s\n",
	    d, n, vec_fname, avg_fname, evec_fname, eval_fname);
    if (avg_fname)  printf ("avg=%s\n", avg_fname);
    if (evec_fname) printf ("evec=%s\n", evec_fname);
    if (eval_fname) printf ("eval=%s\n", eval_fname);
  }

  /* By default, keep all dimensions */
  if (dout == -1)
    dout = d;

  if (d == -1 || n == -1 || !vec_fname)
    usage (argv[0]);


  float *v = fvec_new(n*d);

  if (verbose)
    printf ("* Read data from file %s -> %d vectors of dimension %d\n", vec_fname, n, d);

  FILE * fv = fopen (vec_fname, "r");
  assert (fv);
  ret = fread (v, sizeof (*v), n*d, fv);
  assert (ret == n*d);
  fclose (fv);

  /* compute the mean and subtract it from the vectors */
  if (verbose)
    printf ("* Compute average\n");
  float * avg = fmat_center_columns(d,n,v);

  /* compute the PCA and write eigenvalues and eigenvectors to disk */
  if (verbose)
    printf ("* Compute eigenvectors\n");
  float * eval = fvec_new (d);
  float * evec = fmat_new_pca(d,n,v,eval);  

  if (verbose) {
    printf ("eigenval = ");
    fvec_print (eval, d);
  }
  
  if (avg_fname) 
    write_raw_floats (avg_fname, avg, d);

  if (eval_fname)
    write_raw_floats (eval_fname, eval, d);

  if (evec_fname) 
    write_raw_floats (evec_fname, evec, d*d);

  if (ovec_fname) {
    /* compute the projection of the database vector on the PCA basis */
    float * ovec = fmat_new_mul_tl(evec,v,dout,n,d);
    write_raw_floats (ovec_fname, ovec, d*dout);

    if (verbose) {
      /* compute energy of input and outpt vectors */
      double energy_in = fvec_sum_sqr (v, n * d);
      double energy_out = fvec_sum_sqr (ovec, n * dout);
      printf ("Energy preserved = %.3f\n", (float) (energy_out / energy_in));
    }
    free (ovec);
  }

  free(v);
  free(avg);
  free(eval);
  free(evec);

  return 0;
}

