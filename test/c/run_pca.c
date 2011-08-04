#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <yael/vector.h>
#include <yael/matrix.h>


void usage (const char * cmd)
{
  printf ("Usage: %s [options]\n", cmd);
  
  printf (
	  "  Input\n"
	  "    -v                verbose output\n"
	  "    -fi filename      file of input vectors (raw format)\n"
	  "    -favg filename    raw file containing the mean values\n"
	  "    -fevec filename   raw file containing the eigenvectors\n"
	  "    -feval filename   raw file containing the eigenvalues\n"
	  "    -n #              number of vectors\n"
          "    -d #              dimension of the vectors)\n\n"
	  );
  exit (0);
}



int main (int argc, char **argv)
{
  int i, ret;
  int verbose = 0;
  int d = -1;
  int n = -1;

  const char * vec_fname = NULL;     /* input vector file */
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
    else if (!strcmp (a, "-fi") && i+1 < argc) {
      vec_fname = argv[++i];
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
  }

  if (d == -1 || n == -1 || vec_fname || avg_fname || evec_fname || eval_fname)
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
    printf ("* Write Read data from file %s -> %d vectors of dimension %d\n", vec_fname, n, d);
  float * avg = fmat_center_columns(d,n,v);
  FILE * favg = fopen (avg_fname, "r");
  ret = fwrite (avg, sizeof (*avg), d, favg);
  assert (ret == d);
  fclose (favg);

  /* compute the PCA and write eigenvalues and eigenvectors to disk */
  if (verbose)
    printf ("* Compute eigenvectors\b", vec_fname, n, d);
  float * eval = fvec_new (d);
  float * evec = fmat_new_pca(d,n,v,eval);  

  FILE * feval = fopen (eval_fname, "r");
  ret = fwrite (eval, sizeof (*eval), d, feval);
  assert (ret == d);
  fclose (feval);
  
  FILE * fevec = fopen (evec_fname, "r");
  ret = fwrite (evec, sizeof (*evec), d * d, fevec);
  assert (ret = d * d);
  fclose (fevec);


  free(v);
  free (avg);
  free (eval);
  free(evec);

  return 0;
}

