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
          "    -plaw #           pre-process vector using sqrt component-wise normalization\n"
          "    -norm #           pre-normalization of input vector (may be after powerlaw)\n\n"
	  "    -fi filename      file of input vectors (raw format)\n"
	  "    -fo filename      file of PCA-transformed output vectors (raw format)\n"
	  "    -favg filename    raw file containing the mean values\n"
	  "    -fevec filename   raw file containing the eigenvectors\n"
	  "    -feval filename   raw file containing the eigenvalues\n"
	  );
  exit (0);
}


/* write output file in raw file format */
void write_raw_floats (const char * fname, const float * v, long n)
{
  int ret;

  FILE * f = fopen (fname, "w");
  if (!f) { 
    fprintf (stderr, "Unable to open file %s for writing\n", fname);
    exit (1);
  }

  ret = fwrite (v, sizeof (*v), n, f);
  if (ret != n) {
    fprintf (stderr, "Unable to write %ld floats in file %s\n", n, fname);
    exit (2);
  }
  fclose (f);
}


/* Pre-processing */
void preprocess (float * v, int d, long n, float plaw, float norm)
{
  /* Pre-processing: power-law on components */
  if (plaw >= 0) {
     fvec_spow (v, n * d, plaw);
  }
  
  /* Pre-processing: normalization */
  if (norm >= 0) {
    fvecs_normalize (v, n, d, norm);
    fvec_purge_nans (v, n * d, 0);
  }
}


/* Online PCA -> accumulating covariance matrice on-the-fly, using blocks of data */
#define PCA_BLOCK_SIZE 256

pca_online_t * pca_online (long n, int d, const char * fname, float plaw, float norm)
{
  long i;

  FILE * f = fopen (fname, "r");
  if (!f) { 
    fprintf (stderr, "Unable to open file %s for reading\n", fname);
    exit (1);
  }

  printf ("* PCA: accumulate mean and covariance matrix\n");

  pca_online_t * pca = pca_online_new (d);
  float * vbuf = fvec_new (PCA_BLOCK_SIZE * d);

  for (i = 0 ; i < n ; i += PCA_BLOCK_SIZE) {
    long iend = i + PCA_BLOCK_SIZE;
    if (iend > n) iend = n;
    long ntmp = iend - i;

    int ret = fread (vbuf, sizeof (*vbuf), ntmp * d, f);
    if (ret != ntmp * d) {
      fprintf (stderr, "Unable to read %ld floats in file %s\n", n, fname);
      exit (2);
    }
    preprocess (vbuf, d, ntmp, plaw, norm);

    pca_online_accu (pca, vbuf, ntmp);
  }

  printf ("* PCA: perform the eigen-decomposition\n");
  pca_online_complete (pca);

  free (vbuf);
  fclose (f);
  return pca;
}



/* Apply the matrix multiplication by block */
void apply_pca (const struct pca_online_s * pca, 
		const char * finame, const char * foname, 
		int d, long n, int dout, float plaw, float norm)
{
  int ret;
  long i, ntmp = -1;

  FILE * fi = fopen (finame, "r");
  if (!fi) { 
    fprintf (stderr, "Unable to open file %s for reading\n", finame);
    exit (1);
  }

  FILE * fo = fopen (foname, "w");
  if (!fo) { 
    fprintf (stderr, "Unable to open file %s for writing\n", foname);
    exit (1);
  }

  float * vibuf = fvec_new (PCA_BLOCK_SIZE * d);
  float * vobuf = fvec_new (PCA_BLOCK_SIZE * dout);

  for (i = 0 ; i < n ; i += PCA_BLOCK_SIZE) {
    long iend = i + PCA_BLOCK_SIZE;
    if (iend > n) iend = n;
    ntmp = iend - i;
    
    ret = fread (vibuf, sizeof (*vibuf), ntmp * d, fi);
    if (ret != d * ntmp) {
      fprintf (stderr, "Unable to read %ld floats in file %s\n", n, finame);
      exit (2);
    }

    preprocess (vibuf, d, ntmp, plaw, norm);
    pca_online_project (pca, vibuf, vobuf, d, ntmp, dout);

    ret = fwrite (vobuf, sizeof (*vobuf), ntmp * dout, fo);
    if (ret != dout * ntmp) {
      fprintf (stderr, "Unable to write %ld floats in file %s\n", n, foname);
      exit (2);
    }
  }  

  fmat_center_columns (d, ntmp, vibuf);
  double energy_in = fvec_sum_sqr (vibuf, ntmp * d);
  double energy_out = fvec_sum_sqr (vobuf, ntmp * dout);
  printf ("Last block: Energy preserved = %.3f\n", (float) (energy_out / energy_in));

  free (vibuf);
  free (vobuf);
}



int main (int argc, char **argv)
{
  int i, ret;
  int verbose = 0;
  int d = -1;
  int dout = -1;
  long n = -1;
  
  float plaw = -1;
  float norm = -1;

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
      ret = sscanf (argv[++i], "%ld", &n);
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
    else if (!strcmp (a, "-plaw") && i+1 < argc) {
      ret = sscanf (argv[++i], "%f", &plaw);
      assert (ret);
    }
    else if (!strcmp (a, "-norm") && i+1 < argc) {
      ret = sscanf (argv[++i], "%f", &norm);
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
    printf ("d=%d\nn=%ld\nvec=%s\navg=%s\nevec=%s\neval=%s\novec=%s\n",
	    d, n, vec_fname, avg_fname, evec_fname, eval_fname, ovec_fname);
  }

  if (d == -1 || n == -1 || !vec_fname)
    usage (argv[0]);

  /* By default, keep all dimensions */
  if (dout == -1)
    dout = d;


  /* Online PCA learning */
  pca_online_t * pca = pca_online (n, d, vec_fname, plaw, norm);


  if (verbose) {
    printf ("eigenval = ");
    fvec_print (pca->eigval, d);
  }
  
  if (avg_fname) 
    write_raw_floats (avg_fname, pca->mu, d);

  if (eval_fname)
    write_raw_floats (eval_fname, pca->eigval, d);

  if (evec_fname) 
    write_raw_floats (evec_fname, pca->eigvec, d*d);


  /* Optionnally, apply the PCA */
  if (ovec_fname) {
    apply_pca (pca, vec_fname, ovec_fname, d, n, dout, plaw, norm);    
  }

  pca_online_delete (pca);
  return 0;
}

