#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <yael/vector.h>
#include <yael/matrix.h>


/* 
Copy/paste the points matrix into octave and compare:

[d,n]=size(centered_v)

cov=centered_v * centered_v';

[cov_eigenvectors,cov_eigenvalues]=eig(cov);

cov_eigenvalues=diag(cov_eigenvalues);

[sorted,perm]=sort(cov_eigenvalues)
cov_eigenvectors(:,perm(end:-1:1))

Lines should be the same up to the sign as eigs_f output


*/


void test_std (long n, long d, float *v) 
{
  fmat_center_columns(d,n,v);

  printf("\ncentered_v=");
  fmat_print(v,d,n);

  float *eig_f=fmat_new_pca(d,n,v,NULL);  
  
  printf("\neig_f=");
  fmat_print(eig_f,d,d);

  free(eig_f);
}



/* Another way to do it by accumulating covariance matrice on-the-fly, using blocks of data */
#define PCA_BLOCK_SIZE 4

void cov_accu (const float * v, long d, long n, float * cov, float * mu)
{
  fmat_sum_rows (v, d, n, mu);
  fmat_mul_tr (v, v, d, d, n, cov);
}



/* Apply the matrix multiplication by block */
void apply_pca (const float * eigs, const float * mu, float * v, float * vo, long n, long d)
{
  long i;
  int dout = d;
  const char trmat[2] = {'T', 'N'};

  for (i = 0 ; i < n ; i += PCA_BLOCK_SIZE) {
    long iend = i + PCA_BLOCK_SIZE;
    if (iend > n) iend = n;
    long ntmp = iend - i;
    
    float * vb = v + i * d;

    fmat_subtract_from_columns (d, ntmp, vb, mu);

    fmat_mul_full (eigs, v, dout, n, d, trmat, vo);
  }  
}


void test_online (long n, long d, float *v)
{
  long i;

  float * cov = fvec_new_0 (d*d);
  float * cov_tmp = fvec_new (d*d);
  float * mu = fvec_new_0 (d);
  float * mu_tmp = fvec_new (d * d);

  for (i = 0 ; i < n ; i += PCA_BLOCK_SIZE) {
    long iend = i + PCA_BLOCK_SIZE;
    if (iend > n) iend = n;
    long ntmp = iend - i;
    float * vb = v + i * d;

    cov_accu (vb, d, ntmp, cov_tmp, mu_tmp);

    fvec_add (mu, mu_tmp, d);
    fvec_add (cov, cov_tmp, d*d);
  }

  /* compute the covariance matrix */
  fvec_div_by (mu, d, n);
  fvec_div_by (cov, d * d, n);
  
  fmat_mul_tr (mu, mu, d, d, 1, mu_tmp);
  fvec_sub (cov, mu_tmp, d*d);

  assert(fvec_all_finite(cov,d*d));

  float * eigval = fvec_new (d);
  float * eigvec = fmat_new_pca_from_covariance (d, cov, eigval);

  printf("\neig_f=");
  fmat_print(eigvec,d,d);

  float * vo = fvec_new (n*d);

  apply_pca (eigvec, mu, v, vo, n, d);

  free (eigvec);
  free (eigval);
}



int main (int argc, char **argv)
{
  int d,n;

  if(argc!=3 || sscanf(argv[1],"%d",&n)!=1 || sscanf(argv[2],"%d",&d)!=1) {
    fprintf(stderr,"usage: test_pca npt ndim\n");
    return 1;
  }


  long i;
  float *v = fvec_new(n*d);
  for (i=0;i<n*d;i++) 
    v[i]=drand48()*2-1;
  float * v1 = fvec_new_cpy (v, n*d);

  /* reference version */
  test_std(n, d, v1);

  /* version with on-line reading of vectors */
  test_online(n, d, v);

  /* Project the vector using the PCA matrix */

  free(v);

  return 0;
}

