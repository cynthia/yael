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


void test1 (int n, int d, float *v) 
{
  fmat_center_columns(d,n,v);

  printf("\ncentered_v=");
  fmat_print(v,d,n);

  float *eig_f=fmat_new_pca(d,n,v,NULL);  
  
  printf("\neig_f=");
  fmat_print(eig_f,d,d);

  free(eig_f);
}


void cov_accu (const float * v, int d, int n, float * cov, float * mu)
{
  const char sttmp[2]  = {'T', 'N'};
  fprintf (stderr, "s0-1\n");
  fmat_sum_rows (v, d, n, mu);
  fprintf (stderr, "s0-2\n");
  fmat_mul_full (v, v, d, n, d, sttmp, cov);
  fprintf (stderr, "s0-3\n");
}


/* Another way to do it by accumulating covariance matrice on-the-fly */
# define PCA_BLOCK_SIZE 4

void test2 (int n, int d, float *v)
{
  int i, j;

  float * cov = fvec_new_0 (d*d);
  float * cov_tmp = fvec_new (d*d);
  float * mu = fvec_new_0 (d);
  float * mu_tmp = fvec_new (d);

  fprintf (stderr, "s0\n");

  for (i = 0 ; i < n ; i += PCA_BLOCK_SIZE) {
    int iend = i + PCA_BLOCK_SIZE;
    if (iend > n) iend = n;
    
    int ntmp = iend - i;
    
    cov_accu (v, d, ntmp, cov_tmp, mu_tmp);

    fvec_add (mu, mu_tmp, d);
    fvec_add (cov, cov_tmp, d*d);
  }

  fprintf (stderr, "s1\n");

  /* compute the covariance matrix */
  fvec_div_by (mu, d, n);
  fvec_div_by (cov, d*d, n-1);
  
  fmat_mul_tr (mu, mu, d, d, 1, mu_tmp);
  fvec_sub (cov, mu_tmp, d*d);

  fprintf (stderr, "s2\n");

  float * eigval = fvec_new (d);
  float * eigvec = fmat_new_pca_from_covariance (d, cov, eigval);

  printf("\neig_f=");
  printf("eig_f=");
  fmat_print(eigvec,d,d);

}



int main (int argc, char **argv)
{
  int d,n;

  if(argc!=3 || sscanf(argv[1],"%d",&n)!=1 || sscanf(argv[2],"%d",&d)!=1) {
    fprintf(stderr,"usage: test_pca npt ndim\n");
    return 1;
  }


  int i;
  float *v = fvec_new(n*d);
  for (i=0;i<n*d;i++) 
    v[i]=drand48()*2-1;
  float * v1 = fvec_new_cpy (v, n*d);

  /* reference version */
  test1(n, d, v1);

  test2(n, d, v);

  free(v);

  return 0;
}

