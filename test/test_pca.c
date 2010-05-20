#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <yael/vector.h>
#include <yael/matrix.h>


/* 
Copy/paste the points matrix into octave and compare:

[d,n]=size(points)

avg=sum(points,2)/n;

centered_points=points-avg*ones(1,n);

cov=centered_points * centered_points';

[cov_eigenvectors,cov_eigenvalues]=eig(cov);

cov_eigenvalues=diag(cov_eigenvalues);

[sorted,perm]=sort(cov_eigenvalues)
cov_eigenvectors(:,perm(end:-1:1))'

Lines should be the same up to the sign as eigs_f output


*/




int main (int argc, char **argv)
{
  int d,n;

  if(argc!=3 || sscanf(argv[1],"%d",&n)!=1 || sscanf(argv[2],"%d",&d)!=1) {
    fprintf(stderr,"usage: test_pca npt ndim\n");
    return 1;
  }


  int i;
  float *points=fvec_new(n*d);
  
  for(i=0;i<n*d;i++) points[i]=drand48()*2-1;

  printf("points=");
  fmat_print(points,d,n);

  float *eig_f=fmat_pca(d,n,points);
  
  
  printf("eig_f=");
  fmat_print(eig_f,d,d);


  free(points);
  free(eig_f);

  return 0;
}

