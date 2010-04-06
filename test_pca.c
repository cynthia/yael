#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vector.h"
#include "matrix.h"


/* 
Copy/paste the points matrix into octave and compare:

[n,d]=size(points)

centered_points=points-repmat(mean(points),n,1);

cov=centered_points' * centered_points;

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


  int i,j;
  float *points=fvec_new(n*d);
  
  for(i=0;i<n*d;i++) points[i]=drand48()*2-1;

  {
    const char*pf="points=[";

    for(i=0;i<n;i++) {
      for(j=0;j<d;j++) {
        printf("%s%g",pf,points[i*d+j]);
        pf=" ";
      }
      pf=";\n";
    }
    printf("]\n");
  }  
  float *eig_f=compute_pca(n,d,points);

  {
    const char*pf="eigs_f=[";
    
    for(i=0;i<d;i++) {
      for(j=0;j<d;j++) {
        printf("%s%g",pf,eig_f[i*d+j]);
        pf=" ";
    }
      pf=";\n";
    }
    printf("]\n");
  }

  free(points);
  free(eig_f);

  return 0;
}

