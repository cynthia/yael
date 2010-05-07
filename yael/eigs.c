/*
Copyright © INRIA 2010. 
Authors: Matthijs Douze & Herve Jegou 
Contact: matthijs.douze@inria.fr  herve.jegou@inria.fr

This software is a computer program whose purpose is to provide 
efficient tools for basic yet computationally demanding tasks, 
such as find k-nearest neighbors using exhaustive search 
and kmeans clustering. 

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vector.h"
#include "sorting.h"
#include "machinedeps.h"



extern void dsyev_( char *jobz, char *uplo, FINTEGER *n, double *a, FINTEGER *lda,
        double *w, double *work, FINTEGER *lwork, FINTEGER *info );

extern void dsygv_(FINTEGER * itype, char *jobz, char *uplo, FINTEGER *n, double *a, FINTEGER *lda,
		    double *b, FINTEGER *lbd, double *w, double *work, FINTEGER *lwork, FINTEGER *info );




int eigs_sym (int d, const float * m, float * eigval, float * eigvec)
{
  int i, j;
  double * md = (double *) memalign (16, sizeof (*md) * d * d);

  /* processing is performed in double precision */
  for (i = 0 ; i < d ; i++) {
    for (j = 0 ; j < d ; j++)
      md[i * d + j] = (float) m[i * d + j];
  }

  /* variable for lapack function */
  double workopt = 0;
  int lwork = -1, info;

  double * lambda = (double *) memalign (16, sizeof (*lambda) * d);
  dsyev_( "V", "L", &d, md, &d, lambda, &workopt, &lwork, &info );
  lwork = (int) workopt;
  double * work = (double *) memalign (16, lwork * sizeof (*work));
  dsyev_( "V", "L", &d, md, &d, lambda, work, &lwork, &info );
  
  if (info > 0) {
    fprintf (stderr, "# eigs_sym: problem while computing eigen-vectors/values info=%d\n",info);
    goto error;
  }
  /* normalize the eigenvectors, copy and free */
  double nr = 1;
  for (i = 0 ; i < d ; i++) {
    eigval[i] = (float) lambda[i];
    
    for (j = 0 ; j < d ; j++) 
      eigvec[i * d + j] = (float) (md[i * d + j] / nr);
  }
 error:
  free (md);
  free (lambda);
  free (work);
  return info;
}


int geigs_sym (int d, const float * a, const float * b, float * eigval, float * eigvec)
{
  int i, j;
  double * ad = (double *) memalign (16, sizeof (*ad) * d * d);
  double * bd = (double *) memalign (16, sizeof (*bd) * d * d);

  /* processing is performed in double precision */
  for (i = 0 ; i < d ; i++) 
    for (j = 0 ; j < d ; j++) {
      ad[i * d + j] = (float) a[i * d + j];
      bd[i * d + j] = (float) b[i * d + j];
    }
  
  /* variable for lapack function */
  double workopt = 0;
  int lwork = -1, info, itype = 1;

  double * lambda = (double *) memalign (16, sizeof (*lambda) * d);
  dsygv_ (&itype, "V", "L", &d, ad, &d, bd, &d, lambda, &workopt, &lwork, &info );
  lwork = (int) workopt;
  double * work = (double *) memalign (16, lwork * sizeof (*work));
  dsygv_ (&itype, "V", "L", &d, ad, &d, bd, &d, lambda, work, &lwork, &info );
  
  if (info != 0) {
    fprintf (stderr, "# eigs_sym: problem while computing eigen-vectors/values info=%d\n",info);
    goto error;
  }

  /* normalize the eigenvectors, copy and free */
  double nr = 1;
  for (i = 0 ; i < d ; i++) {
    eigval[i] = (float) lambda[i];
    
    for (j = 0 ; j < d ; j++) 
      eigvec[i * d + j] = (float) (ad[i * d + j] / nr);
  }

 error:
  free (ad);
  free (bd);
  free (lambda);
  free (work);
  return info;
}



void eigs_reorder (int d, float * eigval, float * eigvec, int criterion)
{
  int i;
  int * perm = ivec_new (d);

  float * eigvalst = fvec_new (d * d);
  float * eigvecst = fvec_new (d * d);

  fvec_sort_index (eigval, d, perm);

  if (criterion) 
    for (i = 0 ; i < d / 2 ; i++) {
      int tmp = perm[i];
      perm[i] = perm[d - 1 - i];
      perm[d - 1 - i] = tmp;
    }

  for (i = 0 ; i < d ; i++) {
    eigvalst[i] = eigval[perm[i]];
    memcpy (eigvecst + i * d, eigvec + perm[i] * d, sizeof (*eigvecst) * d);
  }

  memcpy (eigval, eigvalst, d * sizeof (*eigval));
  memcpy (eigvec, eigvecst, d * d * sizeof (*eigvec));

  free (eigvalst);
  free (eigvecst);
  free (perm);
}




#ifdef HAVE_ARPACK

typedef FINTEGER integer;
typedef FINTEGER logical;
typedef float real;

extern void ssaupd_ (integer *ido,const char*bmat,integer *n, const char*which,integer *nev,
                     float* tol, float*resid, integer *ncv, float *v, integer *ldv, 
                     integer *iparam, integer * ipntr, float *workd, float *workl, 
                     integer *lworkl, integer *info );


extern void sseupd_ (logical *rvec, const char *howmny, logical *select, float *d    ,
                     float *z     ,integer *ldz   , float *sigma , const char*bmat,
                     integer *n       , const char*which,integer *nev, float* tol, 
                     float*resid, integer *ncv, float *v, integer *ldv, 
                     integer *iparam, integer * ipntr, float *workd, float *workl, 
                     integer *lworkl, integer *info );

extern void sgemv_(const char *trans, integer *m, integer *n, real *alpha, 
                   const real *a, integer *lda, const real *x, integer *incx, real *beta, real *y, 
                   integer *incy);

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))


int eigs_sym_part (int n, const float * a, int nev, float * sout, float * vout) {

  int ncv=2*nev;  /* should be enough (see remark 4 of ssaupd doc) */

  float *s=NEWA(float,ncv*2);
  int i,j;
  const char *bmat="I",*which="LM";
  float tol=0;
  int info=0;
  int ido=0;
  int lworkl = ncv*(ncv+8);
  float *resid=NEWA(float,n),*workd=NEWA(float,3*n),*workl=NEWA(float,lworkl);
  float *v=NEWA(float,n*(long)ncv);
  int *iparam=NEWA(int,11),*ipntr=NEWA(int,11);
  logical *select=NEWA(logical,ncv);
  int ret=0;

  iparam[0]=1;
  iparam[2]=n;
  iparam[6]=1;

  i=0;
  for(;;) {

    ssaupd_(&ido, bmat, &n, which, &nev, 
            &tol, resid, &ncv, v, &n, 
            iparam, ipntr, workd, workl, &lworkl,
            &info);

    if(ido==-1 || ido==1) {
      
      float *x=workd+ipntr[0]-1;
      float *y=workd+ipntr[1]-1;

      float zero=0,one=1;
      int ione=1;
            
      sgemv_("Trans",&n,&n,&one,a,&n,x,&ione,&zero,y,&ione);

    } else break;   

    printf("ssaupd eval %d\r",i++); fflush(stdout);
  }
  
  printf("\n ssaupd -> sseupd\n");

  if(info<0) {
    printf("eigs_sym_part: ssaupd_ error info=%d\n",info);
    ret=info;
    goto error;
  } 

  {
    int ierr;
    logical rvec=1;
    float sigma;

    sseupd_(&rvec,"All",select,s,
            v,&n, &sigma, bmat, &n, which, &nev, 
            &tol, resid, &ncv, v, &n, 
            iparam, ipntr, workd, workl, &lworkl,&ierr);

    if(ierr!=0) {
      printf("eigs_sym_part: sseupd_ error: %d\n",ierr);
      ret=ierr;     
      goto error;
    }
    int nconv=iparam[4];

    if(nconv<nev) {
      printf("eigs_sym_part: nev=%d, nconv=%d, increase ncv=%d\n",nev,nconv,ncv);
      ret=0xdeadbeef; 
      goto error;
    }

  }


  /* order v by s */
  
  int *perm=NEWA(int,nev);
  fvec_sort_index(s,nev,perm); 
  
  if(vout) 
    for(i=0;i<nev;i++) 
      memcpy(vout+n*i,v+n*perm[nev-1-i],sizeof(float)*n);

  if(sout) 
    for(i=0;i<nev;i++) 
      sout[i]=s[perm[nev-1-i]];

 error: 
  free(select); 
  free(resid); 
  free(workl);
  free(workd);
  free(iparam);
  free(ipntr);

  free(perm); 
  free(v);
  free(s);

  return ret;
}


#else


int eigs_sym_part (int d, const float * m, int nev, float * eigval, float * eigvec) 
{
  fprintf(stderr,"eigs_sym_part: ERROR Yael not compiled with arpack\n");
  return -1;
}

#endif
