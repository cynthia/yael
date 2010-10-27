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
#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "vector.h"
#include "matrix.h"
#include "sorting.h"
#include "machinedeps.h"
#include "eigs.h"

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)


/* blas/lapack subroutines */

#define real float
#define integer FINTEGER

int sgemm_ (char *transa, char *transb, integer * m, integer *
            n, integer * k, real * alpha, const real * a, integer * lda,
            const real * b, integer * ldb, real * beta, real * c__,
            integer * ldc);

int ssyev_ (char *jobz, char *uplo, integer * n, real * a,
            integer * lda, real * w, real * work, integer * lwork,
            integer * info);


int sgeqrf_ (integer * m, integer * n, real * a, integer * lda,
             real * tau, real * work, integer * lwork, integer * info);

int slarft_ (char *direct, char *storev, integer * n, integer *
             k, real * v, integer * ldv, real * tau, real * t, integer * ldt);

int slarfb_ (char *side, char *trans, char *direct, char *storev, integer * m,
             integer * n, integer * k, real * v, integer * ldv, real * t,
             integer * ldt, real * c__, integer * ldc, real * work,
             integer * ldwork);

int ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
           real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
           ldc);


extern void sgemv_(const char *trans, integer *m, integer *n, real *alpha, 
                   const real *a, integer *lda, const real *x, integer *incx, real *beta, real *y, 
                   integer *incy);

#undef real
#undef integer



/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

float *fmat_new (int nrow, int ncol)
{
  float *m = fvec_new (nrow * (long)ncol);
  return m;
}


void fmat_mul_full(const float *left, const float *right,
                   int mi, int ni, int ki,
                   char *transp,
                   float *result) {

  float alpha = 1;
  float beta = 0;
  FINTEGER m=mi,n=ni,k=ki;
  FINTEGER lda = (transp[0] == 'N' ? m : k);
  FINTEGER ldb = (transp[1] == 'N' ? k : n);
  
  sgemm_ (transp, transp+1, &m, &n, &k,
          &alpha, left, &lda, right, &ldb, &beta, result, &m);

}

float* fmat_new_mul_full(const float *left, const float *right,
                         int m, int n, int k,
                         char *transp) {
  float *result=fmat_new(m,n);

  fmat_mul_full(left, right, m, n, k, transp, result);

  return result;
}



void fmat_mul (const float *left, const float *right, int m, int n, int k, float *mout) {
  fmat_mul_full(left,right,m,n,k,"NN",mout);
}

void fmat_mul_tl (const float *left, const float *right, int m, int n, int k, float *mout) {
  fmat_mul_full(left,right,m,n,k,"TN",mout);
}

void fmat_mul_tr (const float *left, const float *right, int m, int n, int k, float *mout) {
  fmat_mul_full(left,right,m,n,k,"NT",mout);
}

void fmat_mul_tlr (const float *left, const float *right, int m, int n, int k, float *mout) {
  fmat_mul_full(left,right,m,n,k,"TT",mout);
}

float* fmat_new_mul (const float *left, const float *right, int m, int n, int k) {
  return fmat_new_mul_full(left,right,m,n,k,"NN");
}

float* fmat_new_mul_tl (const float *left, const float *right, int m, int n, int k) {
  return fmat_new_mul_full(left,right,m,n,k,"TN");
}

float* fmat_new_mul_tr (const float *left, const float *right, int m, int n, int k) {
  return fmat_new_mul_full(left,right,m,n,k,"NT");
}

float* fmat_new_mul_tlr (const float *left, const float *right, int m, int n, int k) {
  return fmat_new_mul_full(left,right,m,n,k,"TT");
}



void fmat_print (const float *a, int nrow, int ncol)
{
  int i, j;

  printf ("[");
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++)
      printf ("%.5g ", a[i + nrow * j]);
    if (i == nrow - 1)
      printf ("]\n");
    else
      printf (";\n");
  }
}

void fmat_print_tranposed(const float *a, int nrow, int ncol)
{
  int i, j;

  printf ("[");
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++)
      printf ("%.5g ", a[i * ncol + j]);
    if (i == nrow - 1)
      printf ("]\n");
    else
      printf (";\n");
  }
}


/*---------------------------------------------------------------------------*/
/* Matrix manipulation functions                                             */
/*---------------------------------------------------------------------------*/


float *fmat_get_submatrix (const float *a, int nrow, 
                           int nrow_out,
                           int ncol) {
  long i;
  float *b=fmat_new(nrow_out,ncol);
  
  for(i=0;i<ncol;i++) 
    memcpy(b+i*nrow_out,a+i*nrow,nrow_out*sizeof(*a));

  return b;
}

float *fmat_get_rows (const float *a, int ncol, int nrowout, const int *rows)
{
  int i;
  float *b = fmat_new (nrowout, ncol);

  for (i = 0; i < nrowout; i++)
    memcpy (b + i * ncol, a + rows[i] * ncol, sizeof (*a) * ncol);

  return b;
}

float *fmat_get_columns (const float *a, int ncola, int nrow, int ncolout, const int *cols) {
  int i,j;
  float *b = fmat_new (nrow, ncolout);
  for(i=0;i<nrow;i++) 
    for(j=0;j<ncolout;j++)
      b[i*ncolout+j]=a[i*ncola+cols[j]];
  return b;
}

float *fmat_sum_columns (const float *a, int ncol, int nrow) {
  long i,j;
  float *sums=fvec_new_0(ncol);
  for(i=0;i<nrow;i++) 
    for(j=0;j<ncol;j++)
      sums[j]+=a[ncol*i+j];
  return sums;
}

/*---------------------------------------------------------------------------*/
/* Special matrices                                                          */
/*---------------------------------------------------------------------------*/
float *fmat_new_rand_gauss (int nrow, int ncol)
{
  int i;
  float *m = fmat_new (nrow, ncol);

  for (i = 0; i < nrow * ncol; i++)
    m[i] = gaussrand ();

  return m;
}


/* method: we compute the QR decomposition of a matrix with Gaussian
   values */
float *random_orthogonal_basis (int di)
{ 
  FINTEGER d=di;
  int i;


  /* generate a Gaussian matrix */
  float *x = fmat_new_rand_gauss (d, d);

  float *tau = NEWA (float, d);

  {                             /* compute QR decomposition */

    /* query work size */
    float lwork_query;
    FINTEGER lwork = -1, info;
    sgeqrf_ (&d, &d, x, &d, tau, &lwork_query, &lwork, &info);
    assert (info == 0);

    lwork = (int) lwork_query;
    float *work = NEWA (float, lwork);
    sgeqrf_ (&d, &d, x, &d, tau, work, &lwork, &info);
    assert (info == 0);

    free (work);
  }

  /* Decomposition now stored in x and tau. Apply to identity to get
     explicit matrix Q */

  float *q = NEWAC (float, d * d);
  {

    float *t = NEWA (float, d * d);

    slarft_ ("F", "C", &d, &d, x, &d, tau, t, &d);

    for (i = 0; i < d; i++)
      q[i + d * i] = 1;

    float *work = NEWA (float, d * d);

    slarfb_ ("Left", "N", "F", "C",
             &d, &d, &d, x, &d, t, &d, q, &d, work, &d);

    free (t);
    free (work);
  }

  free (tau);
  free (x);
  return q;
}


/* Construct a Hadamard matrix of dimension d using the Sylvester construction.
   d should be a power of 2 */
float *hadamard (int d)
{
  assert ((d & (d - 1)) == 0 || !"d must be power of 2");

  int i, j;
  float *had = fvec_new (d * d);

  if (d == 1) {
    had[0] = 1;
    return had;
  }

  /* Compute the Hadamard matrix of dimension d / 2 */
  int dd = d / 2;
  float *had_part = hadamard (dd);

  for (i = 0; i < dd; i++)
    for (j = 0; j < dd; j++) {
      had[i * d + j] = had_part[i * dd + j];
      had[i * d + j + dd] = had_part[i * dd + j];
      had[(i + dd) * d + j] = had_part[i * dd + j];
      had[(i + dd) * d + j + dd] = -had_part[i * dd + j];
    }

  free (had_part);
  return (had);
}




/*---------------------------------------------------------------------------*/
/* Statistical matrix operations                                             */
/*---------------------------------------------------------------------------*/

float *fmat_center_columns(int d,int n,float *v) 
{
  assert(n>0);

  float *accu=fvec_new_cpy(v,d);
  long i;

  for(i=1;i<n;i++) 
    fvec_add(accu,v+i*d,d);

  fvec_div_by(accu,d,n);
  
  for(i=0;i<n;i++) 
    fvec_sub(v+i*d,accu,d);

  return accu;  
}

void fmat_subtract_from_columns(int d,int n,float *v,const float *avg) {
  long i;
  for(i=0;i<n;i++) 
    fvec_sub(v+i*d,avg,d);
}

void fmat_rev_subtract_from_columns(int d,int n,float *v,const float *avg) {
  long i;
  for(i=0;i<n;i++) 
    fvec_rev_sub(v+i*d,avg,d);

}





void fmat_splat_separable(const float *a,int nrow,int ncol,
                          const int *row_assign,const int *col_assign,
                          int k,
                          float *accu) {
  int i,j;

  for(i=0;i<nrow;i++) for(j=0;j<ncol;j++) {
    accu[row_assign[i]*k+col_assign[j]]+=a[i*ncol+j];
  }

}

int *imat_joint_histogram(int n,int k,int *row_assign,int *col_assign) {
  int *hist=ivec_new_0(k*k);
  int i;

  for(i=0;i<n;i++) 
    hist[row_assign[i]*k+col_assign[i]]++;

  return hist;
}



/******************************************************************
 * Covariance and PCA computation
 *****************************************************************/

/* Input matrix: v(d,n) stored by rows.

   x is v data centered for each dimension 0<=j<n
   x = v - (1/n) * u * m 

   where :
   *   u(n,1) contains only 1's
   *   m = u' * v is the sum of values for each column of v 

   cov is the covariance matrix :

   cov = (1/n) x' * x
       = (1/n) v' * v - (1/n^2) * m' * m

   => no need to allocate an auxiliary array.
*/



float *fmat_new_covariance (int d, int n, const float *v, float *avg, int assume_centered)
{
  
  long i, j;

  float *cov = fvec_new_0 (d * d);
  
  if(!assume_centered) {

    float *sums = avg ? avg : fvec_new(d);
    fvec_0(sums,d);
    
    for (i = 0; i < n; i++)
      for (j = 0; j < d; j++)
        sums[j] += v[i * d + j];
    
    
    for (i = 0; i < d; i++)
      for (j = 0; j < d; j++)
        cov[i + j * d] = sums[i] * sums[j];
    
    
    if(avg)
      for(i=0;i<d;i++) avg[i]/=n;
    else
      free (sums);

  } 

  FINTEGER di=d,ni=n;

  if(0)  {
    float alpha = 1.0 / n, beta = -1.0 / (n * n);
    sgemm_ ("N", "T", &di, &di, &ni, &alpha, v, &di, v, &di, &beta, cov, &di);
  } else if(1) {
    /* transpose input matrix */
    float *vt=fvec_new(n*d);
    for(i=0;i<d;i++) 
      for(j=0;j<n;j++) 
        vt[i*n+j]=v[j*d+i];
    float alpha = 1.0 / n, beta = -1.0 / (n * n);
    
    sgemm_ ("T", "N", &di, &di, &ni, &alpha, vt, &ni, vt, &ni, &beta, cov, &di);
    
    free(vt);
  } else {
    float alpha = 1.0 / n, beta = -1.0 / (n * n);
    ssyrk_("L","N", &di, &ni, &alpha,(float*)v,&di,&beta,cov,&di);

    /* copy lower triangle to upper */

    for(i=0;i<d;i++)
      for(j=i+1;j<d;j++) 
        cov[i+j*d]=cov[j+i*d];

  }

  return cov;
}

float* fmat_new_transp (const float *a, int ncol, int nrow)
{
  int i,j;
  float *vt=fvec_new(ncol*nrow);

  for(i=0;i<ncol;i++) 
    for(j=0;j<nrow;j++) 
      vt[i*nrow+j]=a[j*ncol+i];

  return vt;
}

/* algo from http://cheshirekow.com/blog/?p=4 */
void fmat_inplace_transp(float *a, int ncol, int nrow)
{
  int length,k_start,k_new,k,i,j;

  length=ncol*nrow;

  for(k_start=1; k_start < length; k_start++)
  {
    float temp = a[k_start];
    float aux;
    int abort = 0;

    k_new = k = k_start;
    do
    {
      if( k_new < k_start )
      {
	abort = 1;
	break;
      }
      k = k_new;
      i = k% nrow;
      j = k/nrow;
      k_new = i*ncol + j;
    }while(k_new != k_start);

    if(abort)
      continue;

    k_new = k = k_start;
    do
    {
      aux=temp;
      temp = a[k_new];
      a[k_new]=aux;

      k       = k_new;
      i       = k%nrow;
      j       = k/nrow;
      k_new   = i*ncol + j;
    }while(k_new != k_start);
    
    aux=temp;
    temp = a[k_new];
    a[k_new]=aux;
  }
}





static float *fmat_new_pca_from_covariance(int d,const float *cov,
                                float *singvals) {

  float *pcamat=fvec_new(d*d);
  float *evals=singvals;

  if(!singvals) evals=fvec_new(d);

  if(eigs_sym(d,cov,evals,pcamat)!=0) {
    free(pcamat);
    pcamat=NULL;
    goto error;
  }
  eigs_reorder(d,evals,pcamat,1); /* 1 = descending */

 error:
  if(!singvals) free(evals);

  return pcamat;
}




float *fmat_new_pca(int d,int n,const float *v, float *singvals) {

  float *cov=fmat_new_covariance(d,n,v,NULL,1);
  
  assert(fvec_all_finite(cov,d*d));
  
  float *evals=singvals;

  if(!singvals) evals=fvec_new(d);
  
  float *ret=fmat_new_pca_from_covariance(d,cov,evals);

  if(!singvals) free(evals);
    
  free(cov);  
  
  return ret;
}


















#ifdef _OPENMP

#include <omp.h>

#define SET_NT  omp_set_num_threads(nt)  

#else 

#define SET_NT

/* #pragma's will be ignored */

#endif

/* multithreaded matrix-vector multiply */

/* m=nb rows, n=nb cols */
void fmat_mul_v(int mi,int ni,const float*a,int ldai,
                 const float *x,
                 float *y,int nt) {
  int i;
  FINTEGER lda=ldai,n=ni,m=mi;
  SET_NT;
  
#pragma omp parallel 
  {
#pragma omp for 
    for(i=0;i<nt;i++) {
      int i0=i*(long)m/nt;
      int i1=(i+1)*(long)m/nt;
      FINTEGER m1=i1-i0;
      float one=1.0,zero=0.0;
      FINTEGER ione=1;
      // printf("%d %d\n",i,m1);
      sgemv_("Trans",&n,&m1,&one,
             a+lda*(long)i0,&lda,x,&ione,&zero,y+i0,&ione);
      
    }
  }   

}

void fmat_mul_tv(int mi,int ni,const float*a,int ldai,
                 const float *x,
                 float *y,int nt) {
  int i,j;
  FINTEGER lda=ldai,n=ni,m=mi;
  
  SET_NT;

  float *ybuf=malloc(sizeof(float)*nt*m);

  if(nt>n) nt=n;
#pragma omp parallel 
  {
#pragma omp for 
    for(i=0;i<nt;i++) {
      int i0=i*(long)n/nt;
      int i1=(i+1)*(long)n/nt;
      FINTEGER n1=i1-i0;
      float one=1.0,zero=0.0;
      FINTEGER ione=1;
      sgemv_("Not transposed",&m,&n1,&one,
             a+lda*(long)i0,&lda,x+i0,&ione,&zero,ybuf+i*(long)m,&ione);
      
    }  

  }  
  /* accumulate y results */
  memcpy(y,ybuf,sizeof(float)*m);
  float *yb=ybuf;
  for(i=1;i<nt;i++) {    
    yb+=m;
    for(j=0;j<m;j++) 
      y[j]+=yb[j];
  }

  free(ybuf);
}



int fmat_svd_partial_full(int n,int m,int nev,const float *a,int a_transposed,
                          float *s,float *vout,float *uout,int nt) {
  
  arpack_eigs_t *ae=arpack_eigs_begin(n,nev);
  int ret=0;
  
  int j,i;
  float *ax=NEWA(float,m);
  
  int it;

  for(it=0;;it++) {
    float *x,*y;
    ret=arpack_eigs_step(ae,&x,&y); 

    printf("arpack iteration %d ret=%d\r",it,ret);

    if(ret<0) break; /* error */

    if(ret==0) break; /* stop iteration */

    /* ret==1 */

    if(!a_transposed) {
      fmat_mul_v(m,n,a,n,x,ax,nt);
      fmat_mul_tv(n,m,a,n,ax,y,nt);
    } else {
      fmat_mul_tv(m,n,a,m,x,ax,nt);
      fmat_mul_v(n,m,a,m,ax,y,nt);
    }

    fflush(stdout);
  } 
  printf("\n");

  free(ax);

  float *v=vout ? vout : fmat_new(nev,n);

  ret=arpack_eigs_end(ae,s,v);

  if(ret>0) {
    int nconv=ret;
        
    if(s)
      for(j=0;j<nconv;j++) 
        s[j]=sqrt(s[j]);    

    if(uout) 
      for(i=0;i<nconv;i++) {
        float *u=uout+m*(long)i;
        if(!a_transposed)
          fmat_mul_v(m,n,a,n,v+n*(long)i,u,nt);
        else
          fmat_mul_tv(m,n,a,m,v+n*(long)i,u,nt);
        fvec_normalize(u,m,2);
      }               
    
  }

  if(!vout) free(v);
  
  return ret;
}

int fmat_svd_partial(int d,int n,int ns,const float *a,
                     float *singvals,float *u,float *v) {
  return fmat_svd_partial_full(d,n,ns,a,0,singvals,u,v,count_cpu());
}



float *fmat_new_pca_part(int d,int n,int nev,
                         const float *v,float *singvals) {

  if(!(nev<=d && nev<=n)) {
    fprintf(stderr,"fmat_new_pca_part: asking for too many eigenvalues (%d) wrt %d*%d data\n",nev,n,d);
    return NULL;
  }


  float *pcamat=fmat_new(d,nev);  

  
  int ret;

  if(n>=d) {
    ret=fmat_svd_partial_full(d,n,nev,v,0,singvals,pcamat,NULL,count_cpu());
  } else {
    fprintf(stderr,"fmat_new_pca_part: warn fewer learning points (%d) than dimensions (%d): transposing\n",n,d);
    
    ret=fmat_svd_partial_full(n,d,nev,v,1,singvals,NULL,pcamat,count_cpu());
  }

  if(ret<0) {
    free(pcamat); 
    pcamat=NULL;
  }

  return pcamat;
}

