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

/* Generate Gaussian random value, mean 0, variance 1  
   From Python source code. */

#define NV_MAGICCONST  1.71552776992141

static double gaussrand ()
{
  double z;
  while (1) {
    float u1, u2, zz;
    u1 = drand48 ();
    u2 = drand48 ();
    z = NV_MAGICCONST * (u1 - .5) / u2;
    zz = z * z / 4.0;
    if (zz < -log (u2))
      break;
  }
  return z;
}



/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

float *fmat_new (int nrow, int ncol)
{
  float *m = fvec_new (nrow * ncol);
  return m;
}


void fmat_mul (const float *left, const float *right,
               int n, int m, int k, float *mout)
{

  float alpha = 1;
  float beta = 0;
  char trans = 'N';

  sgemm_ (&trans, &trans, &k, &n, &m,
          &alpha, right, &k, left, &m, &beta, mout, &k);
}


float *fmat_new_mul (const float *left, const float *right, int n, int m,
                     int k)
{
  float *a = fmat_new (k, n);
  fmat_mul (left, right, n, m, k, a);
  return a;
}


void fmat_mul_tl (const float *left, const float *right,
                  int n, int m, int k, float *mout)
{

  float alpha = 1;
  float beta = 0;
  char transleft = 't';
  char transright = 'n';

  sgemm_ (&transright, &transleft, &k, &n, &m,
          &alpha, right, &k, left, &n, &beta, mout, &k);
}


float *fmat_new_mul_tl (const float *left, const float *right,
                        int n, int m, int k)
{
  float *a = fmat_new (k, n);
  fmat_mul_tl (left, right, n, m, k, a);
  return a;
}


void fmat_mul_tr (const float *left, const float *right,
                  int n, int m, int k, float *mout)
{

  float alpha = 1;
  float beta = 0;
  char transleft = 'n';
  char transright = 't';

  sgemm_ (&transright, &transleft, &k, &n, &m,
          &alpha, right, &m, left, &m, &beta, mout, &k);
}


float *fmat_new_mul_tr (const float *left, const float *right,
                        int n, int m, int k)
{
  float *a = fmat_new (k, n);
  fmat_mul_tr (left, right, n, m, k, a);
  return a;
}


void fmat_mul_tlr (const float *left, const float *right,
                   int n, int m, int k, float *mout)
{

  float alpha = 1;
  float beta = 0;
  char transleft = 't';
  char transright = 't';

  sgemm_ (&transright, &transleft, &k, &n, &m,
          &alpha, right, &m, left, &n, &beta, mout, &k);
}


float *fmat_new_mul_tlr (const float *left, const float *right,
                         int n, int m, int k)
{
  float *a = fmat_new (k, n);
  fmat_mul_tlr (left, right, n, m, k, a);
  return a;
}


/*! @brief Multiply a matrix by a vector */
float * fmat_mul_fvec (const float * a, const float * v, int nrow, int ncol)
{
  int i;
  float * res = malloc (nrow * sizeof (*res));
  for (i = 0 ; i < nrow ; i++)
    res[i] = fvec_inner_product (a + i * ncol, v, ncol);

  return res;
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


/*---------------------------------------------------------------------------*/
/* Matrix manipulation functions                                             */
/*---------------------------------------------------------------------------*/

float *fmat_get_submatrix (const float *a, int ncola, int r1, int c1, int r2,
                           int c2)
{
  int i, j;
  int nrow = r2 - r1;
  int ncol = c2 - c1;
  float *b = fmat_new (nrow, ncol);

  for (i = r1; i < r2; i++)
    for (j = c1; j < c2; j++)
      b[(i - r1) * ncol + (j - c1)] = a[i * ncola + j];

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
float *random_orthogonal_basis (int d)
{
  int i;


  /* generate a Gaussian matrix */
  float *x = fmat_new_rand_gauss (d, d);

  float *tau = NEWA (float, d);

  {                             /* compute QR decomposition */

    /* query work size */
    float lwork_query;
    int lwork = -1;
    int info;
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

float *fmat_center_columns(int d,int n,float *v) {
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



float *fmat_covariance (int d, int n, const float *v, float *avg)
{
  
  float *sums = avg ? avg : fvec_new(d);
  long i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < d; j++)
      sums[j] += v[i * d + j];

  float *cov = fvec_new_nan (d * d);

  for (i = 0; i < d; i++)
    for (j = 0; j < d; j++)
      cov[i + j * d] = sums[i] * sums[j];

  if(avg)
    for(i=0;i<d;i++) avg[i]/=n;
  else
    free (sums);


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
    ssyrk_("L","N", &di, &ni, &alpha,v,&di,&beta,cov,&di);

    /* copy lower triangle to upper */

    for(i=0;i<d;i++)
      for(j=i+1;j<d;j++) 
        cov[i+j*d]=cov[j+i*d];

  }

  return cov;
}




float *fmat_covariance_thread (int d, int n, const float *v, float *avg, int nt)
{
  fprintf (stderr, "# function fmat_covariance_thread is currently not implemented\n");
  exit (1);
}




float *fmat_pca(int d,int n,const float *v) {

  float *cov=fmat_covariance(d,n,v,NULL);
  
  float *ret=fmat_pca_from_covariance(d,cov,NULL);
    
  free(cov);  
  
  return ret;
}


float *fmat_pca_from_covariance(int d,const float *cov,
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

float *fmat_pca_part_from_covariance(int d,int nv,const float *cov,
                                     float *singvals) {
  float *pcamat=fvec_new(d*nv);
  float *evals=singvals;

  if(!singvals) evals=fvec_new(nv);

  if(eigs_sym_part(d,cov,nv,evals,pcamat)!=0) {
    free(pcamat);
    pcamat=NULL;
    goto error;
  }
  eigs_reorder(d,evals,pcamat,1); /* 1 = descending */

 error:
  if(!singvals) free(evals);

  return pcamat;
  
}


#if 0


/* to accumulate covariances */
typedef struct {
  int n,d;  
  float *cov; /*< (d,d) covariance matrix */
  float *sum; /*< (d) sum of points */
} covariance_accu_t;

void covariance_accu_init(covariance_accu_t *ca, int d) {
  ca->cov=fvec_new_0(d*d);
  ca->sum=fvec_new_0(d);
}

void covariance_accu_add(covariance_accu_t *ca,int n,const float* points) {
  int i,d=ca->d;
  
  for(i=0;i<n;i++) fvec_add(ca->sum,points+i*d,d);
  
  float alpha = 1.0, beta = 0;
  FINTEGER di=ca->d;  
  FINTEGER ni=n;

  sgemm_ ("N", "T", &di, &di, &ni, &alpha, points, &di, points, &di, &beta, ca->cov, &di);
  
}

void covariance_accu_merge(covariance_accu_t *ca,const covariance_accu_t * other) {
  int d=ca->d;
  fvec_add(ca->cov,other->cov,d*d);
  fvec_add(ca->sum,other->sum,d);
}

void covariance_accu_to_cov(covariance_accu_t *ca) {
  

}

void covariance_accu_delete(covariance_accu_t *ca) {
  free(ca->sum);
  free(ca->cov);
}





#endif
























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





#ifdef HAVE_ARPACK

typedef FINTEGER integer;

extern void ssaupd_ (integer *ido,const char*bmat,integer *n, const char*which,integer *nev,
                     float* tol, float*resid, integer *ncv, float *v, integer *ldv, 
                     integer *iparam, integer * ipntr, float *workd, float *workl, 
                     integer *lworkl, integer *info );

typedef FINTEGER logical;

extern void sseupd_ (logical *rvec, const char *howmny, logical *select, float *d    ,
                     float *z     ,integer *ldz   , float *sigma , const char*bmat,
                     integer *n       , const char*which,integer *nev, float* tol, 
                     float*resid, integer *ncv, float *v, integer *ldv, 
                     integer *iparam, integer * ipntr, float *workd, float *workl, 
                     integer *lworkl, integer *info );


#ifdef _OPENMP

#include <omp.h>

#define SET_NT  omp_set_num_threads(nt)  

#else 

#define SET_NT

/* #pragma's will be ignored */

#endif

/* multithreaded matrix-vector multiply */

/* m=nb rows, n=nb cols */
void fmat_mul_v(int m,int n,const float*a,int lda,
                 const float *x,
                 float *y,int nt) {
  int i;
  SET_NT;
  
#pragma omp parallel 
  {
#pragma omp for 
    for(i=0;i<nt;i++) {
      int i0=i*(long)m/nt;
      int i1=(i+1)*(long)m/nt;
      int m1=i1-i0;
      float one=1.0,zero=0.0;
      int ione=1;
      // printf("%d %d\n",i,m1);
      sgemv_("Trans",&n,&m1,&one,
             a+lda*(long)i0,&lda,x,&ione,&zero,y+i0,&ione);
      
    }
  }   

}

void fmat_mul_tv(int m,int n,const float*a,int lda,
                 const float *x,
                 float *y,int nt) {
  int i,j;
  
  SET_NT;

  float *ybuf=malloc(sizeof(float)*nt*m);

/*  printf("x=[");
  for(j=0;j<n;j++) printf("%g ",x[j]);
  printf("]\n");  
*/
  if(nt>n) nt=n;
#pragma omp parallel 
  {
#pragma omp for 
    for(i=0;i<nt;i++) {
      int i0=i*(long)n/nt;
      int i1=(i+1)*(long)n/nt;
      int n1=i1-i0;
      float one=1.0,zero=0.0;
      int ione=1;
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
/*
  printf("y=[");
  for(j=0;j<m;j++) printf("%g ",y[j]);
  printf("]\n");
*/
}



int partial_svd(int m,int n,const float *a,
                int nev,
                float *sout,
                float *uout, float *vout,
                int nt) {
  
  int ncv=2*nev;  /* should be enough (see remark 4 of ssaupd doc) */

  float *s=NEWA(float,ncv*2);
  int i,j;
  const char *bmat="I",*which="LM";
  float tol=0;
  int info=0;
  int ido=0;
  int lworkl = ncv*(ncv+8);
  float *resid=NEWA(float,n),*workd=NEWA(float,3*n),*workl=NEWA(float,lworkl);
  float *ax=NEWA(float,m);
  float *v=NEWA(float,n*(long)ncv);
  int *iparam=NEWA(int,11),*ipntr=NEWA(int,11);

  iparam[0]=1;
  iparam[2]=n;
  iparam[6]=1;

  double dt0=0,dt1=0;
  
  i=0;
  for(;;) {

    /*     double t0=getmillisecs(); */

    ssaupd_(&ido, bmat, &n, which, &nev, 
            &tol, resid, &ncv, v, &n, 
            iparam, ipntr, workd, workl, &lworkl,
            &info);

    /*     double t1=getmillisecs(); */
    
    if(ido==-1 || ido==1) {
      
      
      float *x=workd+ipntr[0]-1;
      float *w=ax;
      float *y=workd+ipntr[1]-1;

      if(0) {
        float zero=0,one=1;
        int ione=1;
        
        sgemv_("Trans",&n,&m,&one,a,&n,x,&ione,&zero,w,&ione);
                
        sgemv_("No",&n,&m,&one,a,&n,w,&ione,&zero,y,&ione);
      } else {

        fmat_mul_v(m,n,a,n,x,w,nt);

        fmat_mul_tv(n,m,a,n,w,y,nt);
        
      } 

    } else break;   

    /*     double t2=getmillisecs(); */
    /*     dt0+=t1-t0; */
    /*     dt1+=t2-t1; */
    /*     printf("ssaupd eval %d (dt0=%.3f dt1=%.3f) \r",i++,dt0,dt1); fflush(stdout); */

    printf("ssaupd eval %d\r",i++); fflush(stdout);
  }
  
  printf("\n ssaupd -> sseupd\n");

  if(info<0) {
    printf("partial_pca: ssaupd_ error info=%d\n",info);
    return info;
  } else {
    logical *select=NEWA(logical,ncv);
    int ierr;
    logical rvec=1;
    float sigma;
    sseupd_(&rvec,"All",select,s,
            v,&n, &sigma, bmat, &n, which, &nev, 
            &tol, resid, &ncv, v, &n, 
            iparam, ipntr, workd, workl, &lworkl,&ierr);

    if(ierr!=0) {
      printf("partial_pca: sseupd_ error: %d\n",ierr);
      return ierr;     
    }
    int nconv=iparam[4];

    if(nconv<nev) {
      printf("partial_pca: nev=%d, nconv=%d, increase ncv=%d\n",nev,nconv,ncv);
      return 0xdeadbeef;     
    }

    for(j=0;j<nconv;j++) {
      s[j]=sqrt(s[j]);
    } 

    /*
    printf("nconv=%d s=[",nconv);
    for(i=0;i<nev;i++)  printf("%g ",s[i]);
    printf("]\n");
    */
    free(select); 
  }

  free(resid); 
  free(workl);
  free(workd);
  free(ax);
  free(iparam);
  free(ipntr);

  /* order v by s */
  
  int *perm=NEWA(int,nev);
  fvec_sort_index(s,nev,perm); 
  
  if(vout) 
    for(i=0;i<nev;i++) 
      memcpy(vout+n*i,v+n*perm[nev-1-i],sizeof(float)*n);

  if(sout) 
    for(i=0;i<nev;i++) 
      sout[i]=s[perm[nev-1-i]];

  if(uout) { /* compute u */
    for(i=0;i<nev;i++) {
      float *u=uout+m*i;
      fmat_mul_v(m,n,a,n,v+n*perm[nev-1-i],u,nt);
      fvec_normalize(u,m,2);
    }               
  }

  free(perm); 
  free(v);
  free(s);

  return 0;
} 


#else

int partial_svd(int m,int n,const float *a,
                int nev,
                float *sout,
                float *uout, float *vout,
                int nt) {
  fprintf(stderr,"partial_svd: ERROR bigimbaz not compiled with arpack\n");
  return -1;
}

#endif

int partial_pca(int m,int n,const float *a,
                int nev,float *vout) {
  if(!(nev<=m && nev<=n)) {
    fprintf(stderr,"partial_pca: asking for too too many eigenvalues (%d) wrt %d*%d data\n",nev,m,n);
    return -1;
  }

  if(m>=n) 
    return partial_svd(m,n,a,nev,NULL,NULL,vout,count_cpu());
  else {
    fprintf(stderr,"partial_pca: warn fewer learning points (%d) than dimensions (%d): transposing\n",m,n);
    float *at=NEWA(float,m*n);
    
    int i,j;
    for(i=0;i<m;i++) for(j=0;j<n;j++) 
      at[i+j*m]=a[i*n+j];

    int ret=partial_svd(n,m,a,nev,NULL,vout,NULL,count_cpu());

    free(at);
    return ret;
  }
}

