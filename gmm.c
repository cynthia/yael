#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>


#include "vector.h"
#include "matrix.h"
#include "kmeans.h"
#include "nn.h"
#include "gmm.h"
#include "sorting.h"

#include <sys/time.h>




static double getmillisecs() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec*1e3 +tv.tv_usec*1e-3;
}


/* Estimation of a Gaussian mixture (diagonal covariance matrix)
     k              number of mixture
     d              vector dimension
     g              gmm structure, namely:
     g->w   (k)     weights of the mixture
     g->mu  (k*d)   the centroids (mean of the mixture)
     g->sigma (k*d) the  diagonal of the covariance matrix
*/   


/* Initialize a new GMM structure */
static gmm_t * gmm_new (int d, int k)
{
  gmm_t * g = (gmm_t *) malloc (sizeof(*g));
  g->d=d;
  g->k=k;
  g->w = fvec_new (k);
  g->mu = fvec_new (k * d);
  g->sigma = fvec_new (k * d);

  return g;
}

/* Free an existing GMM structure */
void gmm_delete (gmm_t * g)
{
  free(g->w);
  free(g->mu);
  free(g->sigma);
  free(g);
}




#define real float
#define integer FINTEGER

int sgemm_ (char *transa, char *transb, integer * m, integer *
            n, integer * k, real * alpha, const real * a, integer * lda,
            const real * b, integer * ldb, real * beta, real * c__,
            integer * ldc);


int sgemv_(char *trans, integer *m, integer *n, real *alpha, 
           const real *a, integer *lda, const real *x, integer *incx, real *beta, real *y, 
           integer *incy);

#undef real
#undef integer

/* compute sum and diagonal of covariance matrix of a set of points (v) weighted by probabilities (p) */
static void compute_sum_dcov(int n,int k,int d,
                             const float *v,const float *mu_old,const float *p,
                             float *mu,float *sigma,float *w) {
  long i,j,l;

  for (j = 0 ; j < k ; j++) {
    double dtmp = 0;
    for (i = 0 ; i < n ; i++) dtmp += p[i * k + j];
    w[j] = dtmp;
  }

  float zero=0,one=1;
  sgemm_("Not transposed","Transposed",&d,&k,&n,&one,v,&d,p,&k,&zero,mu,&d);
  
  float *v2=fvec_new_cpy(v,n*(long)d);
  fvec_sqr(v2,n*(long)d);
  
  sgemm_("Not transposed","Transposed",&d,&k,&n,&one,v2,&d,p,&k,&zero,sigma,&d);
  free(v2);
  
  for (j = 0 ; j < k ; j++) {
    float *sigma_j=sigma+j*d;
    const float *mu_old_j=mu_old+j*d;
    const float *mu_j=mu+j*d;
    for(l=0;l<d;l++) {
      sigma_j[l]+=mu_old_j[l]*(mu_old_j[l]*w[j]-2*mu_j[l]);
    }
  }    

}

/* see threaded version below */
static void compute_sum_dcov_thread(int n,int k,int d,
                                    const float *v,const float *mu_old,const float *p,
                                    float *mu,float *sigma,float *w,
                                    int n_thread);


static float min_sigma=1e-10;

/* estimate the GMM parameters */
static void gmm_compute_params (int n, const float * v, const float * p, 
                                gmm_t * g,
                                int flags,                         
                                int n_thread)
{
  long i, j;

  long d=g->d, k=g->k;
  float * vtmp = fvec_new (d);
  float * mu_old = fvec_new_cpy (g->mu, k * d);
  float * w_old = fvec_new_cpy (g->w, k);

  fvec_0 (g->w, k);
  fvec_0 (g->mu, k * d);
  fvec_0 (g->sigma, k * d);

  if(0) {
    /* slow and simple */
    for (j = 0 ; j < k ; j++) {
      double dtmp = 0;
      for (i = 0 ; i < n ; i++) {
        /* contribution to the gaussian weight */
        dtmp += p[i * k + j];
        /* contribution to mu */
        
        fvec_cpy (vtmp, v + i * d, d);
        fvec_mul_by (vtmp, d, p[i * k + j]);
        fvec_add (g->mu + j * d, vtmp, d);
        
        /* contribution to the variance */
        
        fvec_cpy (vtmp, v + i * d, d);
        fvec_sub (vtmp, mu_old + j * d, d);
        fvec_sqr (vtmp, d);
        fvec_mul_by (vtmp, d, p[i * k + j]);
        fvec_add (g->sigma + j * d, vtmp, d);
        
      }
      g->w[j] = dtmp;
    }

  } else {
    /* fast and complicated */

    if(n_thread<=1) 
      compute_sum_dcov(n,k,d,v,mu_old,p,g->mu,g->sigma,g->w);
    else
      compute_sum_dcov_thread(n,k,d,v,mu_old,p,g->mu,g->sigma,g->w,n_thread);
  }

  if(flags & GMM_FLAGS_1SIGMA) {
    for (j = 0 ; j < k ; j++) {
      float *sigma_j=g->sigma+j*d;
      double var=fvec_sum(sigma_j,d)/d;
      fvec_set(sigma_j,d,var);
    }
  } else if(0) { /* grouped sigmas, not useful */
    int group_sigma=2;
    int *perm=ivec_new(d);
    for (j = 0 ; j < k ; j++) {
      float *sigma_j=g->sigma+j*d;
      fvec_sort_index (sigma_j,d,perm);
      int group_size=d>>(group_sigma-2),gbegin,l;
      for(gbegin=0;gbegin<d;gbegin+=group_size) {
        double var=0;
        for(l=0;l<group_size;l++) 
          var+=sigma_j[perm[gbegin+l]];
        var/=group_size;
        for(l=0;l<group_size;l++) 
          sigma_j[perm[gbegin+l]]=var;
      }     
      
    } 
    free(perm);
  }


  long nz=0;
  for(i=0;i<k*d;i++) 
    if(fabs(g->sigma[i])<min_sigma) {
      g->sigma[i]=min_sigma;
      nz++;
    }

  if(nz) printf("WARN %ld sigma diagonals are too small (set to %g)\n",nz,min_sigma);

  for (j = 0 ; j < k ; j++) {
    fvec_div_by (g->mu + j * d, d, g->w[j]);
    fvec_div_by (g->sigma + j * d, d, g->w[j]);
  }

  //  fvec_mul_by (g->sigma, d * k, 10);


  fvec_normalize (g->w, k, 1);

  printf ("w = ");
  fvec_print (g->w, k);
  double imfac = k * fvec_sum_sqr (g->w, k);
  printf (" imfac = %.3f\n", imfac);

  //  printf ("mu = \n");
  //  fmat_print (g->mu, k, d);

  //  printf ("sigma = \n");
  //  fmat_print (g->sigma, k, d);

  free (vtmp);
  free (w_old);
  free (mu_old);
}



double static inline sqr (double x)
{
  return x * x;
}


/* support function to compute log(a+b) from log(a) and log(b) 
   without loss of precision */
static double log_sum(double log_a, double log_b)
{
  double log_s;
  if (log_a < log_b)
    log_s=log_b + log (1 + exp(log_a - log_b));
  else
    log_s=log_a + log (1 + exp(log_b - log_a));
  return log_s;
}

#define CHECKFINITE(a) if(!finite(a)) {fprintf(stderr,"!!!! gmm_compute_p: not finite " #a "=%g at line %d\n",a,__LINE__); }; 


static void compute_mahalanobis_sqr(int n,long k,long d,
                                    const float *mu,const float *sigma,
                                    const float *v,
                                    float *p) {
  FINTEGER di=d,ki=k; /* for blas functions */
  long i, j, l;
    
  float *mu2_sums=fvec_new(k);
  
  for (j = 0 ; j < k ; j++) {
    double dtmp = 0;
    for (l = 0 ; l < d ; l++) 
      dtmp += sqr(mu[j * d + l]) / sigma[j * d + l];      
    mu2_sums[j]=dtmp;
  }
  
  for (i = 0 ; i < n ; i++) 
    for (j = 0 ; j < k ; j++) 
      p[i * k + j]=mu2_sums[j];
  
  free(mu2_sums);
  
  float *v2=fvec_new(d*n);
  for (i = 0 ; i < n*d ; i++) 
    v2[i]=v[i]*v[i];
  
  float *inv_sigma=fvec_new(k*d);
  for (i = 0 ; i < k*d ; i++) 
    inv_sigma[i]=1.0/sigma[i];
  
  float one=1;
  
  sgemm_("Transposed","Not transposed",&ki,&n,&di,&one,inv_sigma,&di,v2,&di,&one,p,&ki);
  
  free(v2);
  
  float *mu_sigma=inv_sigma;
  for (i = 0 ; i < k*d ; i++) 
    mu_sigma[i]=mu[i]/sigma[i];
  
  float minus_two=-2;
  
  sgemm_("Transposed","Not transposed",&ki,&n,&di,&minus_two,mu_sigma,&di,v,&di,&one,p,&ki);  
  
  free(mu_sigma);      

}

/* compute p(ci|x). Warning: also update det */

void gmm_compute_p (int n, const float * v, 
                    const gmm_t * g, 
                    float * p,
                    int flags)
{
  if(n==0) return; /* sgemm doesn't like empty matrices */

  long i, j, l;
  double dtmp;
  long d=g->d, k=g->k;


  float * logdetnr = fvec_new(k);

  for (j = 0 ; j < k ; j++) {
    logdetnr[j] = -d / 2.0 * log (2 * M_PI);
    for (i = 0 ; i < d ; i++)
      logdetnr[j] -= 0.5 * log (g->sigma[j * d + i]);
  }

  /* compute all probabilities in log domain */

  /* compute squared Mahalanobis distances (result in p) */

  if(0) { /* simple & slow */
    for (i = 0 ; i < n ; i++) {
      for (j = 0 ; j < k ; j++) {
        dtmp = 0;
        for (l = 0 ; l < d ; l++) {
          dtmp += sqr (v[i * d + l] - g->mu[j * d + l]) / g->sigma[j * d + l];
        }
        p[i * k + j] = dtmp;
      }
    }
  } else { /* complicated & fast */
    compute_mahalanobis_sqr(n,k,d,g->mu,g->sigma,v,p); 
  }

  /* convert distances to probabilities, staying in the log domain as
     until the very end */
  for (i = 0 ; i < n ; i++) {

    for (j = 0 ; j < k ; j++) {
      p[i * k + j] = logdetnr[j] - 0.5 * p[i * k + j];
      CHECKFINITE(p[i * k + j]);
    }

    //    printf ("p[%d] = ", i);
    //    fvec_print (p + i * k, k);

    /* at this point, we have p(x|ci) -> we want p(ci|x) */
    

    if(flags & GMM_FLAGS_NO_NORM) {     /* compute the normalization factor */

      dtmp=0;

    } else {

      dtmp = p[i * k + 0];
      
      if(flags & GMM_FLAGS_W) 
        dtmp+=log(g->w[0]);

      for (j = 1 ; j < k ; j++) {
        double log_p=p[i * k + j];

        if(flags & GMM_FLAGS_W) 
          log_p+=log(g->w[j]);

        dtmp = log_sum (dtmp, log_p);
      }

      /* now dtmp contains the log of sums */
    } 

    for (j = 0 ; j < k ; j++) {
      double log_norm=0;

      if(flags & GMM_FLAGS_W)
        log_norm=log(g->w[j])-dtmp;
      else
        log_norm=-dtmp;

      p[i * k + j] = exp (p[i * k + j] + log_norm);
      CHECKFINITE(p[i * k + j]);
    }

    //    printf ("p[%d] = ", i);
    //    fvec_print (p + i * k, k);
  }


}



void gmm_handle_empty(int n, const float *v, gmm_t *g, float *p) {
  long d=g->d, k=g->k;
  
  long nz=fvec_count_occurrences(p,k*n,0);
  printf("nb of 0 probabilities: %ld / (%ld*%d) = %.1f %%\n",
         nz,k,n,nz*100.0/(k*n));         

  int i,j;
  float *w=fvec_new_0(k);
  for (i = 0 ; i < n ; i++) 
    for (j = 0 ; j < k ; j++) 
      w[j]+=p[j+i*k];
      
  for (j = 0 ; j < k ; j++) if(w[j]==0) {
    printf("center %d is empty....",j);
    fflush(stdout);
    int j2;

    do {
      j2=random()%k;
    } while(w[j2]==0);
    
    /* dimension to split: that with highest variance */
    int split_dim=fvec_max_index(g->sigma+d*j2,d);

    /* transfer half(?) of the points from j2 -> j */
    int nt=0,nnz=0;
    for(i=0;i<n;i++) if(p[j2+i*k]>0) { 
      nnz++;
      if(v[j*d+split_dim]<g->mu[split_dim]) {
        p[j+i*k]=p[j2+i*k];
        p[j2+i*k]=0;
        nt++;
      }
    }

    printf("split %d at dim %d (transferred %d/%d pts)\n",j2,split_dim,nt,nnz);    

    w[j2]=0; /* avoid further splits */
  }

  free(w);

}


gmm_t * gmm_learn (int di, int ni, int ki, int niter, 
	   const float * v, int nt, int seed, int nredo,
	   int flags)
{
  long d=di,k=ki,n=ni;

  int iter, iter_tot = 0;
  double old_key, key = 666;

  niter = (niter == 0 ? 10000 : niter);

  int * assign = ivec_new (n);

  /* the GMM parameters */
  float * p = fvec_new_0 (n * k);      /* p(ci|x) for all i */
  gmm_t * g = gmm_new (d, k);

  /* initialize the GMM: k-means + variance estimation */
  int * nassign = ivec_new (n);  /* not useful -> to be removed when debugged */
  float * dis = fvec_new (n);
  kmeans (d, n, k, niter, v, nt, seed, nredo, g->mu, dis, assign, nassign); 
  
  fflush (stderr);
  fprintf (stderr, "assign = ");
  ivec_print (nassign, k);
  fprintf (stderr, "\n");
  free (nassign);

  /* initialization of the GMM parameters assuming a diagonal matrix */
  fvec_set (g->w, k, 1.0 / k);
  double sig = fvec_sum (dis, n) / n;
  printf ("sigma at initialization = %.3f\n", sig);
  fvec_set (g->sigma, k * d, sig);
  free (dis);


  /* start the EM algorithm */
  fprintf (stdout, "<><><><> GMM  <><><><><>\n");
      
  if(flags & GMM_FLAGS_PURE_KMEANS) niter=0;

  for (iter = 1 ; iter <= niter ; iter++) {
    
    double t0=getmillisecs();
  
    gmm_compute_p_thread (n, v, g, p, flags, nt);
    fflush(stdout);

    gmm_handle_empty(n, v, g, p);
    
    /* printf("gmm_compute_p: %.3f ms\n",getmillisecs()-t0); */

    t0=getmillisecs();

    gmm_compute_params (n, v, p, g, flags, nt);
    fflush(stdout);

    /*    printf("gmm_compute_params: %.3f ms\n",getmillisecs()-t0); */

    iter_tot++;

    /* convergence reached -> leave */
    old_key = key;
    key = fvec_sum (g->mu, k * d);

    printf ("keys %5d: %.6f -> %.6f\n", iter, old_key, key);
    fflush(stdout);

    if (key == old_key)
      break;
  }
  fprintf (stderr, "\n");
  free(p);
  
  return g;
}

size_t gmm_dp_dlambda_sizeof(const gmm_t * g,int flags) {
  int sz=0;
  if(flags & GMM_FLAGS_W) sz+=g->k-1;
  if(flags & GMM_FLAGS_MU) sz+=g->d*g->k;
  if(flags & GMM_FLAGS_1SIGMA) sz+=g->k;
  if(flags & GMM_FLAGS_SIGMA) sz+=g->d*g->k;
  return sz;
}


void gmm_compute_dp_dlambda(int n, const float *v, const gmm_t * g, int flags, float *dp_dlambda) {
  long d=g->d, k=g->k;
  float *p=fvec_new(n*k);
  long i,j,l;
  long ii=0;

  gmm_compute_p(n,v,g,p,flags);

#define P(j,i) p[(i)*k+(j)]
#define V(i,l) v[(i)*d+(l)]
#define MU(j,l) g->mu[(j)*d+(l)]
#define SIGMA(j,l) g->sigma[(j)*d+(l)]

  if(flags & GMM_FLAGS_W) {

    for(j=1;j<k;j++) {
      double accu=0;
      
      for(i=0;i<n;i++) 
        accu+= P(j,i)/g->w[j] - P(0,i)/g->w[0];
      
      /* normalization */
      double f=n*(1/g->w[j]+1/g->w[0]);
      
      dp_dlambda[ii++]=accu/sqrt(f);
    }
  } 
  
  if(flags & GMM_FLAGS_MU) {

    for(j=0;j<k;j++) {
      for(l=0;l<d;l++) {
        double accu=0;
        
        for(i=0;i<n;i++) 
          accu += P(j,i) * (V(i,l)-MU(j,l)) / SIGMA(j,l);
        
        double f=flags & GMM_FLAGS_NO_NORM ? 1.0 : n*g->w[j]/SIGMA(j,l);
        
        dp_dlambda[ii++]=accu/sqrt(f);
                
      }
    }
  }


  if(flags & (GMM_FLAGS_SIGMA | GMM_FLAGS_1SIGMA)) {

    for(j=0;j<k;j++) {
      double accu2=0;
      for(l=0;l<d;l++) {
        double accu=0;
        
        for(i=0;i<n;i++) 
          accu += P(j,i) * (sqr(V(i,l)-MU(j,l)) / SIGMA(j,l) - 1) / sqrt(SIGMA(j,l));
        
        if(flags & GMM_FLAGS_SIGMA) {

          double f=flags & GMM_FLAGS_NO_NORM ? 1.0 : 2*n*g->w[j]/SIGMA(j,l);
          
          dp_dlambda[ii++]=accu/sqrt(f);
        } 
        accu2+=accu;        
      }

      if(flags & GMM_FLAGS_1SIGMA) {
        double f=flags & GMM_FLAGS_NO_NORM ? 1.0 : 2*d*n*g->w[j]/SIGMA(j,0);
        dp_dlambda[ii++]=accu2/sqrt(f);        
      }

    }  

  }
  
  assert(ii==gmm_dp_dlambda_sizeof(g,flags));
#undef P
#undef V
#undef MU
#undef SIGMA
  free(p);
}



void gmm_print(const gmm_t *g) {
  printf("gmm (%d gaussians in %d dim)=[\n",g->k,g->d);
  int i,j;
  for(i=0;i<g->k;i++) {
    printf("   w=%g, mu=[",g->w[i]);
    for(j=0;j<g->d;j++) printf("%g ",g->mu[i*g->d+j]);
    printf("], sigma=diag([");
    for(j=0;j<g->d;j++) printf("%g ",g->sigma[i*g->d+j]);    
    printf("])\n");
  }
  printf("]\n");
}

#define WRITEANDCHECK(a,n) if(fwrite(a,sizeof(*a),n,f)!=n) {perror("gmm_write"); abort(); }


void gmm_write(const gmm_t *g,FILE *f) {
  
  WRITEANDCHECK(&g->d,1);
  WRITEANDCHECK(&g->k,1);
  WRITEANDCHECK(g->w,g->k);
  WRITEANDCHECK(g->mu,g->k*g->d);
  WRITEANDCHECK(g->sigma,g->k*g->d);
  
}

#define READANDCHECK(a,n) if(fread(a,sizeof(*a),n,f)!=n) {perror("gmm_read"); abort(); }


gmm_t * gmm_read(FILE *f) {
  int d,k;

  READANDCHECK(&d,1);
  READANDCHECK(&k,1);

  gmm_t *g=gmm_new(d,k);
  
  READANDCHECK(g->w,g->k);
  READANDCHECK(g->mu,g->k*g->d);
  READANDCHECK(g->sigma,g->k*g->d);
  
  return g;
}



/****************************************************************************************************************
 ***************** threaded versions */



typedef struct {
  long n;
  const float * v;
  const gmm_t * g;
  float * p;
  int do_norm;
  int n_thread;   
} compute_p_params_t;

/* n sliced */
static void compute_p_task_fun (void *arg, int tid, int i) {
  compute_p_params_t *t=arg;
  long n0=i*t->n/t->n_thread;
  long n1=(i+1)*t->n/t->n_thread;
  
  gmm_compute_p(n1-n0, t->v+t->g->d*n0, t->g, t->p+t->g->k*n0, t->do_norm);
}

void gmm_compute_p_thread (int n, const float * v, 
                           const gmm_t * g, 
                           float * p, 
                           int do_norm,
                           int n_thread) {
  compute_p_params_t t={n,v,g,p,do_norm,n_thread};
  compute_tasks(n_thread,n_thread,&compute_p_task_fun,&t);
}


typedef struct {
  long n,k,d;
  const float *v,*mu_old,*p;
  float *mu,*sigma,*w;
  int n_thread;
} compute_sum_dcov_t;

/* n sliced */
static void compute_sum_dcov_task_fun (void *arg, int tid, int i) {
  compute_sum_dcov_t *t=arg;
  long n0=i*t->n/t->n_thread;
  long n1=(i+1)*t->n/t->n_thread;
  
  compute_sum_dcov(n1-n0,t->k,t->d,t->v+t->d*n0,
                   t->mu_old,t->p+n0*t->k,
                   t->mu+i*t->d*t->k,
                   t->sigma+i*t->d*t->k,
                   t->w+t->k*i);

}



static void compute_sum_dcov_thread(int ni,int ki,int di,
                                    const float *v,const float *mu_old,const float *p,
                                    float *mu,float *sigma,float *w,
                                    int n_thread) {
  long n=ni,d=di,k=ki;

  compute_sum_dcov_t t={
    n,k,d,
    v,mu_old,p,
    fvec_new(n_thread*d*k), /* mu */
    fvec_new(n_thread*d*k), /* sigma */
    fvec_new(n_thread*k), /* w */
    n_thread
  };

  compute_tasks(n_thread,n_thread,&compute_sum_dcov_task_fun,&t);
  
  /* accumulate over n's */

  long i;
  fvec_cpy(mu,t.mu,k*d);
  fvec_cpy(sigma,t.sigma,k*d);
  fvec_cpy(w,t.w,k);
  for(i=1;i<n_thread;i++) {
    fvec_add(mu,t.mu+i*d*k,d*k);
    fvec_add(sigma,t.sigma+i*d*k,d*k);    
    fvec_add(w,t.w+i*k,k);    
  }
  free(t.mu);
  free(t.sigma);
  free(t.w);
}




/****************************************************************************************************************
 ***************** Deprectated: computes a VLAD descriptor from a GMM structure. See vlad.[ch] for the current version
 */



void gmm_compute_fisher_simple(int n, const float *v, const gmm_t * g, int flags, float *desc) {
  int i,j,l;
  int k=g->k,d=g->d;


  if(flags<11 || flags>=13) {
    int *assign=ivec_new(n);

    nn(n,k,d,g->mu,v,assign,NULL,NULL);    
    
    if(flags==6 || flags==7) {
      int n_quantile=flags==6 ? 3 : 1;
      fvec_0(desc,k*d*n_quantile);
      int *perm=ivec_new(n);
      float *tab=fvec_new(n);
      ivec_sort_index(assign,n,perm);
      int i0=0;
      for(i=0;i<k;i++) {
        int i1=i0;
        while(i1<n && assign[perm[i1]]==i) i1++;
        
        if(i1==i0) continue;
        
        for(j=0;j<d;j++) {        
          for(l=i0;l<i1;l++)
            tab[l-i0]=v[perm[l]*d+j];
          int ni=i1-i0;
          fvec_sort(tab,ni);
          for(l=0;l<n_quantile;l++) 
            desc[(i*d+j)*n_quantile+l]=(tab[(l*ni+ni/2)/n_quantile]-g->mu[i*d+j])*ni;
        }
        
        i0=i1;
      }
      free(perm);
      free(tab);
    } else if(flags==5) {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j];
      }
      
    } else if(flags==8 || flags==9) {
      fvec_0(desc,k*d);
      
      float *u=fvec_new(d);
      
      for(i=0;i<n;i++) {
        fvec_cpy(u,v+i*d,d);
        fvec_sub(u,g->mu+assign[i]*d,d);
        float un=sqrt(fvec_norm2sqr(u,d));
        
        if(un==0) continue;
        if(flags==8) {        
          fvec_div_by(u,d,un);
        } else if(flags==9) {
          fvec_div_by(u,d,sqrt(un));
        }
        
        fvec_add(desc+assign[i]*d,u,d);
        
      }
      free(u);
      
    } else if(flags==10) {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j];
      }
      
      for(i=0;i<k;i++) 
        fvec_normalize(desc+i*d,d,2.0);   
      
    } else if(flags==13) {

      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=sqr(v[i*d+j]-g->mu[assign[i]*d+j]);
      }     

    } else if(flags==14) {
      float *avg=fvec_new_0(k*d);
      
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          avg[assign[i]*d+j]+=v[i*d+j]-g->mu[assign[i]*d+j];

      int *hist=ivec_new_histogram(k,assign,n);

      for(i=0;i<k;i++) 
        if(hist[i]>0) 
          for(j=0;j<d;j++) 
            avg[i*d+j]/=hist[i];

      free(hist);

      fvec_0(desc,k*d);
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=sqr(v[i*d+j]-g->mu[assign[i]*d+j]-avg[assign[i]*d+j]);
      
      fvec_sqrt(desc,k*d);
      
      free(avg);
    }  else if(flags==15) {
      fvec_0(desc,k*d*2);
      float *sum=desc;
      
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          sum[assign[i]*d+j]+=v[i*d+j]-g->mu[assign[i]*d+j];

      int *hist=ivec_new_histogram(k,assign,n);

      float *mom2=desc+k*d;

      for(i=0;i<n;i++) {
        int ai=assign[i];
        for(j=0;j<d;j++) 
          mom2[ai*d+j]+=sqr(v[i*d+j]-g->mu[ai*d+j]-sum[ai*d+j]/hist[ai]);
      }
      fvec_sqrt(mom2,k*d);
      free(hist);
    
      
    } else if(flags==17) {
      fvec_0(desc,k*d*2);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) {
          float diff=v[i*d+j]-g->mu[assign[i]*d+j];
          if(diff>0) 
            desc[assign[i]*d+j]+=diff;
          else 
            desc[assign[i]*d+j+k*d]-=diff;
        }
      }
  
    } else {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j]-g->mu[assign[i]*d+j];
      }
      
      
      if(flags==1) {
        int *hist=ivec_new_histogram(k,assign,n);
        /* printf("unbalance factor=%g\n",ivec_unbalanced_factor(hist,k)); */
        
        for(i=0;i<k;i++) 
          for(j=0;j<d;j++) 
            desc[i*d+j]/=hist[i];    
        
        free(hist);
      }
      
      if(flags==2) {
        for(i=0;i<k;i++) 
          fvec_normalize(desc+i*d,d,2.0);
      }
      
      if(flags==3 || flags==4) {
        for(i=0;i<k;i++) 
          for(j=0;j<d;j++) 
            desc[i*d+j]/=flags==3 ? sqrt(g->sigma[i*d+j]) : g->sigma[i*d+j];    
      }

      if(flags==16) {
        int *hist=ivec_new_histogram(k,assign,n);
        for(i=0;i<k;i++) if(hist[i]>0) {
          fvec_norm(desc+i*d,d,2);
          fvec_mul_by(desc+i*d,d,sqrt(hist[i]));
        }
        free(hist);
      }
   

    }
    free(assign);
  } else if(flags==11 || flags==12) {
    int a,ma=flags==11 ? 4 : 2;
    int *assign=ivec_new(n*ma);

    float *dists=knn(n,k,d,ma,g->mu,v,assign,NULL,NULL);    

    fvec_0(desc,k*d);

    for(i=0;i<n;i++) {
      for(j=0;j<d;j++) 
        for(a=0;a<ma;a++) 
          desc[assign[ma*i+a]*d+j]+=v[i*d+j]-g->mu[assign[ma*i+a]*d+j];
    } 
    
    free(dists);

    free(assign);
  }

}
