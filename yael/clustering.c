#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include "clustering.h"
#include "vector.h"
#include "nn.h"
#include "machinedeps.h"
#include "sorting.h" 
#include "kmeans.h"


#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)

static double sqr (double a)
{
  return a * a;
}


float *clustering_kmeans_assign_with_score (int n, int di,
                                            const float *points, int k, int nb_iter_max,                                            
                                            double normalize, 
                                            int n_thread,
                                            double *score, int **clust_assign_out)
{

  long d=di; /* to force 64-bit address computations */ 
  float *centroids = fvec_new (k * d);
  int *ca=clust_assign_out ? ivec_new(n) : NULL;
  int nredo=1;
  
  if(nb_iter_max/100000!=0) {
    nredo=nb_iter_max/100000;
    nb_iter_max=nb_iter_max % 100000;    
/*    printf("redo: %d iter: %d\n",nredo,nb_iter_max); */
  }   

  kmeans(di,n,k,nb_iter_max,points,n_thread,0,nredo,centroids,NULL,ca,NULL);

  if(clust_assign_out) *clust_assign_out=ca;
  
  return centroids;

}

float *clustering_kmeans_assign (int n, int d,
                                 const float *points, int k, int nb_iter_max,
                                 double normalize, int **clust_assign_out)
{
   return clustering_kmeans_assign_with_score (n, d, points, k, nb_iter_max,
                                               normalize, count_cpu(), NULL, clust_assign_out);
}

float *clustering_kmeans (int n, int d,
                          const float *points, int k, int nb_iter_max,
                          double normalize)
{

  int *clust_assign;

  float *centroids = clustering_kmeans_assign_with_score (n, d, points, k, nb_iter_max,
                                               normalize, count_cpu(), NULL, &clust_assign);
  free (clust_assign);

  return centroids;
}




static void clustering_cdm_factors(void (*find_knn)(void *arg,float *neigh_dists),
                                   void *find_arg,
                                   int n, 
                                   double alpha, int n_iter,
                                   int n_neigh,
                                   float *cdm_factors) {
  
  int i,j,k;

  /* reverse assignement: we want neighbours of the cluster centroids */ 
  int *centroid_neighs=ivec_new(n*n_neigh);

  /* average distance to neighbourhood */
  float *r=fvec_new(n),*f2=fvec_new(n),*neigh_dists=fvec_new(n*n_neigh);

  for(i=0;i<n;i++) 
    cdm_factors[i]=1.0;

  for(i=0;i<n_iter;i++) {
    
    /**** compute nearest neighbours using distance correction */
   
    /* square correction term because all distances are squared */

    /* 

    for(j=0;j<n;j++) f2[j]=sqr(cdm_factors[j]);    

    knn_full_thread (n,n,d,n_neigh,
                                   centroids,centroids,f2,
                                   centroid_neighs,neigh_dists,
                                   count_cpu(),NULL,NULL);

    */

    (*find_knn)(find_arg,neigh_dists);
    
    /* new distances */

    for(j=0;j<n;j++) {
      for(k=0;k<n_neigh;k++) {
        assert(finite(neigh_dists[j*n_neigh+k]));
        float d=neigh_dists[j*n_neigh+k];
        if(d<0) d=0;
        neigh_dists[j*n_neigh+k]=sqrt(d)*cdm_factors[j];
        assert(finite(neigh_dists[j*n_neigh+k]));
      }
    }

    /**** compute average distances to neighbourhood */

    for(j=0;j<n;j++) {
      r[j]=fvec_sum(neigh_dists+j*n_neigh,n_neigh)/n_neigh;
      assert(finite(r[j]));
    }
    /**** compute overall geometrical mean average */

    double sum=0;

    for(j=0;j<n;j++) sum+=log(r[j]);

    double r_bar=exp(sum/n);

    /**** compute distance correction term */ 
    
    for(j=0;j<n;j++) 
      cdm_factors[j]*=pow(r_bar/r[j],alpha);

  }

  free(r);
  free(f2);
  free(neigh_dists);  
  free(centroid_neighs);

}


typedef struct {
  int n,d,n_neigh;
  const float *centroids;
  float *cdm_factors;
} cmd_l2_args_t;


static void cdm_l2_find_knn(void *arg,float *dists) {
  cmd_l2_args_t *a=arg;
  int j;
  int n=a->n,k=a->n_neigh;
  const float *cdm_factors=a->cdm_factors;

  float *f2=fvec_new(n);
  
  for(j=0;j<n;j++) f2[j]=sqr(cdm_factors[j]);    

  int *centroid_neighs=ivec_new(k*n);

  knn_full_thread (2, n,n,a->d,k,
                                 a->centroids,a->centroids,f2,
                                 centroid_neighs,dists,
                                 count_cpu(),NULL,NULL);
  for(j=0;j<n*k;j++) {
    float d=dists[j];
    dists[j]=d<0 ? 0 : sqrt(d);
  }
  
  free(f2);
  free(centroid_neighs); 

}

void clustering_cdm_factors_l2(int n, int d,  
                               const float *centroids, 
                               double alpha, int n_iter,
                               int n_neigh,
                               float *cdm_factors) {
  cmd_l2_args_t args={n,d,n_neigh,centroids,cdm_factors};

  clustering_cdm_factors(&cdm_l2_find_knn,&args,n,
                         alpha, n_iter, n_neigh,
                         cdm_factors);
}

typedef struct {
  int n,n_neigh;
  const float *distances;
  float *cdm_factors;
} cmd_dists_args_t;


static void cdm_dists_find_knn(void *arg,float *dists) {
  cmd_dists_args_t *a=arg;
  int n=a->n,k=a->n_neigh;
  const float *cdm_factors=a->cdm_factors;
  int i,j;
  
  float *wd=fvec_new(n);

  for(i=0;i<n;i++) {

    const float *dl=a->distances+i*n;

    for(j=0;j<n;j++) 
      wd[j]=cdm_factors[j]*dl[j];

    fvec_quantile(wd,n,k);
    
    memcpy(dists+i*k,wd,k*sizeof(float));

  }
  
  free(wd);

}

void clustering_cdm_factors_dists(int n, 
                                  const float *distances, 
                                  double alpha, int n_iter,
                                  int n_neigh,
                                  float *cdm_factors) {

  cmd_dists_args_t args={n,n_neigh,distances,cdm_factors};

  clustering_cdm_factors(&cdm_dists_find_knn,&args,n,
                         alpha, n_iter, n_neigh,
                         cdm_factors);
  
}






static double getmillisecs ()
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}


double clustering_kmedoids_from_dists (int n, int n_med, int k,
                                       int nb_iter_max, const float *alldists,
                                       int **clust_assign_out,
                                       int **med_subset_out)
{
  int i, j;

  double t0 = getmillisecs ();

#define ALLDISTS(clustno,ptno) alldists[(ptno)+(clustno)*n]

  /* begin with a random subset of vectors */
  int *med_subset = ivec_new_random_perm (n_med);

  /* compute assignement and sse */
  double sse = 0;
  int *clust_assign = NEWA (int, n);

  for (i = 0; i < n; i++) {
    int clus = -1;
    double se = 1e50;
    for (j = 0; j < k; j++) {
      double new_se = ALLDISTS (med_subset[j], i);
      if (new_se < se) {
        se = new_se;
        clus = j;
      }
    }
    sse += se;
    clust_assign[i] = clus;
  }


  /* iterations */
  int *clust_assign_new = NEWA (int, n);
  int iter;
  int n_change = 0;

  for (iter = 0; iter < nb_iter_max; iter++) {

    /* select random swap */
    int to_p = random () % k;
    int from;
    do {                        /* steal from another cluster */
      from = random () % n_med;
    } while (clust_assign[from] == to_p);

    /* compute new sse and cluster assignement */
    double new_sse = 0;
    for (i = 0; i < n; i++) {

      int clust;
      double se;

      if (clust_assign[i] == to_p) {
        /* re-assign completely */
        clust = -1;
        se = 1e50;
        for (j = 0; j < k; j++) {
          int l = j == to_p ? from : med_subset[j];
          double d2 = ALLDISTS (l, i);
          if (d2 < se) {
            se = d2;
            clust = j;
          }
        }
      } else {
        clust = clust_assign[i];
        se = ALLDISTS (med_subset[clust], i);
        /* check if point would be assigned to new cluster */
        double d2 = ALLDISTS (from, i);
        if (d2 < se) {
          se = d2;
          clust = to_p;
        }
      }
      clust_assign_new[i] = clust;
      new_sse += se;
    }
    /*
       printf("replace centroid %d with %d: new sse=%g/%g, %s \n",
       perm[to_p],from,new_sse,sse,new_sse<sse ? "keep" : "reject");
     */

    /* check if improves sse */
    if (new_sse < sse) {
      n_change++;
      sse = new_sse;

      med_subset[to_p] = from;

      {                         /* swap current & new */
        int *tmp = clust_assign_new;
        clust_assign_new = clust_assign;
        clust_assign = tmp;
      }
    }

    if (iter % 1000 == 0) {     /* verify every 1000 iterations */

      printf ("iter %d n_change %d sse %g t=%.3f s\n",
              iter, n_change, sse, (getmillisecs () - t0) / 1000.0);

    }

  }


  free (clust_assign_new);

  if (clust_assign_out)
    *clust_assign_out = clust_assign;
  else
    free (clust_assign);

  if (med_subset_out)
    *med_subset_out = med_subset;
  else
    free (med_subset);

#undef ALLDISTS

  return sse;
}

float *clustering_kmedoids (int n, int d,
                            const float *points, int k, int nb_iter_max,
                            int **clust_assign_out)
{
  int i;

  /* arbitrarily chose medoids among the n_subset first points */
  int n_subset = 20 * k;
  if (n_subset > n)
    n_subset = n;
  n_subset = n;

  float *alldists = fvec_new (n_subset * n);

  compute_cross_distances (d, n, n_subset, points, points, alldists);

  int *clust_assign;
  int *med_subset;

  double sse =
      clustering_kmedoids_from_dists (n, n_subset, k, nb_iter_max, alldists,
                                      &clust_assign, &med_subset);

  free (alldists);

  float *centroids = fvec_new (k * d);

  for (i = 0; i < k; i++)
    memcpy (centroids + i * d, points + med_subset[i] * d,
            d * sizeof (float));

  {
    double verif_sse = 0;

    for (i = 0; i < n; i++)
      verif_sse +=
          fvec_distance_L2sqr (points + med_subset[clust_assign[i]] * d,
                        points + i * d, d);
    printf ("verif_sse=%g ?= %g\n", verif_sse, sse);
  }

  free (med_subset);

  if (clust_assign_out) {
    *clust_assign_out = clust_assign;
  } else
    free (clust_assign);

  return centroids;
}


