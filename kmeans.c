#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "kmeans.h"
#include "nn.h"


static void nn_full (int d, int n, int nb, const float * v, const float *b, 
	      int nt, int * assign, float * dis)
{
  knn_full_thread (2, n, nb, d, 1, b, v, NULL, 
				 assign, dis, nt, NULL, NULL);
}


static void random_init(long d, int n, int k, const float * v, int * sel) {
  int *perm=ivec_new_random_perm(n);
  ivec_cpy(sel,perm,k);
  free(perm);
}

/* the kmeans++ initialization (see wikipedia) */
static void kmeanspp_init (long d, int n, int k, const float * v, int * sel)
{
  

  /* select the first centroid and set the others unitialized*/


  long i, j;
/*
  for(i=0;i<k;i++) sel[i]=i;

  return;
*/
  float * disbest = fvec_new_set (n, HUGE_VAL);
  float * distmp = fvec_new (n);

  sel[0] = lrand48() % k;

  for (i = 1 ; i < k ; i++) {
    int newsel = sel[i - 1];
    
    if(i%10==0) {
      printf("%d/%d\r",(int)i,k); fflush(stdout);
    }

    if(0) { /* simple and slow */
    
      for (j = 0 ; j < n ; j++) {
        distmp[j] = fvec_distance_L2sqr (v + d * j, v + d * newsel, d);
        if (distmp[j] < disbest[j])
          disbest[j] = distmp[j];
      }

    } else { /* complicated and fast */

/*
      compute_cross_distances(d,1,n,
                              v + d * newsel,
                              v,distmp);
*/
      compute_distances_1(d,n,
                          v + d * newsel,
                          v,distmp);
      
      for(j=0;j<n;j++) 
        if(distmp[j]<disbest[j]) disbest[j]=distmp[j];
    }

    
    /* convert the best distances to probabilities */
    memcpy (distmp, disbest, n * sizeof (*distmp));

    fvec_normalize (distmp, n, 1);

/*
    {
      int *perm=ivec_new(n);
      fvec_sort_index(distmp,n,perm);
      double accu=0;
      for(j=n*9/10;j<n;j++) 
        accu+=distmp[perm[j]];
      printf("it %d/%d p 10%%=%.3f\n",i,k,accu);      
      
      free(perm);      
    }
*/    
    double rd = drand48();
    
    for (j = 0 ; j < n - 1 ; j++) {
      rd -= distmp[j];
	if (rd < 0)
	  break;
    }
    
    sel[i] = j;
  }
  printf("\n");
  free (disbest);
  free (distmp);
}


float kmeans (int di, int n, int k, int niter, 
	      const float * v, int flags, int seed, int redo, 
	      float * centroids_out, float * dis_out, 
	      int * assign_out, int * nassign_out)
{
  long i, run, iter, iter_tot = 0, d=di;

  int nt=flags & 0xffff;

  int verbose = !(flags & KMEANS_QUIET);
  
  niter = (niter == 0 ? 10000 : niter);

  /* look at which variables have to be returned / allocated */
  int isout_centroids = (centroids_out == NULL ? 0 : 1);
  int isout_dis = (dis_out == NULL ? 0 : 1);
  int isout_assign = (assign_out == NULL ? 0 : 1);
  int isout_nassign = (nassign_out == NULL ? 0 : 1);

  /* the centroids */
  float * centroids = fvec_new (k * (size_t) d);

  /* store the distances from points to their nearest centroids */
  float * dis = fvec_new (n);

  /* the centroid indexes to which each vector is assigned */
  int * assign = ivec_new (n);

  /* the number of points assigned to a cluster */
  int * nassign = ivec_new (k);

  /* the total quantization error */
  double qerr, qerr_old, qerr_best = HUGE_VAL;

  /* for the initial configuration */
  int * selected = ivec_new (k);

  for (run = 0 ; run < redo ; run++) {
  do_redo: 
    
    if(verbose)
      fprintf (stderr, "<><><><> kmeans / run %d <><><><><>\nqerr: inf", (int)run);


    if( flags& KMEANS_INIT_RANDOM ) {
      random_init(d,n,k,v,selected);
    } else {

      int nsubset=n;
      
      if(n>k*8 && n>8192) {
        nsubset=k*8;
        if(verbose) 
          printf("Restricting k-means++ intialization to %d points\n",nsubset);
      }
      kmeanspp_init (d, nsubset, k, v, selected);
      
    }

     
    for (i = 0 ; i < k ; i++) 
      fvec_cpy (centroids + i * d, v + selected[i] * d, d);

    /* the quantization error */
    qerr = HUGE_VAL;

    for (iter = 1 ; iter <= niter ; iter++) {
      ivec_0 (nassign, k);
      iter_tot++;

      /* Assign point to cluster and count the cluster size */
      nn_full (d, n, k, v, centroids, nt, assign, dis);

      for (i = 0 ; i < n ; i++) 
	nassign[assign[i]]++;

      for (i = 0 ; i < k ; i++) {
        if(nassign[i]==0) {
          fprintf(stderr,"WARN nassign %d is 0, redoing!\n",(int)i);
          goto do_redo;
        }
      }
      

      /* update the centroids */

      if(flags & KMEANS_NORMALIZE_SOPHISTICATED) {
        
        float *norms=fvec_new(k);

        fvec_0 (centroids, d * k);
        
        for (i = 0 ; i < n ; i++) {
          fvec_add (centroids + assign[i] * d, v + i * d, d);
          norms[assign[i]]+=fvec_norm(v + i * d,d,2);
        }

        for (i = 0 ; i < k ; i++) {          
          fvec_normalize(centroids + i * d, d,2);
          fvec_mul_by (centroids + i * d, d, norms[i] / nassign[i]);
        }

        free(norms);

      } else {

        fvec_0 (centroids, d * k);
        
        for (i = 0 ; i < n ; i++)
          fvec_add (centroids + assign[i] * d, v + i * d, d);
        
        /* normalize */
        
        for (i = 0 ; i < k ; i++) {          
          fvec_mul_by (centroids + i * d, d, 1.0 / nassign[i]);
        }
        
        if(flags & KMEANS_NORMALIZE_CENTS) 
          for (i = 0 ; i < k ; i++) 
            fvec_normalize(centroids + i * d, d, 2.0);
      }

      assert(qerr>=0);

      /* compute the quantization error */
      qerr_old = qerr;
      qerr = fvec_sum (dis, n);

      if (qerr_old == qerr)
	break;

      if(verbose)
        fprintf (stderr, " -> %.3f", qerr / n);
    }
    if(verbose)      
      fprintf (stderr, "\n");

    /* If this run is the best encountered, save the results */
    if (qerr < qerr_best) {
      qerr_best = qerr;

      if (isout_centroids) 
	memcpy (centroids_out, centroids, k * d * sizeof (*centroids));
      if (isout_dis) 
	memcpy (dis_out, dis, n * sizeof (*dis));
      if (isout_assign) 
	memcpy (assign_out, assign, n * sizeof (*assign));
      if (isout_nassign)
	memcpy (nassign_out, nassign, k * sizeof (*nassign));
    }
  }

  if(verbose)
    fprintf (stderr, "Total number of iterations: %d\n", (int)iter_tot);

  /* printf("unbalanced factor of last iteration: %g\n",ivec_unbalanced_factor(nassign,k)); */
  
  /* free the variables that are not returned */
  free (selected);
  free (centroids);
  free (dis);
  free (assign);
  free (nassign);

  return qerr_best / n; 
}

