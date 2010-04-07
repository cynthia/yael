/*---------------------------------------------------------------------------*/

#ifndef __clustering_h
#define __clustering_h

/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*! @addtogroup utils
 *  @{
 */
/*---------------------------------------------------------------------------*/

/*! @brief k-means clustering. All float arrays are allocated by malloc_ffq */
float* clustering_kmeans (int n, int d,
                          const float *points, int k, int nb_iter_max, 
                          double normalize);

/*! @brief Same as k-means clustering, but generate in addition the assignment
 *  performed on the input set
 */
float* clustering_kmeans_assign (int n, int d,
				 const float *points, int k, int nb_iter_max, 
				 double normalize, 
				 int ** clust_assign_out);



float* clustering_kmeans_assign_with_score (int n, int d,
                                            const float *points, int k, int nb_iter_max,
                                            double normalize, int n_thread, double *score_out,
                                            int ** clust_assign_out);

/* dense k-medoids clustering */
float* clustering_kmedoids (int n, int d,
                            const float *points, int k, int nb_iter_max, 
                            int ** clust_assign_out);


/*
 * costs[i+n*j]=cost(i,j) is the cost of assigning point 0<=i<n to
 * medoid 0<=j<n_med
 * 
 * 0<=i<n --(clust_assign)--> 0<=m<k --(med_subset)--> 0<=j<n_med
 *
 * the function outputs a subset of the medoids in med_subset[m],
 * 0<=m<k, and a cluster assignment clust_assign[i] to this subset
 * such that
 * 
 *   sse = sum_i cost(i, med_subset[clust_assign[i]])
 *
 * is minimal. The function returns sse.
 *
 */

double clustering_kmedoids_from_dists(int n,int n_med,int k,int nb_iter_max,
                                      const float *costs,
                                      int **clust_assign_out,
                                      int **med_subset_out);




/* compute cdm factors on a clustered dataset */
void clustering_cdm_factors_l2(int n, int d,  
                               const float *centroids, 
                               double alpha, int n_iter,
                               int n_neigh,
                               float *cdm_factors);

/* compute cdm factors on a dense distance table */ 

void clustering_cdm_factors_dists(int n, 
                                  const float *distances, 
                                  double alpha, int n_iter,
                                  int n_neigh,
                                  float *cdm_factors);


/*---------------------------------------------------------------------------*/
/*! @} */
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
