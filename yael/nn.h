/*---------------------------------------------------------------------------*/

#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

/*---------------------------------------------------------------------------*/
/* Nearest-neighbor (NN) functions                                           */
/*---------------------------------------------------------------------------*/


/*  Finds nearest neighbours of vectors in a base 
 * 
 *     distance_type: 2=L2 distance
 *     n:            number of vectors to assign 
 *     nb:           number of base vectors to assign to
 *     k:            number of neighbors to return
 *     v(n,d):       vectors
 *     b(nb,d):      base
 *     assign(n,k):  NNs of vector i are assign(i,0) to assign(i,k-1)
 *     b_weights(nb):  multiply squared distances by this for each base vector (may be NULL)
 *     dis(n,k):     distances of i to its NNs are dis(i,0) to dis(i,k-1)
 * 
 * all matrices are stored in row-major order. The declaration: 
 *
 *     a(n,m) 
 * 
 * means that element a(i,j) is accessed with a[i*m+j] and that
 *
 *     0<=i<n and 0<=j<m
 *
 * peek_fun needs not to be reentrant 
 */

void knn_full (int distance_type,
               int n, int nb, int d, int k,
               const float *b, const float *v,
               const float *b_weights,
               int *assign, float *dis,                                             
               void (*peek_fun) (void *arg,double frac),
               void *peek_arg);

/* multi-threaded version 
 * falls back on mono-thread version if task too small.
 */
void knn_full_thread (int distance_type,
                      int n, int nb, int d, int k,
                      const float *b, const float *v,
                      const float *b_weights,
                      int *assign, float *dis,
                      int n_thread,
                      void (*peek_fun) (void *arg,double frac),
                      void *peek_arg);




/* next functions are simplified calls of the previous */


/* single NN */
void nn (int n, int nb, int d, 
         const float *b, const float *v,
         int *assign,                                              
         void (*peek_fun) (void *arg,double frac),
         void *peek_arg);

void nn_thread (int n, int nb, int d, 
                const float *b, const float *v,
                int *assign,    
                int n_thread,
                void (*peek_fun) (void *arg,double frac),
                void *peek_arg);


/* functions return the set of distances to centroids (alloc'ed with malloc) */

float* knn (int n, int nb, int d, int k,
            const float *b, const float *v,
            int *assign,                                             
            void (*peek_fun) (void *arg,double frac),
            void *peek_arg);


float* knn_thread (int n, int nb, int d, int k,
                   const float *b, const float *v,
                   int *assign,    
                   int n_thread,
                   void (*peek_fun) (void *arg,double frac),
                   void *peek_arg);


/* computes a subset of L2 distances between b and v. 

   assign[i*k]..assign[i*k+k-1] , for i=0..n-1

  contains the indices in nb that must be reordered. On output, 

   dists[i*k]..dists[i*k+k-1] 
 
  contains the associated distances  
*/

void knn_reorder_shortlist(int n, int nb, int d, int k,
                           const float *b, const float *v,
                           int *assign,
                           float *dists);


/*****************************************************************
 *  Low-level function to compute distances between 2 sets of vectors 
 */

/* 
 *
 *  a is na*d \n
 *  b is nb*d \n
 *  dist2[i+na*j] = || a(i,:)-b(j,:) ||^2 \n
 *  uses BLAS if available
 */
void compute_cross_distances (int d, int na, int nb,
                              const float *a,
                              const float *b, float *dist2);

void compute_cross_distances_nonpacked (int d, int na, int nb,
                                        const float *a, int lda,
                                        const float *b, int ldb, 
                                        float *dist2, int ldd);

void compute_cross_distances_thread (int d, int na, int nb,
                                     const float *a,
                                     const float *b, float *dist2,
                                     int nt);





/*! @ alternative distances. 
 *
 * distance_type==1: L1, 
 * ==3: symmetric chi^2  
 * ==4: symmetric chi^2  with absolute value
 */

void compute_cross_distances_alt (int distance_type, int d, int na, int nb,
                                  const float *a,
                                  const float *b, float *dist2);


void compute_cross_distances_alt_thread (int distance_type,int d, int na, int nb,
                                         const float *a,
                                         const float *b, float *dist2,
                                         int nt);


/* version of compute_cross_distances where na==1 */
void compute_distances_1 (int d, int nb,
                          const float *a, 
                          const float *b,                         
                          float *dist2); 

void compute_distances_1_nonpacked (int d, int nb,
                                    const float *a, 
                                    const float *b, int ldb, 
                                    float *dist2);





/* compute_tasks creates nthread threads that call task_fun n times 
 * with arguments:
 *   arg=task_arg
 *   tid=identifier of the thread in 0..nthread-1
 *   i=call number in 0..n-1
 */

void compute_tasks (int n, int nthread,
                    void (*task_fun) (void *arg, int tid, int i),
                    void *task_arg);


#endif

/*---------------------------------------------------------------------------*/
