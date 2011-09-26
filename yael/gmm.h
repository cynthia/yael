#ifndef GMM_H_INCLUDED
#define GMM_H_INCLUDED



/*---------------------------------------------------------------------------*/
/*! @addtogroup gmm
 *  @{
 */

/*! @defgroup gmm 
 *
 * Gaussian Mixture Model implementation and Fisher Kernels, as
 * defined in: F. Perronnin and C. Dance, Fisher kernels on visual
 * vocabularies for image categorization, CVPR 2006.
 */ 

/*! Gaussian Mixture Model (GMM) implementation */
typedef struct gmm_s {
  int d;          /*!< vector dimension */
  int k;          /*!< number of mixtures */
  float * w;      /*!< weights of the mixture elements (size k) */
  float * mu;     /*!< centroids (d-by-k) */
  float * sigma;  /*!< diagonal of the covariance matrix (d-by-k) */
} gmm_t;

/*! during computation of probabilities: take weights into account */
#define GMM_FLAGS_W 1

/*! do not normalize probabilities (bad!) */
#define GMM_FLAGS_NO_NORM 2

/*! during learning: compute a single value for the sigma diagonal */
#define GMM_FLAGS_1SIGMA 4

/*! during gmm learning: just do a kmeans */
#define GMM_FLAGS_PURE_KMEANS 32

/*! dp_dlambda: include mu and sigma in derivatives  */
#define GMM_FLAGS_SIGMA 8
#define GMM_FLAGS_MU 16


/*! Estimate the Gaussian mixture in two stages: 
 * - standard kmeans, 
 * - EM to find parameters.
 * 
 * @param d,k    see gmm_t structure
 * @param n      nb of learning points
 * @param niter  nb of iterations (same for both stages)
 * @param v(d,n) input vectors
 * @param nt     nb of threads 
 * @param seed   usedd by kmeans to initialize random number generator
 * @param nredo  number of "redo"  (launch several kmeans)
 * @param flags  see GMM_* flags. Typically, use GMM_FLAGS_W (to estimate weights).
 */
gmm_t * gmm_learn (int d, int n, int k, int niter,
                   const float * v, int nt, int seed, int nredo,
                   int flags);


/*! Describe to stdout */
void gmm_print(const gmm_t *g);

/*! free a GMM structure */
void gmm_delete (gmm_t * g);


/*!  compute probabilities of centroids for each vector p(c_i|x).
 *
 * @param v(d,n)  v(:,i) is c_i
 * @param p(k,n)  output probability values
 */
void gmm_compute_p (int n, const float * v, 
                    const gmm_t * g, 
                    float * p,
                    int flags);

/*! Fisher descriptor. 
 *
 * Compute 
 *
 *   nabla_lambda p(x, lambda) 
 *
 * where 
 *
 *   lambda = (w, mu, sqrt(sigma))
 *
 * @param v(d,n)           vectors where to compute descriptor 
 * @param flags combination of GMM_FLAGS_*. Typically, use
 *                         yael.GMM_FLAGS_MU (only interested in the derivative wrt mu)
 * @param fisher_vector_out(dd) output descriptor. The output descriptor size dd is given by gmm_fisher_sizeof(flags)
 *
 */
void gmm_fisher (int n, const float *v, const gmm_t * g, 
                 int flags, float * fisher_vector_out);

size_t gmm_fisher_sizeof (const gmm_t * g, int flags);



/*! write the GMM structure parameter into a file */
void gmm_write(const gmm_t *g, FILE *f); 

/*! read the GMM from a file */
gmm_t * gmm_read (FILE *f); 


/*! Threaded version of gmm_compute_p */
void gmm_compute_p_thread (int n, const float * v, 
                           const gmm_t * g, 
                           float * p, 
                           int flags,
                           int n_thread);


/*! @} */
#endif
