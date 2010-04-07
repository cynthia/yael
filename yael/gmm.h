/*
Copyright � INRIA 2010. 
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

#ifndef GMM_H_INCLUDED
#define GMM_H_INCLUDED


/********************************************************************************
 * Gaussian Mixture Model implementation and Fisher Kernels, main ref: 
 * 
 * @INPROCEEDINGS{PerDa06,
 *   author = {F Perronnin and C Dance},
 *   title = {Fisher kernels on visual vocabularies for image categorization},
 *   booktitle = {CVPR},
 *   year = {2006},
 * }
 * 
 */


/* GMM description */
typedef struct gmm_s {
  int d;          /* vector dimension */
  int k;          /* number of mixtures */
  float * w;      /* weights of the mixture elements (size k) */
  float * mu;     /* centroids (k-by-d) */
  float * sigma;  /* diagonal of the covariance matrix (k-by-d) */
} gmm_t;

/* during computation of probabilities: take weights into account */
#define GMM_FLAGS_W 1

/* do not normalize probabilities (bad!) */
#define GMM_FLAGS_NO_NORM 2

/* during learning: compute a single value for the sigma diagonal */
#define GMM_FLAGS_1SIGMA 4

/* during gmm learning: just do a kmeans */
#define GMM_FLAGS_PURE_KMEANS 32

/* dp_dlambda: include mu and sigma in derivatives  */
#define GMM_FLAGS_SIGMA 8
#define GMM_FLAGS_MU 16


/* 
 * Estimate the Gaussian mixture. Stages: 
 * 1. standard kmeans
 * 2. EM to find parameters.
 * 
 * d,k          see gmm_t structure
 * n            nb of learning points
 * niter        nb of iterations (the same for both stages)
 * v            v[i*d]..v[i*d+d-1] contains point i for i=0..n-1
 * nt           nb of threads 
 * seed, nredo  used by kmeans
 * flags        see above
 */
gmm_t * gmm_learn (int d, int n, int k, int niter,
                   const float * v, int nt, int seed, int nredo,
                   int flags);


/* Describe to stdout */
void gmm_print(const gmm_t *g);

/* Free a GMM structure */
void gmm_delete (gmm_t * g);


/* compute p(c_i|x).
 * v: n-by-d matrix with c_i values i=0..n-1
 * p: n-by-k matrix of probability values
 */
void gmm_compute_p (int n, const float * v, 
                    const gmm_t * g, 
                    float * p,
                    int flags);



/* Fisher descriptor: 
 *  compute \nabla_\lambda p(x,\lambda) 
 *  where \lambda = (w, mu, sqrt(sigma))
 *  size of db_dlambda depends on flags, call gmm_dp_dlambda_sizeof to find out
 */
void gmm_compute_dp_dlambda(int n, const float *v, const gmm_t * g, int flags, float *dp_dlambda);

size_t gmm_dp_dlambda_sizeof(const gmm_t * g,int flags);



/* I/O */
void gmm_write(const gmm_t *g,FILE *f); 
gmm_t * gmm_read(FILE *f); 


/* threaded version */
void gmm_compute_p_thread (int n, const float * v, 
                           const gmm_t * g, 
                           float * p, 
                           int flags,
                           int n_thread);


/* deprectated implementation of the VLAD descriptor. flags==0 gives
 * the standard VLAD, flags==15 gives the one with sigma
 * derivatives */
void gmm_compute_fisher_simple(int n, const float *v, const gmm_t * g, int flags, float *desc);


#endif
