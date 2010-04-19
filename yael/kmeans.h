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

#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED


/*---------------------------------------------------------------------------*/
/*! @addtogroup clustering
 *  @{  */


/* layout of flags: 

 flags & 0xffff : use this many threads to compute 

 flags & KMEANS_QUIET: suppress kmeans output

 flags & KMEANS_INIT_RANDOM: random initialization 

 flags & KMEANS_NORMALIZE_CENTS: normalize centroids to L2=1 after they are computed

 flags & KMEANS_NORMALIZE_SOPHISTICATED: ditto, more sophisticated

*/

#define KMEANS_QUIET           0x10000
#define KMEANS_INIT_RANDOM     0x20000
#define KMEANS_NORMALIZE_CENTS 0x40000
#define KMEANS_NORMALIZE_SOPHISTICATED 0x80000


/*! @brief compute the k-means algorithm and return the quantization error. 
   The centroids parameters must be allocated. 
   TThe other output vectors (dis, assign and nassign) are not used if NULL */
float kmeans (int d, int n, int k, int niter, 
	      const float * v, int flags, int seed, int redo, 
	      float * centroids, float * dis, 
	      int * assign, int * nassign);


/*----------- Following functions are used for forward compatibility -----------*/

/*! @brief k-means clustering. */
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


/*! @} */
#endif
