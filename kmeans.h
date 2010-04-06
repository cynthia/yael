#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED



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


/* compute the k-means algorithm and return the quantization error. 
   The centroids parameters must be allocated. 
   TThe other output vectors (dis, assign and nassign) are not used if NULL */
float kmeans (int d, int n, int k, int niter, 
	      const float * v, int flags, int seed, int redo, 
	      float * centroids, float * dis, 
	      int * assign, int * nassign);


#endif
