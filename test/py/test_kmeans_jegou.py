from yael import yael
import time

k = 40                           # number of cluster to create
d = 64                           # dimensionality of the vectors
n = 2000                         # number of vectors
nt = 2                            # number of threads to use
v = yael.fvec_new_randn (d * n)    # random set of vectors 
niter = 30                        # number of iterations
redo = 5                          # number of redo
dstep = 2                         # step for the dimensionality increment (Jegou's k-means)
niterstep = 5                     # maximum number of iteration per step

#[centroids, dis, assign] = yael_kmeans (v, 100, 'nt', 2, 'niter', 25);

centroids = yael.fvec_new (d * k) # output: centroids
dis = yael.fvec_new (n)           # point-to-cluster distance
assign = yael.ivec_new (n)        # quantization index of each point
nassign = yael.ivec_new (k)       # output: number of vectors assigned to each centroid

nassign = yael.IntArray.acquirepointer (nassign)

t1 = time.time()
yael.kmeans (d, n, k, niter, v, nt, 0, redo, centroids, dis, assign, nassign)
t2 = time.time()

yael.kmeans_jegou (d, n, k, dstep, niterstep, v, nt, 0, centroids)
yael.kmeans (d, n, k, niter, v, nt + yael.KMEANS_INIT_USER, 0, redo, centroids, dis, assign, nassign)

t3 = time.time()

print [nassign[i] for i in xrange(k)]
print 'kmeans performed in %.3fs' % (t2 - t1)
