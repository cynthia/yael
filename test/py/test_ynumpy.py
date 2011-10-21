import pdb
import numpy 
from yael import ynumpy


print "test knn"

d = 4
n = 5
nq = 3
nnn = 2

base = numpy.array([range(i, i+d) for i in range(5)], 
                   dtype = numpy.float32).transpose()

queries = numpy.array([[x + 0.25 for x in range(i, i+d)]
                       for i in range(nq)], 
                      dtype = numpy.float32).transpose()

print "base="
print base

print "queries="
print queries

idx, dis = ynumpy.knn(base, queries, nnn, distance_type = 1)

print "indices="
print idx 

print "distances="
print dis


try: 
    v, meta = ynumpy.siftgeo_read('/Users/matthijs//Desktop/papers/lhl/trunk/data/test_query_10k.siftgeo')
except Exception, e: 
    print e
    print "generating random data"
    v = numpy.random.normal(0, 1, size = (20, 4)).transpose()
    v[:,10:] += numpy.tile(numpy.random.uniform(-10, 10, size = (4, 1)), (1, 10))
else: 
    print "vectors = "
    print v
    print "meta info = "
    print meta
    
    # numpy has a bias en favor of C-ordered arrays, hence this 
    v = v.transpose().astype(numpy.float32).transpose()


print "kmeans:"

centroids = ynumpy.kmeans(v, 3)

print "result centroids ="
print centroids[:10,:]

print "gmm:"

gmm = ynumpy.gmm_learn(v, 3)

(w, mu, sigma) = gmm

print "mu = "
print mu[:10,:]

print "sigma = "
print sigma[:10,:]

muc = mu.transpose().copy().transpose()
muc += numpy.random.normal(0, 0.2, size = muc.shape)

fish = ynumpy.fisher(gmm, muc)

print fish.shape
