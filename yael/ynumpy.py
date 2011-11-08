"""
This is a wrapper for yael's functions, so that all I/O of vectors and
matrices is done via numpy types. All errors are also raised as exceptions.
"""

import pdb 

import numpy
import yael


def _check_col_float32(a): 
    if a.dtype != numpy.float32: 
        raise TypeError('expected float32 matrix, got %s' % a.dtype)
    if not a.flags.f_contiguous:
        raise TypeError('expected Fortran order matrix')

def knn(base, queries, 
        nnn = 1, 
        distance_type = 2,
        nt = 1):
    _check_col_float32(base)
    _check_col_float32(queries)
    d, n = base.shape
    d2, nq = queries.shape
    assert d == d2, "base and queries must have same nb of rows (got %d != %d) " % (d, d2)
    
    idx = numpy.zeros((nnn, nq), dtype = numpy.int32, order = 'FORTRAN')
    dis = numpy.zeros((nnn, nq), dtype = numpy.float32, order = 'FORTRAN')

    yael.knn_full_thread(distance_type, 
                         nq, n, d, nnn,
                         yael.numpy_to_fvec_ref(base),
                         yael.numpy_to_fvec_ref(queries), 
                         None, 
                         yael.numpy_to_ivec_ref(idx), 
                         yael.numpy_to_fvec_ref(dis), 
                         nt, None)
    return idx, dis
    



def kmeans(v, k,
           distance_type = 2, 
           nt = 1, 
           niter = 30,
           seed = 0, 
           redo = 1, 
           verbose = True,
           init = 'random',
           output = 'centroids'):
    _check_col_float32(v)
    d, n = v.shape
    
    centroids = numpy.zeros((d, k), dtype = numpy.float32, order = 'FORTRAN')
    dis = numpy.zeros(n, dtype = numpy.float32)
    assign = numpy.zeros(n, dtype = numpy.int32)
    nassign = numpy.zeros(k, dtype = numpy.int32)
    
    flags = nt 
    if not verbose:          flags |= yael.KMEANS_QUIET

    if distance_type == 2:   pass # default
    elif distance_type == 1: flags |= yael.KMEANS_L1
    elif distance_type == 3: flags |= yael.KMEANS_CHI2

    if init == 'random':     flags |= yael.KMEANS_INIT_RANDOM
    elif init == 'kmeans++': pass # default

    qerr = yael.kmeans(d, n, k, niter, 
                       yael.numpy_to_fvec_ref(v), flags, seed, redo, 
                       yael.numpy_to_fvec_ref(centroids), 
                       yael.numpy_to_fvec_ref(dis), 
                       yael.numpy_to_ivec_ref(assign), 
                       yael.numpy_to_ivec_ref(nassign))

    if qerr < 0: 
        raise RuntimeError("kmeans: clustering failed. Is dataset diverse enough?")

    if output == 'centroids': 
        return centroids
    else: 
        return (centroids, qerr, dis, assign, nassign)



def fvecs_fsize(filename): 
    (fsize, d, n) = yael.fvecs_fsize(filename)
    if n < 0 and d < 0: 
        return IOError("fvecs_fsize: cannot read " + filename)
    # WARN: if file is empty, (d, n) = (-1, 0)
    return (d, n)

def fvecs_read(filename):
    (fvecs, n, d) = yael.fvecs_new_read(filename)
    if n == -1: 
        raise IOError("could not read " + filename)
    elif n == 0: d = 0    
    fvecs = yael.fvec.acquirepointer(fvecs)
    # TODO find a way to avoid copy
    a = yael.fvec_to_numpy(fvecs, n * d)
    return a.reshape((d, n), order='FORTRAN')


def fvecs_write(filename, matrix): 
    _check_col_float32(matrix)
    d, n = matrix.shape
    ret = yael.fvecs_write(filename, d, n, yael.numpy_to_fvec_ref(matrix))
    if ret != n:
        raise IOError("write error" + filename)

def ivecs_write(filename, matrix): 
    pass

def siftgeo_read(filename):

    # I/O via double pointers (too lazy to make proper swig interface)
    v_out = yael.BytePtrArray(1)
    meta_out = yael.FloatPtrArray(1)
    d_out = yael.ivec(2)

    n = yael.bvecs_new_from_siftgeo(filename, d_out, v_out.cast(),     
                                    d_out.plus(1), meta_out.cast())
    
    if n < 0: 
        raise IOError("cannot read " + filename)
    if n == 0: 
        v = numpy.array([[]], dtype = numpy.uint8, order = 'FORTRAN')
        meta = numpy.array([[]*9], dtype = numpy.float32, order = 'FORTRAN')
        return v, meta

    v_out = yael.bvec.acquirepointer(v_out[0])
    meta_out = yael.fvec.acquirepointer(meta_out[0])

    d = d_out[0]
    d_meta = d_out[1]
    assert d_meta == 9

    v = yael.bvec_to_numpy(v_out, n * d)
    v = v.reshape((d, n), order = 'FORTRAN')
    
    meta = yael.fvec_to_numpy(meta_out, n * d_meta)
    meta = meta.reshape((d_meta, n), order = 'FORTRAN')

    return v, meta

# In numpy, we represent gmm's as 3 matrices (like in matlab)
# when a gmm is needed, we build a "fake" yael gmm struct with 
# 3 vectors

def _gmm_to_numpy(gmm): 
    d, k = gmm.d, gmm.k
    w = yael.fvec_to_numpy(gmm.w, k)
    mu = yael.fvec_to_numpy(gmm.mu, d * k)
    mu = mu.reshape((d, k), order = 'FORTRAN')
    sigma = yael.fvec_to_numpy(gmm.sigma, d * k)
    sigma = sigma.reshape((d, k), order = 'FORTRAN')
    return w, mu, sigma

def _gmm_del(gmm): 
    gmm.mu = gmm.sigma = gmm.w = None
    yael.gmm_delete(gmm)
    # yael._yael.delete_gmm_t(gmm)


def _numpy_to_gmm((w, mu, sigma)):
    # produce a fake gmm from 3 numpy matrices. They should not be
    # deallocated while gmm in use
    _check_col_float32(mu)
    _check_col_float32(sigma)
    
    d, k = mu.shape
    assert sigma.shape == mu.shape
    assert w.shape == (k,)

    gmm = yael.gmm_t()
    gmm.d = d
    gmm.k = k
    gmm.w = yael.numpy_to_fvec_ref(w)
    gmm.mu = yael.numpy_to_fvec_ref(mu)
    gmm.sigma = yael.numpy_to_fvec_ref(sigma)
    gmm.__del__ = _gmm_del
    return gmm

def gmm_learn(v, k,
              nt = 1,
              niter = 30,
              seed = 0,
              redo = 1,
              use_weights = True): 
    _check_col_float32(v)
    d, n = v.shape
    
    flags = 0
    if use_weights: flags |= yael.GMM_FLAGS_W

    gmm = yael.gmm_learn(d, n, k, niter, 
                         yael.numpy_to_fvec_ref(v), nt, seed, redo, flags)
    
    gmm_npy = _gmm_to_numpy(gmm) 

    yael.gmm_delete(gmm)    
    return gmm_npy

def fisher(gmm_npy, v, 
           include = 'mu'): 

    _check_col_float32(v)
    d, n = v.shape

    gmm = _numpy_to_gmm(gmm_npy)
    assert d == gmm.d
    
    flags = 0

    if 'mu' in include:    flags |= yael.GMM_FLAGS_MU
    if 'sigma' in include: flags |= yael.GMM_FLAGS_SIGMA
    if 'w' in include:     flags |= yael.GMM_FLAGS_W

    d_fisher = yael.gmm_fisher_sizeof(gmm, flags)

    fisher_out = numpy.zeros(d_fisher, dtype = numpy.float32)    

    yael.gmm_fisher(n, yael.numpy_to_fvec_ref(v), gmm, flags, yael.numpy_to_fvec_ref(fisher_out))

    return fisher_out

    



