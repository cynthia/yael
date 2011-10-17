"""
This is a wrapper for yael's functions, so that all I/O of vectors and
matrices is done via numpy types. All errors are also converted to exceptions.
"""

import pdb 

import numpy
import yael


def check_col_float32(a): 
    if a.dtype != numpy.float32: 
        raise TypeError('expected float32 matrix, got %s' % a.dtype)
    if not a.flags.f_contiguous:
        raise TypeError('expected Fortran order matrix')

def knn(base, queries, 
        nnn = 1, 
        distance_type = 2,
        nt = 1):
    check_col_float32(base)
    check_col_float32(queries)
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
           output = 'centroids'):
    check_col_float32(v)
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
    if d == -1: 
        raise IOError("could not read " + filename)
    fvecs = yael.acquirepointer(fvecs)
    # TODO find a way to avoid copy
    a = yael.fvec_to_numpy(fvecs, n * d)
    return a.reshape((d, n), order='FORTRAN')

def fvecs_write(filename, matrix): 
    pass

def ivecs_write(filename, matrix): 
    pass

def siftgeo_read(filename):

    # I/O via double pointers (too lazy to make 
    v_out = yael.BytePtrArray(1)
    meta_out = yael.FloatPtrArray(1)
    d_out = yael.ivec(2)

    n = yael.bvecs_new_from_siftgeo(filename, d_out, v_out.cast(),     
                                    d_out.plus(1), meta_out.cast())
    if n < 0: 
        raise IOError("cannot read " + filename)

    d = d_out[0]
    d_meta = d_out[1]
    assert d_meta == 9

    v = yael.bvec_to_numpy(v_out[0], n * d)
    v = v.reshape((d, n), order='FORTRAN')
    
    meta = yael.fvec_to_numpy(meta_out[0], n * d_meta)
    meta = meta.reshape((d_meta, n), order='FORTRAN')

    return v, meta


    
