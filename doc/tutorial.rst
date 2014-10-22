
Yael tutorial
=============

This tutorial presents some fully functional examples of how to use
Yael for image retrieval. They are implemented in the three languages
Yael is available in: C, Python and Matlab. 

The tutorial assumes Yael is compiled correctly (numpy should be
enabled for the Python examples). There is no separate compile and
install stage for Yael.

A simple example: k-means
-------------------------

C version 
+++++++++

Yael was originally designed as a lightweihgt library for vector
manipulation in C. Functions in ``yael/vector.h`` do exactly
that. They are seldom useful when used from numerical languages like
Numpy or Matlab.

The following example peforms a k-means clustering on a set of random
vectors.

.. code-block:: c
  
  #include <stdio.h>
  
  #include <yael/vector.h>
  #include <yael/kmeans.h>
  #include <yael/machinedeps.h>
  
  int main (int argc, char ** argv)
  {
    int k = 50;                           /* number of cluster to create */
    int d = 20;                           /* dimensionality of the vectors */
    int n = 1000;                         /* number of vectors */
    int nt = 2;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */
  
    float * v = fvec_new_rand (d * n);    /* random set of vectors */
  
    /* variables are allocated externaly */
    float * centroids = fvec_new (d * k); /* output: centroids */
    float * dis = fvec_new (n);           /* point-to-cluster distance */
    int * assign = ivec_new (n);          /* quantization index of each point */
    int * nassign = ivec_new (k);         /* output: number of vectors assigned to each centroid */
  
    double t1 = getmillisecs();
    kmeans (d, n, k, niter, v, 1, 1, redo, centroids, dis, assign, nassign);
    double t2 = getmillisecs();
  
    printf ("kmeans performed in %.3fs\n", (t2 - t1)  / 1000);
    
    ivec_print (nassign, k);
  
    free(v); free(centroids); free(dis); free(assign); free(nassign);

    return 0;
  }

This code can be compiled with::

  gcc test_kmeans.c -I my_yael_dir -L my_yael_dir/yael/ -lyael

where ``my_yael_dir`` is the directory where Yael was compiled. Since
there is no separate install stage, the shared library and include files are
in a single directory (``my_yael_dir/yael``). It should run in less than 100 ms.

By convention, the includes are relative to the install directory,
so they always have a ``yael/`` prefix.

``kmeans`` is an important Yael function. It has several variants that
return only part of the information, eg. only the centroid table. When
there are different variants, the primitive one returns vector and
matrix results in arrays allocated by the caller. 




Python version
++++++++++++++

The equivalent call to kmeans is

.. code-block:: python

  import numpy as np  
  from yael import ynumpy
  import time

  k = 50                           # number of cluster to create 
  d = 20                           # dimensionality of the vectors 
  n = 1000                         # number of vectors 
  nt = 2                           # number of threads to use 
  niter = 0                        # number of iterations (0 for convergence)
  redo = 1                         # number of redo 

  v = np.random.rand(n, d)         # line vectors in Numpy!
  v = v.astype('float32')          # yael likes floats better than doubles
  
  t0 = time.time()

  (centroids, qerr, dis, assign, nassign) = \
        ynumpy.kmeans(v, k, nt = nt, niter = niter, redo = redo, output = 'full')
    
  t1 = time.time()

  print "kmeans performed in %.3f s" % (t1 - t0)

  print nassign
  
To run this, the PYTHONPATH should point to ``my_yael_dir``. Since the
import statement is ``from yael import ...`` Python will know it has to
look in the subdirectory ``yael``.

The kmeans call is very similar to the C version. Only the arguments
``v`` and ``k`` are mandatory. For the other ones, it will use
reasonable defaults.

Matlab version
++++++++++++++ 

.. code-block:: matlab

  % The subdirectory 'matlab' of yael should be in the Matlab path
  % This can be done with the command addpath('MY_YAEL_MATLAB_PATH')
  
  k = 50;                    % number of cluster to create
  d = 20;                    % dimensionality of the vectors
  n = 1000;                  % number of vectors
  v = single(rand (d, n));   % random set of vectors 
  niter = 0;                 % typically use no more than 50 in practice
  redo = 1;                  % number of redo
  seed = 3;                  % 0: no seeding, values > 0 are used as seed

  tic
  % Only the two first arguments are mandatory
  [centroids, dis, assign, nassign] = yael_kmeans (v, k, 'niter', niter, 'redo', 1, 'seed', seed);
  toc


Image indexing example
----------------------

Here we work out an image indexing engine and apply it to a tiny image
dataset. 

We are going to work on the 100 first query images of the 
`Holidays <http://lear.inrialpes.fr/~jegou/data.php#holidays>`_ dataset, 
and their associated database examples. Download
the images and the SIFT descriptors from here:

http://pascal.inrialpes.fr/data2/douze/holidays_subset/images.tgz
http://pascal.inrialpes.fr/data2/douze/holidays_subset/sifts.tgz

Unzip them to a ``holidays_100`` subdirectory.

Image indexing in Python with Fisher vectors
++++++++++++++++++++++++++++++++++++++++++++

Image indexing based on Fisher vectors consists in computing a global
Fisher vector (FV) for each image, using the local SIFTs from these
images. Then the L2 distance between FVs is a good approximation of
the similarity of the contents of the images. See 
`Aggregating local image descriptors into compact codes <https://hal.inria.fr/inria-00633013>`_
for more details.

The FV computation relies on a training where a Gaussian Mixture Model
(GMM) is fitted to a set of representative local descriptors. For
simplicity, we are going to use the descriptors of the database we
index. 

In the following, you can just copy/paste the code to the Python
interpreter (or put it in a script). You can inspect the variables,
which are plain numpy arrays.

We first load all the descriptors

.. code-block:: python
   
   import os
   import numpy as np
   from yael import ynumpy

   # list of available images 
   image_names = [filename.split('.')[0] 
                  for filename in os.listdir('holidays_100') 
                  if filename.endswith('.siftgeo')]

   # load the SIFTS for these images
   image_descs = []
   for imname in image_names: 
       desc, meta = ynumpy.siftgeo_read("holidays_100/%s.siftgeo" % imname)
       if desc.size == 0: desc = np.zeros((0, 128), dtype = 'uint8')
       # we drop the meta-information (point coordinates, orientation, etc.)
       image_descs.append(desc)
   
So now we can sample the descriptors to reduce their dimensionality by
PCA and computing a GMM. For a GMM of size k (let's set it to 16), we
need about 1000*k training descriptors

.. code-block:: python

   # make a big matrix with all image descriptors
   all_desc = np.vstack(image_descs)

   k = 16
   n_sample = k * 1000
   
   np.random.shuffle(all_desc)
   sample = all_desc[:n_sample]

   # until now sample was in uint8. Convert to float32
   sample = sample.astype('float32')

   # compute mean and covariance matrix for the PCA
   mean = sample.mean(axis = 0)
   sample = sample - mean
   cov = np.dot(sample.T, sample)
   
   # compute PCA matrix and keep only 64 dimensions
   eigvals, eigvecs = np.linalg.eig(cov)
   perm = eigvals.argsort()                   # sort by increasing eigenvalue   
   pca_transform = eigvecs[:, perm[64:128]]   # eigenvectors for the 64 last eigenvalues

   # transform sample with PCA
   sample = np.dot(sample, pca_transform)

   # train GMM
   gmm = ynumpy.gmm_learn(sample, k)
   
The gmm is a tuple containing the a-priori weights per mixture
component, the mixture centres and the diagonal of the component
covariance matrices.

The training is finished. The next stage is to encode the SIFTs into
one vector per image. We choose to include only the derivatives w.r.t
mu in the FV, which results in a FV of size k * 64.

.. code-block:: python

   image_fvs = []
   for image_desc in image_descs: 
      # first apply the PCA to the image descriptor
      image_desc = np.dot(image_desc - mean, pca_transform)
      fv = ynumpy.fisher(gmm, image_desc)
      image_fvs.append(fv)

   # make one matrix with all FVs
   image_fvs = np.vstack(image_fvs)

   # power-normalize all descriptors at once
   image_fvs = np.sign(image_fvs) * np.abs(image_fvs) ** 0.5
   
   # L2 normalize
   norms = np.sqrt(np.sum(image_fvs ** 2, 1))
   image_fvs /= norms.reshape(-1, 1)
   
   # handle images with no local descriptor (100 = far away from "normal" images)
   image_fvs[np.isnan(image_fvs)] = 100

Now the FV can be used to compare images, so we loop over the Holidays
query images and retrieve the nearest images in the ``image_fvs`` matrix.

.. code-block:: python

   # get the indices of the query images (the ones whose names end with "00")
   query_images = [i for i, name in enumerate(image_names) if name[-2:] == "00"]

   # corresponding descriptors
   query_fvs = image_fvs[query_images]

   # get the 4 NNs for all query images in the image_fvs array
   results, distances = ynumpy.knn(query_fvs, image_fvs, nnn = 10)


display results

compute mAP


Image indexing in Matlab with inverted files
++++++++++++++++++++++++++++++++++++++++++++

