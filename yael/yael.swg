// -*- c++ -*- 

%module yael;


/*****************************************************
 * a few useful functions & objects for everywhere:
 * - array objects 
 * - nearest neighbour and clustering functions
 */


%{

#include "nn.h"
#include "kmeans.h"
#include "hkm.h"
#include "sorting.h"
#include "vector.h"
#include "matrix.h"
#include "eigs.h"
#include "gmm.h"
#include "vlad.h"
#include "machinedeps.h"

/* seed generators used in the rest */
void common_srandom(int seed) {
  srandom(seed);
  srand(seed);
  srand48(seed); 
}

 typedef int * int_pointer_t;
 typedef float * float_pointer_t;

%}

%include "common.swg"



array_class_plus(double,DoubleArray)
array_class_plus(float,FloatArray)
array_class_plus(int,IntArray)
array_class_plus(unsigned char,bvec)
array_class_plus(long long,lvec)
array_class_plus(unsigned long long,UInt64Array)


%extend IntArray {
  IntArray(PyObject *list) {
    if(!PySequence_Check(list)) {
      PyErr_SetString(PyExc_TypeError,"need sequence as input");
      return NULL;
    }  
    long sz=PySequence_Size(list),i;
    int *ret=malloc(sizeof(int)*sz);
    for(i=0;i<sz;i++) {
      PyObject *li=PySequence_GetItem(list,i);
      if(!PyInt_Check(li))  {
        PyErr_SetString(PyExc_TypeError,"need ints in sequence");
        free(ret);
        return NULL;
      }  
      ret[i]=PyInt_AsLong(li);
    }
    return ret;
  }
}



%extend FloatArray {
  FloatArray(PyObject *list) {
    if(!PySequence_Check(list)) {
      PyErr_SetString(PyExc_TypeError,"need sequence as input");
      return NULL;
    }  
    long sz=PySequence_Size(list),i;
    float *ret=malloc(sizeof(float)*sz);
    for(i=0;i<sz;i++) {
      PyObject *li=PyList_GetItem(list,i);
      if(!PyFloat_Check(li))  {
        PyErr_SetString(PyExc_TypeError,"need floats in sequence");
        free(ret);
        return NULL;
      }  
      ret[i]=PyFloat_AsDouble(li);
    }
    return ret;
  }
}

typedef int * int_pointer_t;
typedef float * float_pointer_t;

array_class_plus(int_pointer_t,IntPtrArray)
array_class_plus(float_pointer_t,FloatPtrArray)

%pythoncode {

  fvec=FloatArray
  dvec=DoubleArray
  ivec=IntArray
}



/*************************************************************************
 * nn.h
 */

%exception {
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

// ?vecs_size functions returns a tuple (nbytes, n,d)
%typemap(in,numinputs=0) 
  (int *d_out,int *n_out)
  (int d,int n) {
  $1=&d;
  $2=&n;
}

%typemap(argout) (int *d_out,int *n_out)  {
  $result=Py_BuildValue
    ("(Nii)", $result, *$1,*$2);
}



// fvecs_new_read and fvecs_new_fread_max returns a tuple 
// (floats,n,d)
// floats= float table
// d=dimension of the pts (128)
// n=nb of vectors or -1 on error

%typemap(in,numinputs=0) 
  (int *d_out,float **vf)
  (int d,float *vf) {
  $1=&d;
  $2=&vf;
}

%typemap(argout) (int *d_out,float **vf)  {
  $result=Py_BuildValue
        ("(NNi)",
         SWIG_NewPointerObj(*$2,SWIGTYPE_p_float,0),
         $result,*$1);
}


// ivecs_new_read returns a tuple 
// (ints,n,d)
// ints= float table
// d=dimension of the pts (128)
// n=nb of centroids or -1 on error

%typemap(in,numinputs=0) 
  (int *d_out,int **vi)
  (int d,int *vi) {
  $1=&d;
  $2=&vi;
}

%typemap(argout) (int *d_out,int **vi)  {
  $result=Py_BuildValue
        ("(NNi)",
         SWIG_NewPointerObj(*$2,SWIGTYPE_p_int,0),
         $result,*$1);
}

// for ivec_new_read

%typemap(in,numinputs=0) 
  (int *d_out)
  (int d) {
  $1=&d;
}

%typemap(argout) (int *d_out)  {
  $result=Py_BuildValue
        ("(Ni)",
         $result,*$1);
}




// slow functions with peek
PARALLEL_FUNCTION_PEEK(nn)
PARALLEL_FUNCTION_PEEK(knn)
PARALLEL_FUNCTION_PEEK(knn_thread)
PARALLEL_FUNCTION_PEEK(nn_thread)

PARALLEL_FUNCTION_PEEK(knn_full)
PARALLEL_FUNCTION_PEEK(knn_full_thread)




%include "nn.h"


/*************************************************************************
 * kmeans.h
 */


// wrap clustering_kmeans_assign to return (clust_assign,*cluster_size)
%typemap(in,numinputs=0)
        (int ** clust_assign_out) 
  (int *clust_assign) {
  $1=&clust_assign;
}

%typemap(argout) (int ** clust_assign_out)  {
  $result=Py_BuildValue
        ("(NN)",
         $result,
        SWIG_NewPointerObj(*$1,SWIGTYPE_p_int,0));
}



%include "kmeans.h"

%include "hkm.h"

/*************************************************************************
 * sorting.h
 */

%typemap(in,numinputs=0)
        (int **labels_out,float **vals_out) 
  (int *labels,float *vals) {
  $1=&labels;
  $2=&vals;
}

%typemap(argout) (int **labels_out,float **vals_out) {
  $result=Py_BuildValue
        ("(NNN)",
         $result,
         SWIG_NewPointerObj(*$1,SWIGTYPE_p_int,0),
         SWIG_NewPointerObj(*$2,SWIGTYPE_p_float,0));
}




%include "sorting.h"

/*************************************************************************
 * vector.h
 */


%typemap(in,numinputs=0) 
  (int ** idx_out, float ** v_out)
  (int *idx,float *v) {
  $1=&idx;
  $2=&v;
}

%typemap(argout) (int ** idx_out, float ** v_out)  {
  $result=Py_BuildValue
        ("(NNN)",
         $result,
         SWIG_NewPointerObj(*$1,SWIGTYPE_p_int,0),
         SWIG_NewPointerObj(*$2,SWIGTYPE_p_float,0));
}


%typemap(in,numinputs=0) 
  (float ** vf_out)
  (float *vf) {
  $1=&vf;
}

%typemap(argout) (float ** vf_out)  {
  $result=Py_BuildValue
        ("(NN)",
         $result,
         SWIG_NewPointerObj(*$1,SWIGTYPE_p_float,0));
}




%include "vector.h"



/*************************************************************************
 * matrix.h
 */


%include "matrix.h"

%typemap(in,numinputs=0) 
  (float **x_out, float **y_out)
  (float *x,float *y) {
  $1=&x;
  $2=&y;
}

%typemap(argout) (float **x_out, float **y_out) {
  $result=Py_BuildValue
        ("(NNN)",
         $result,
         SWIG_NewPointerObj(*$1,SWIGTYPE_p_float,0),
         SWIG_NewPointerObj(*$2,SWIGTYPE_p_float,0));
}


%include "eigs.h"


/*************************************************************************
 * gmm.h
 */


%include "gmm.h"


%include "vlad.h"


/*************************************************************************
 * allows freeing some elements allocated by malloc/calloc in python
 */
void free (void *);


%include "machinedeps.h"

void common_srandom(int seed);


void free(void*);



/*************************************************************************
 * Numpy wrapper 
 */

#ifdef HAVE_NUMPY


%{
#define SWIG_FILE_WITH_INIT

#include <numpy/arrayobject.h>


PyObject *fvec_to_numpy(const float *src,long di) {
  npy_intp d=di;
  PyObject* ob=PyArray_SimpleNew(1,&d,NPY_FLOAT);
  float *dest=PyArray_DATA(ob);
  memcpy(dest,src,sizeof(float)*d);
  return ob;
}

PyObject *ivec_to_numpy(const int *src,long di) {
  npy_intp d=di;
  PyObject* ob=PyArray_SimpleNew(1,&d,NPY_INT);
  int *dest=PyArray_DATA(ob);
  memcpy(dest,src,sizeof(int)*d);
  return ob;
}

PyObject *fvec_to_numpy_ref(float *src,long di) {
  npy_intp d=di;
  return PyArray_SimpleNewFromData(1,&d,NPY_FLOAT,src);
}

PyObject *ivec_to_numpy_ref(int *src,long di) {
  npy_intp d=di;
  return PyArray_SimpleNewFromData(1,&d,NPY_INT,src);
}


#define CHECK_NUMPY_ARRAY(expected_npy)                                 \
  if(!a) {                                                              \
    fprintf(stderr,"%s: NULL input\n",__PRETTY_FUNCTION__);             \
    return NULL;                                                        \
  }                                                                     \
  if(!PyArray_Check(a)) {                                               \
    fprintf(stderr,"%s: input not numpy array\n",__PRETTY_FUNCTION__);  \
    return NULL;                                                        \
  }                                                                     \
  if(PyArray_TYPE(a)!=expected_npy) {                                   \
    fprintf(stderr,"%s: input has bad type (type id %d != %d)\n",__PRETTY_FUNCTION__, \
            PyArray_TYPE(a),expected_npy);                              \
    return NULL;                                                        \
  }

/*
                                                                     \
  if(((PyArrayObject *)a)->nd!=1) {                                     \
    fprintf(stderr,"%s: input in %d D (expected 1D)\n",__PRETTY_FUNCTION__, \
            ((PyArrayObject *)a)->nd);                  \
    return NULL;                                                        \
  }
*/
float *numpy_to_fvec(PyObject *a) {
  CHECK_NUMPY_ARRAY(NPY_FLOAT)
  size_t sz=PyArray_SIZE(a)*PyArray_ITEMSIZE(a);
  void *ret=malloc(sz);
  memcpy(ret,PyArray_DATA(a),sz);
  return ret;
}

float *numpy_to_fvec_ref(PyObject *a) {
  CHECK_NUMPY_ARRAY(NPY_FLOAT)
  return PyArray_DATA(a);
}

int *numpy_to_ivec(PyObject *a) {
  CHECK_NUMPY_ARRAY(NPY_INT)
  size_t sz=PyArray_SIZE(a)*PyArray_ITEMSIZE(a);
  void *ret=malloc(sz);
  memcpy(ret,PyArray_DATA(a),sz);
  return ret;
}

int *numpy_to_ivec_ref(PyObject *a) {
  CHECK_NUMPY_ARRAY(NPY_INT)
  return PyArray_DATA(a);
}


%}


%init %{
/* needed, else crash at runtime */
    import_array();
%}



/* yael -> numpy */

PyObject *fvec_to_numpy(const float *fv,int n);
PyObject *ivec_to_numpy(const int *fv,int n);

/* numpy -> yael */

float *numpy_to_fvec(PyObject *numpyarray);
int *numpy_to_ivec(PyObject *a);

/* Ditto, without copying data */

PyObject *fvec_to_numpy_ref(float *fv,int n);
PyObject *ivec_to_numpy_ref(int *fv,int n);

float *numpy_to_fvec_ref(PyObject *numpyarray);
int *numpy_to_ivec_ref(PyObject *a);


#endif
