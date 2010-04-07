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
#include "sorting.h"
#include "vector.h"
#include "matrix.h"
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
array_class_plus(unsigned long long,UInt64Array)

%extend IntArray {
  IntArray(PyObject *list) {
    if(!PyList_Check(list)) {
      PyErr_SetString(PyExc_TypeError,"need List as input");
      return NULL;
    }  
    long sz=PyList_Size(list),i;
    int *ret=malloc(sizeof(int)*sz);
    for(i=0;i<sz;i++) {
      PyObject *li=PyList_GetItem(list,i);
      if(!PyInt_Check(li))  {
        PyErr_SetString(PyExc_TypeError,"need ints in list");
        free(ret);
        return NULL;
      }  
      ret[i]=PyInt_AsLong(li);
    }
    return ret;
  }
}

typedef int * int_pointer_t;
typedef float * float_pointer_t;

array_class_plus(int_pointer_t,IntPtrArray)
array_class_plus(float_pointer_t,FloatPtrArray)





/*************************************************************************
 * nn.h
 */

%exception {
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}


// fvecs_new_read and fvecs_new_fread_max returns a tuple 
// (floats,n,d)
// floats= float table
// d=dimension of the pts (128)
// n=nb of centroids or -1 on error

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
 * clustering.h
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

void fvec_to_numpy(const float *src,float* dest, long n) {
  memcpy(dest,src,sizeof(*dest)*n);
}

float *numpy_to_fvec(const float* x, long n) {
  float *ret=malloc(sizeof(*x)*n);
  memcpy(ret,x,sizeof(*x)*n);
  return ret;
}

void ivec_to_numpy(const int *src,int* dest, long n) {
  memcpy(dest,src,sizeof(*dest)*n);
}

int *numpy_to_ivec(const int* x, long n) {
  int *ret=malloc(sizeof(*x)*n);
  memcpy(ret,x,sizeof(*x)*n);
  return ret;
}


%}

%include "numpy.i"

%init %{
    import_array();
%}

void fvec_to_numpy(const float *fv,float* ARGOUT_ARRAY1, int DIM1);
float *numpy_to_fvec(float* INPLACE_ARRAY1, int DIM1);

void ivec_to_numpy(const int *fv,int* ARGOUT_ARRAY1, int DIM1);
int *numpy_to_ivec(int* INPLACE_ARRAY1, int DIM1);


#endif
