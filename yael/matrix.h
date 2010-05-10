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

#ifndef __matrix_h
#define __matrix_h

/*---------------------------------------------------------------------------*/
/*! @addtogroup matrix
 *  @{
 */


/*! @defgroup matrix
 * Matrix functions
 *
 * All matrices are stored in column-major order (like Fortran) and
 * indexed from 0 (like C, unlike Fortran). The declaration:
 *
 *     a(m, n) 
 * 
 * means that element a(i, j) is accessed with a[ i * m + j ] where
 *
 *     0 <= i < m and 0 <= j < n
 *
 */



/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

/*! Allocate a new nrow x ncol matrix */
float *fmat_new (int nrow, int ncol);



/*!  Matrix multiplication
 *
 * WARNING some matrix multiplication functions assume row-major storage! (noted with RM) 
 * 
 * computes mout = left * right
 * where 
 *   mout     is n-by-k 
 *   left     is n-by-m
 *   right    is m-by-k
 * (all matrices stored by lines, like in C, and packed)
 */
void fmat_mul (const float *left, const float *right,
	       int n, int m, int k, float *mout);

/*! RM Same as fmat_mul, but allocate the memory and return the corresponding pointer */
float * fmat_new_mul (const float *left, const float *right,
		      int n, int m, int k);

/*! RM Same as fmat_mul, but transpose left matrix (left of size m x n) */
void fmat_mul_tl (const float *left, const float *right,
		  int n, int m, int k, float *mout);

/*! RM Same as fmat_mul_tl, but allocate the memory */
float *fmat_new_mul_tl (const float *left, const float *right, 
			int n, int m, int k);

/*! RM Same as fmat_mul, but transpose right matrix (right of size k x m) */
void fmat_mul_tr (const float *left, const float *right,
		  int n, int m, int k, float *mout);

/*! RM Same as fmat_mul_tr, but allocate the memory */
float *fmat_new_mul_tr (const float *left, const float *right, 
			int n, int m, int k);

/*! RM Same as fmat_mul, but transpose both left and right matrices
  left is of size m * n and right of size k x m */
void fmat_mul_tlr (const float *left, const float *right,
		   int n, int m, int k, float *mout);

/*! RM Same as fmat_mul_tlr, but allocate the memory */
float *fmat_new_mul_tlr (const float *left, const float *right, 
			int n, int m, int k);

/*! RM Multiply a matrix by a vector */
float * fmat_mul_fvec (const float * a, const float * v, int nrow, int ncol);

/*! display the matrix in matlab-like format */
void fmat_print (const float *a, int nrow, int ncol);



/*---------------------------------------------------------------------------*/
/* Matrix manipulation functions                                             */
/*---------------------------------------------------------------------------*/

/*! RM  return the submatrix defined by left-upper corner (included) 
  and top-down corner (not included) */
float *fmat_get_submatrix (const float *a, int ncola, int r1, int c1, int r2, int c2);

/* RM return the submatrix defined by a list of columns  */
float *fmat_get_columns (const float *a, int ncola, int nrow, int ncolout, const int *cols);

/*! RM  produce a matrix composed of the rows indicated by the vector rows */
float *fmat_get_rows (const float *a, int ncol, int nrowout, const int *rows);

/*! RM  per-column sum of matrix elements */
float *fmat_sum_columns (const float *a, int ncol, int nrow);

/*! RM 
 * a is ncol-by-nrow
 * accu is k-by-k
 *
 * for i=0..ncol-1,j=0..nrow-1, do 
 *    accu(row_assign[i],col_assign[j]) += a(i,j)
 *
 */ 
void fmat_splat_separable(const float *a,int nrow,int ncol,
                          const int *row_assign,const int *col_assign,
                          int k,
                          float *accu); 

int *imat_joint_histogram(int n,int k,int *row_assign,int *col_assign);

/*---------------------------------------------------------------------------*/
/* Special matrices                                                          */
/*---------------------------------------------------------------------------*/


/*! RM  produce a new matrix of size nrow x ncol, filled with gaussian values */
float * fmat_new_rand_gauss (int nrow, int ncol);

/*! produce a random orthogonal basis matrix of size d*d */
float *random_orthogonal_basis (int d);

/*! Construct a Hadamard matrix of dimension d using the Sylvester construction.
   d should be a power of 2 */
float * hadamard (int d);


/*---------------------------------------------------------------------------*/
/* Statistical matrix operations                                             */
/*---------------------------------------------------------------------------*/

/* compute average of v matrix columns, subtract it to v and return average */
float *fmat_center_columns(int d,int n,float *v);

/* subtract a vector from all columns of a matrix m_i := m_i - avg*/
void fmat_subtract_from_columns(int d,int n,float *m,const float *avg);

/* reverse: m_i := avg - m_i */
void fmat_rev_subtract_from_columns(int d,int n,float *m,const float *avg);








/*! Perform the Principal Component Analysis of a set of vectors
 *
 * @param v(d,n)   vectors to perform the PCA on 
 *
 * @return (d,d) matrix of eigenvectors. To transform a
 *               vector a low-dimension space, multiply by the d2 first lines of the matrix
 */
float *fmat_pca(int d,int n,const float *v); 


/*! Compute covariance of a set of vectors
 * 
 * @param v(d,n)  vectors to compute covariance
 * @param avg(d)  on output, average vector (can be NULL)
 * 
 * @return (d,d)  covariance matrix
 */
float *fmat_covariance (int d, int n, const float *v,
                        float *avg);

/*! same as fmat_covariance, threaded 
 * 
 * @param nt      nb of computing threads
 */
float *fmat_covariance_thread (int d, int n, const float *v,
                               float *avg, int nt);


/*! Perform the Principal Component Analysis from a covariance matrix 
 *
 * @param cov(d,d)     covariance to compute the PCA on
 * @param singvals(d)  on output, associated singular values (can be NULL)
 *
 * @return (d,d) matrix of eigenvectors. To transform a
 *               vector a low-dimension space, multiply by the d2 first lines of the matrix
 */
float *fmat_pca_from_covariance(int d,const float *cov,
                                float *singvals); 

/*! same as fmat_pca_from_covariance, but return only part of the vectors 
 *
 * @param cov(d,d)     covariance to compute the PCA on
 * @param singvals(nv)  on output, associated singular values (can be NULL)
 *
 * @return (d,nv) matrix of eigenvectors. 
 */
float *fmat_pca_part_from_covariance(int d,int nv,const float *cov,
                                     float *singvals); 






/*! DEPRECATED compute only a few (nev) PCA vectors */
int partial_pca(int n,int d,const float *a,
                int nev,float *pcamat_out);


/*! DEPRECATED compute the nev first lines of U and V and S for a (row-major, m rows, n columns) 

   sout has size nev
   uout has size nev-by-m
   vout has size nev-by-m
   all can be NULL if you are not interested.
*/
int partial_svd(int m,int n,const float *a,
                int nev,
                float *sout,
                float *uout, float *vout,
                int nt);

/*---------------------------------------------------------------------------*/
/*! @} */
/*---------------------------------------------------------------------------*/

#endif 

/*---------------------------------------------------------------------------*/

