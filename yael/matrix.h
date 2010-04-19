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

#ifndef __matrix_h
#define __matrix_h

/*---------------------------------------------------------------------------*/
/*! @addtogroup matrix
 *  @{
 */


/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

/*! @brief Allocate a new nrow x ncol matrix */
float *fmat_new (int nrow, int ncol);


/*! @brief Matrix multiplication
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

/*! @brief Same as fmat_mul, but allocate the memory and return the corresponding pointer */
float * fmat_new_mul (const float *left, const float *right,
		      int n, int m, int k);

/*! @brief Same as fmat_mul, but transpose left matrix (left of size m x n) */
void fmat_mul_tl (const float *left, const float *right,
		  int n, int m, int k, float *mout);

/*! @brief Same as fmat_mul_tl, but allocate the memory */
float *fmat_new_mul_tl (const float *left, const float *right, 
			int n, int m, int k);

/*! @brief Same as fmat_mul, but transpose right matrix (right of size k x m) */
void fmat_mul_tr (const float *left, const float *right,
		  int n, int m, int k, float *mout);

/*! @brief Same as fmat_mul_tr, but allocate the memory */
float *fmat_new_mul_tr (const float *left, const float *right, 
			int n, int m, int k);

/*! @brief Same as fmat_mul, but transpose both left and right matrices
  left is of size m * n and right of size k x m */
void fmat_mul_tlr (const float *left, const float *right,
		   int n, int m, int k, float *mout);

/*! @brief Same as fmat_mul_tlr, but allocate the memory */
float *fmat_new_mul_tlr (const float *left, const float *right, 
			int n, int m, int k);

/*! @brief Multiply a matrix by a vector */
float * fmat_mul_fvec (const float * a, const float * v, int nrow, int ncol);

/*! @brief display the matrix in matlab-like format */
void fmat_print (const float *a, int nrow, int ncol);



/*---------------------------------------------------------------------------*/
/* Matrix manipulation functions                                             */
/*---------------------------------------------------------------------------*/

/*! @brief return the submatrix defined by left-upper corner (included) 
  and top-down corner (not included) */
float *fmat_get_submatrix (const float *a, int ncola, int r1, int c1, int r2, int c2);

/* return the submatrix defined by a list of columns  */
float *fmat_get_columns (const float *a, int ncola, int nrow, int ncolout, const int *cols);

/*! @brief produce a matrix composed of the rows indicated by the vector rows */
float *fmat_get_rows (const float *a, int ncol, int nrowout, const int *rows);

/*! per-column sum of matrix elements */
float *fmat_sum_columns (const float *a, int ncol, int nrow);

/*! 
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


/*! @brief produce a new matrix of size nrow x ncol, filled with gaussian values */
float * fmat_new_rand_gauss (int nrow, int ncol);

/*! @brief produce a random orthogonal basis matrix of size d*d */
float *random_orthogonal_basis (int d);

/*! @brief Construct a Hadamard matrix of dimension d using the Sylvester construction.
   d should be a power of 2 */
float * hadamard (int d);


/*---------------------------------------------------------------------------*/
/* Statistical matrix operations                                             */
/*---------------------------------------------------------------------------*/

/* compute average of v matrix rows, subtract it to v and return average */
float *fmat_center_rows(int n,int d,float *v);

/* subtract a vector from all rows of a matrix m_i := m_i - avg*/
void fmat_subtract_from_rows(int n,int d,float *m,const float *avg);

/* reverse: m_i := avg - m_i */
void fmat_rev_subtract_from_rows(int n,int d,float *m,const float *avg);


/*! @brief Perform the Principal Component Analysis of a set of vectors,
 * v(n,d) stored per row
 *
 * return d*d matrix of eigenvectors, stored by row. To transform a
 * vector a low-dimension space, multiply by the d2 first lines of the matrix
 *
 */
float *compute_pca(int n, int d, float *v);

/* in covariance matrix, multiply of-block diagonal elements with
   weight (weight=1 => normal pca). bs=size of diagonal blocks */
float *compute_pca_with_weighted_blocks (int n, int d, float *v,
                                         int bs, double weight); 


/*! @brief compute only a few (nev) PCA vectors */
int partial_pca(int n,int d,const float *a,
                int nev,float *pcamat_out);


/*! @brief compute the nev first lines of U and V and S for a (row-major, m rows, n columns) 

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

