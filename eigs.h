#ifndef __eigs_h
#define __eigs_h


/* compute the eigenvalues and eigvectors of a symmetric matrix m

  eigval   the n eigenvalues
  eigvec   Eigenvector j is  eigvec[j*d] .. eigvec[j*(d+1)-1]

  the vectors eigval and eigvec must be allocated externally
*/
int eigs_sym (int d, const float * m, float * eigval, float * eigvec);


/* generalized eigenvector problem */
int geigs_sym (int d, const float * a, const float * b, float * eigval, float * eigvec);


/* re-ordering of the eigenvalues and eigenvectors for a given criterion
   criterion=0    ascending order 
   criterion=1    descending order 
*/
void eigs_reorder (int d, float * eigval, float * eigvec, int criterion);

#endif
