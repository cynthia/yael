#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vector.h"
#include "sorting.h"
#include "machinedeps.h"



extern void dsyev_( char *jobz, char *uplo, FINTEGER *n, double *a, FINTEGER *lda,
        double *w, double *work, FINTEGER *lwork, FINTEGER *info );

extern void dsygv_(FINTEGER * itype, char *jobz, char *uplo, FINTEGER *n, double *a, FINTEGER *lda,
		    double *b, FINTEGER *lbd, double *w, double *work, FINTEGER *lwork, FINTEGER *info );



int eigs_sym (int d, const float * m, float * eigval, float * eigvec)
{
  int i, j;
  double * md = (double *) memalign (16, sizeof (*md) * d * d);

  /* processing is performed in double precision */
  for (i = 0 ; i < d ; i++) {
    for (j = 0 ; j < d ; j++)
      md[i * d + j] = (float) m[i * d + j];
  }

  /* variable for lapack function */
  double workopt = 0;
  int lwork = -1, info;

  double * lambda = (double *) memalign (16, sizeof (*lambda) * d);
  dsyev_( "V", "L", &d, md, &d, lambda, &workopt, &lwork, &info );
  lwork = (int) workopt;
  double * work = (double *) memalign (16, lwork * sizeof (*work));
  dsyev_( "V", "L", &d, md, &d, lambda, work, &lwork, &info );
  
  if (info > 0)
    fprintf (stderr, "# eigs_sym: problem while computing eigen-vectors/values\n");

  /* normalize the eigenvectors, copy and free */
  double nr = 1;
  for (i = 0 ; i < d ; i++) {
    eigval[i] = (float) lambda[i];
    
    for (j = 0 ; j < d ; j++) 
      eigvec[i * d + j] = (float) (md[i * d + j] / nr);
  }

  free (md);
  free (lambda);
  free (work);
  return info <= 0;
}


int geigs_sym (int d, const float * a, const float * b, float * eigval, float * eigvec)
{
  int i, j;
  double * ad = (double *) memalign (16, sizeof (*ad) * d * d);
  double * bd = (double *) memalign (16, sizeof (*bd) * d * d);

  /* processing is performed in double precision */
  for (i = 0 ; i < d ; i++) 
    for (j = 0 ; j < d ; j++) {
      ad[i * d + j] = (float) a[i * d + j];
      bd[i * d + j] = (float) b[i * d + j];
    }
  
  /* variable for lapack function */
  double workopt = 0;
  int lwork = -1, info, itype = 1;

  double * lambda = (double *) memalign (16, sizeof (*lambda) * d);
  dsygv_ (&itype, "V", "L", &d, ad, &d, bd, &d, lambda, &workopt, &lwork, &info );
  lwork = (int) workopt;
  double * work = (double *) memalign (16, lwork * sizeof (*work));
  dsygv_ (&itype, "V", "L", &d, ad, &d, bd, &d, lambda, work, &lwork, &info );
  
  if (info > 0)
    fprintf (stderr, "# eigs_sym: problem while computing eigen-vectors/values\n");

  /* normalize the eigenvectors, copy and free */
  double nr = 1;
  for (i = 0 ; i < d ; i++) {
    eigval[i] = (float) lambda[i];
    
    for (j = 0 ; j < d ; j++) 
      eigvec[i * d + j] = (float) (ad[i * d + j] / nr);
  }

  free (ad);
  free (bd);
  free (lambda);
  free (work);
  return info <= 0;
}



void eigs_reorder (int d, float * eigval, float * eigvec, int criterion)
{
  int i;
  int * perm = ivec_new (d);

  float * eigvalst = fvec_new (d * d);
  float * eigvecst = fvec_new (d * d);

  fvec_sort_index (eigval, d, perm);

  if (criterion) 
    for (i = 0 ; i < d / 2 ; i++) {
      int tmp = perm[i];
      perm[i] = perm[d - 1 - i];
      perm[d - 1 - i] = tmp;
    }

  for (i = 0 ; i < d ; i++) {
    eigvalst[i] = eigval[perm[i]];
    memcpy (eigvecst + i * d, eigvec + perm[i] * d, sizeof (*eigvecst) * d);
  }

  memcpy (eigval, eigvalst, d * sizeof (*eigval));
  memcpy (eigvec, eigvecst, d * d * sizeof (*eigvec));

  free (eigvalst);
  free (eigvecst);
  free (perm);
}
