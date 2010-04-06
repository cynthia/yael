#include <stdio.h>
#include <stdlib.h>

#include "eigs.h"
#include "vector.h"

int main()
{
  int i, j, d = 10;
  float * a = fvec_new (d * d);
  float * b = fvec_new (d * d);
  float * lambda = fvec_new (d);
  float * v = fvec_new (d * d);

  for (i = 0 ; i < d ; i++)
    for (j = 0 ; j  <= i ; j++) {
      a[i * d + j] = drand48();
      a[j * d + i] = a[i * d + j];
      b[i * d + j] = drand48();
      b[j * d + i] = b[i * d + j];
    }

  printf ("a = ");
  for (i = 0 ; i < d ; i++)
    fvec_print (a + i * d, d);
  printf ("\nb = ");

  for (i = 0 ; i < d ; i++)
    fvec_print (b + i * d, d);
	
  printf ("\n");
  eigs_sym (d, a, lambda, v);
  printf ("\n");

  printf ("Solution of the eigenproblem Av=lambda v\n");
  for (i = 0 ; i < d ; i++)
    fvec_print (v + i * d, d);

  fprintf(stdout, "lambda = ");
  fvec_print (lambda, d);
  printf ("\n");

  printf ("\n");
  geigs_sym (d, a, b, lambda, v);
  printf ("\n");

  printf ("Solution of the generalized eigenproblem Av=lambda B v\n");

  for (i = 0 ; i < d ; i++)
    fvec_print (v + i * d, d);
  
  fprintf(stdout, "lambda = ");
  fvec_print (lambda, d);
  printf ("\n");

  free (a);
  free (lambda);
  free (v);

  return 0;
}
