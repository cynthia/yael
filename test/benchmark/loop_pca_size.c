

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <yael/machinedeps.h>
#include <yael/vector.h>
#include <yael/eigs.h>
#include <yael/matrix.h>

/* 
executed on node15
gcc -o loop_pca_size loop_pca_size.c -O3 -I../.. -L ../../yael -lyael && ./loop_pca_size 


*/

void print_some_eigs(long d, long nev, const float *eigval, const float *eigvec) {
  printf("eigenvals = ["); 
  int i; 
  for(i = 0; i < nev && i < 10; i++) printf("%g ", eigval[i]); 
  
  printf("]\nvecs=["); 
  
  for(i = 0; i < d && i < 10; i++) {
    int j; 
    for(j = 0; j < nev && j < 10; j++)       
      printf("%9.6f ", eigvec[i + j * d]);
    if(j < nev) printf("...\n      "); 
    else printf("\n      "); 
  } 
  printf("]\n"); 
}


double sqr(double x) {return x*x; } 

#define real float
#define integer FINTEGER

int ssytrd_(char *uplo, integer *n, real *a, integer *lda, 
            real *d__, real *e, real *tau, real *work, integer *lwork, integer *
            info); 

int sstebz_(char *range, char *order, integer *n, real *vl, 
            real *vu, integer *il, integer *iu, real *abstol, real *d__, real *e, 
            integer *m, integer *nsplit, real *w, integer *iblock, integer *
            isplit, real *work, integer *iwork, integer *info);


#undef real 
#undef integer




int main(int argc, char** argv) {

/*
  long d = 64 * 256;
  long nblock = 160, blocksize = 1000; 
*/
  long d = 64 * 64; 
  long nblock = 8, blocksize = 1000; 
  long n = nblock * blocksize; 
  
  float *x = fvec_new(n * d); 

  int i; 
  

  for(i = 0; i < nblock; i++) {
    char fname[1024]; 
    //    sprintf(fname, "/scratch2/bigimbaz/dataset/flickr//smalldesc/fisher/ff_k256_%02d.fvecs", i); 
    sprintf(fname, "/scratch2/bigimbaz/dataset/flickr//smalldesc/fisher/ff_k64_%02d.fvecs", i); 
    printf("loading block %d %s\n", i, fname); 
    fvecs_read(fname, d, blocksize, x + i * blocksize * d);     
    
    long nnan = fvec_purge_nans(x + i * blocksize * d, blocksize * d, 0); 
    printf("  purged %d nans\n", nnan);
  }
  
  printf("loaded %ld pts in %ld dimensions\n", n, d);

  pca_online_t * pca_online = pca_online_new (d); 
  
  {
    printf("computing covariance matrix\n"); 
    double t0 = getmillisecs(); 
    
    for(i = 0; i < nblock; i++) {
      printf("block %d\n", i); 
      pca_online_accu(pca_online, x + i * blocksize * d, blocksize);
    }

    pca_online_cov(pca_online);
    
    printf("covariance time: %.3f ms\n", getmillisecs() - t0); 
  }

  {
    double t0 = getmillisecs(); 
    
    printf("subtracting mean from data\n"); 
    fmat_subtract_from_columns (d, n, x, pca_online->mu);

    printf("subtract mean time: %.3f ms\n", getmillisecs() - t0); 
  }
  
  {
    printf("computing full eigenvals / vecs\n"); 
    
    double t1 = getmillisecs(); 
    
    pca_online_complete(pca_online); 
    
    printf("full time: %.3f ms\n", getmillisecs() - t1); 
    
    print_some_eigs(d, d, pca_online->eigval, pca_online->eigvec); 
  }
  
  
  {
    int nev; 

    for(nev = 1; nev < d / 2; nev *= 2) {
      
      {
        memset(pca_online->eigval, -1, sizeof(*pca_online->eigval) * d); 
        memset(pca_online->eigvec, -1, sizeof(*pca_online->eigvec) * d * d); 
        
        printf("Arpack partial PCA %d evs (on-the-fly matrix-vec multiply):\n", nev); 
        
        double t1 = getmillisecs();       
        
        fmat_svd_partial_full(d, n, nev, x, 0, pca_online->eigval, pca_online->eigvec, NULL, 1); 
        
        for(i = 0; i < nev; i++) pca_online->eigval[i] = sqr(pca_online->eigval[i]) / (n - 1);           

        printf("time: %.3f ms\n", getmillisecs() - t1); 
        
        print_some_eigs(d, nev, pca_online->eigval, pca_online->eigvec);        
        
      }
      
      {
        memset(pca_online->eigval, -1, sizeof(*pca_online->eigval) * d); 
        memset(pca_online->eigvec, -1, sizeof(*pca_online->eigvec) * d * d);      
        printf("Arpack partial PCA %d evs (from covariance matrix):\n", nev); 

        double t1 = getmillisecs(); 
        
        pca_online_complete_part(pca_online, nev); 
        
        printf("time: %.3f ms\n", getmillisecs() - t1); 
        
        print_some_eigs(d, nev, pca_online->eigval, pca_online->eigvec);       
      }

      {
        memset(pca_online->eigval, -1, sizeof(*pca_online->eigval) * d); 
        memset(pca_online->eigvec, -1, sizeof(*pca_online->eigvec) * d * d);      
        printf("dstebz partial PCA %d evs (from covariance matrix):\n", nev); 

        double t1 = getmillisecs(); 
        

        {
          /* tri-diagonalize cov matrix */
          float *a = fvec_new(d * d); 
          fvec_cpy(a, pca_online->cov, d * d * sizeof(*a)); 
          float *tmp = fvec_new(d * 3); 
          float *dtab = tmp, *etab = tmp + d, tautab = tmp + 2 * d; 
          FINTEGER info, lwork = -1; 
          float workq[1];           

          FINTEGER ni = d; 

          ssytrd_("U", &ni, a, &ni, dtab, etab, tautab, workq, &lwork, &info); 
          
          lwork = (long)workq[0]; 
          float *work = fvec_new(lwork); 
                   
          ssytrd_("U", &ni, a, &ni, dtab, etab, tautab, work, &lwork, &info); 

          if(info != 0) {
            printf("  ssytrd info = %d\n", info); 
            abort();
          }
          
          printf("tri-diagonalization time: %.3f ms\n", getmillisecs() - t1); 
        

          /* call sstebz */
          float unused = -1; 
          FINTEGER il = d - nev, iu = nev, m, nsplit; 
          float abstol = 1e-3; 

          FINTEGER *iblock = malloc(sizeof(FINTEGER) * d); 
          FINTEGER *isplit = malloc(sizeof(FINTEGER) * d); 
          FINTEGER *iwork = malloc(sizeof(FINTEGER) * d * 3); 
          free(work); 
          work = fvec_new(3 * n); 

          sstebz_("I", "E", &ni, NULL, NULL, &il, &iu, &abstol, dtab, etab, &m, &nsplit, pca_online->eigval, 
                  iblock, isplit, work, iwork, &info); 
          
          if(info != 0) {
            printf("  sstebz info = %d\n", info); 
            abort();
          }
                            
        }



        
        printf("time: %.3f ms\n", getmillisecs() - t1); 
        
        print_some_eigs(d, nev, pca_online->eigval, pca_online->eigvec);       
      }

    }
    
    
  }
  

  
  return 0; 

}
