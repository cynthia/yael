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


#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "nn.h"
#include "vector.h"
#include "sorting.h"


size_t vlad_sizeof(int k, int d, int flags) {
  size_t s=d*k;
  if(flags==6) s*=3;
  if(flags==15) s*=2;
  if(flags==17) s*=2;
  return s;
}


double static inline sqr (double x)
{
  return x * x;
}


void vlad_compute(int k, int d, const float *centroids, 
                  int n, const float *v,
                  int flags, float *desc) {
  
  int i,j,l;


  if(flags<11 || flags>=13) {
    int *assign=ivec_new(n);

    nn(n,k,d,centroids,v,assign,NULL,NULL);    
    
    if(flags==6 || flags==7) {
      int n_quantile=flags==6 ? 3 : 1;
      fvec_0(desc,k*d*n_quantile);
      int *perm=ivec_new(n);
      float *tab=fvec_new(n);
      ivec_sort_index(assign,n,perm);
      int i0=0;
      for(i=0;i<k;i++) {
        int i1=i0;
        while(i1<n && assign[perm[i1]]==i) i1++;
        
        if(i1==i0) continue;
        
        for(j=0;j<d;j++) {        
          for(l=i0;l<i1;l++)
            tab[l-i0]=v[perm[l]*d+j];
          int ni=i1-i0;
          fvec_sort(tab,ni);
          for(l=0;l<n_quantile;l++) 
            desc[(i*d+j)*n_quantile+l]=(tab[(l*ni+ni/2)/n_quantile]-centroids[i*d+j])*ni;
        }
        
        i0=i1;
      }
      free(perm);
      free(tab);
    } else if(flags==5) {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j];
      }
      
    } else if(flags==8 || flags==9) {
      fvec_0(desc,k*d);
      
      float *u=fvec_new(d);
      
      for(i=0;i<n;i++) {
        fvec_cpy(u,v+i*d,d);
        fvec_sub(u,centroids+assign[i]*d,d);
        float un=sqrt(fvec_norm2sqr(u,d));
        
        if(un==0) continue;
        if(flags==8) {        
          fvec_div_by(u,d,un);
        } else if(flags==9) {
          fvec_div_by(u,d,sqrt(un));
        }
        
        fvec_add(desc+assign[i]*d,u,d);
        
      }
      free(u);
      
    } else if(flags==10) {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j];
      }
      
      for(i=0;i<k;i++) 
        fvec_normalize(desc+i*d,d,2.0);   
      
    } else if(flags==13) {

      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=sqr(v[i*d+j]-centroids[assign[i]*d+j]);
      }     

    } else if(flags==14) {
      float *avg=fvec_new_0(k*d);
      
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          avg[assign[i]*d+j]+=v[i*d+j]-centroids[assign[i]*d+j];

      int *hist=ivec_new_histogram(k,assign,n);

      for(i=0;i<k;i++) 
        if(hist[i]>0) 
          for(j=0;j<d;j++) 
            avg[i*d+j]/=hist[i];

      free(hist);

      fvec_0(desc,k*d);
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=sqr(v[i*d+j]-centroids[assign[i]*d+j]-avg[assign[i]*d+j]);
      
      fvec_sqrt(desc,k*d);
      
      free(avg);
    }  else if(flags==15) {
      fvec_0(desc,k*d*2);
      float *sum=desc;
      
      for(i=0;i<n;i++) 
        for(j=0;j<d;j++) 
          sum[assign[i]*d+j]+=v[i*d+j]-centroids[assign[i]*d+j];

      int *hist=ivec_new_histogram(k,assign,n);

      float *mom2=desc+k*d;

      for(i=0;i<n;i++) {
        int ai=assign[i];
        for(j=0;j<d;j++) 
          mom2[ai*d+j]+=sqr(v[i*d+j]-centroids[ai*d+j]-sum[ai*d+j]/hist[ai]);
      }
      fvec_sqrt(mom2,k*d);
      free(hist);
    
      
    } else if(flags==17) {
      fvec_0(desc,k*d*2);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) {
          float diff=v[i*d+j]-centroids[assign[i]*d+j];
          if(diff>0) 
            desc[assign[i]*d+j]+=diff;
          else 
            desc[assign[i]*d+j+k*d]-=diff;
        }
      }
  
    } else {
      fvec_0(desc,k*d);
      
      for(i=0;i<n;i++) {
        for(j=0;j<d;j++) 
          desc[assign[i]*d+j]+=v[i*d+j]-centroids[assign[i]*d+j];
      }
      
      
      if(flags==1) {
        int *hist=ivec_new_histogram(k,assign,n);
        /* printf("unbalance factor=%g\n",ivec_unbalanced_factor(hist,k)); */
        
        for(i=0;i<k;i++) 
          for(j=0;j<d;j++) 
            desc[i*d+j]/=hist[i];    
        
        free(hist);
      }
      
      if(flags==2) {
        for(i=0;i<k;i++) 
          fvec_normalize(desc+i*d,d,2.0);
      }
      
      if(flags==3 || flags==4) {
        assert(!"not implemented");
      }

      if(flags==16) {
        int *hist=ivec_new_histogram(k,assign,n);
        for(i=0;i<k;i++) if(hist[i]>0) {
          fvec_norm(desc+i*d,d,2);
          fvec_mul_by(desc+i*d,d,sqrt(hist[i]));
        }
        free(hist);
      }
   

    }
    free(assign);
  } else if(flags==11 || flags==12) {
    int a,ma=flags==11 ? 4 : 2;
    int *assign=ivec_new(n*ma);

    float *dists=knn(n,k,d,ma,centroids,v,assign,NULL,NULL);    

    fvec_0(desc,k*d);

    for(i=0;i<n;i++) {
      for(j=0;j<d;j++) 
        for(a=0;a<ma;a++) 
          desc[assign[ma*i+a]*d+j]+=v[i*d+j]-centroids[assign[ma*i+a]*d+j];
    } 
    
    free(dists);

    free(assign);
  }

}
