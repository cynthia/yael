#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <assert.h>


#include "machinedeps.h"


#ifdef __linux__

#define __USE_GNU
#include <sched.h>

int count_cpu (void)
{
  cpu_set_t set;
  sched_getaffinity (0, sizeof (cpu_set_t), &set);
  int i, count = 0;
  for (i = 0; i < CPU_SETSIZE; i++)
    if (CPU_ISSET (i, &set))
      count++;
  return count;
}


#elif defined(__APPLE__)

#include <sys/types.h>
#include <sys/sysctl.h>


int count_cpu (void) {
  int count=-1;
  size_t count_size=sizeof(count);
  sysctlbyname("hw.ncpu",&count,&count_size,NULL,0);
  return count;
}

#else

int count_cpu() {
  return 1;
}


#endif

#ifndef __APPLE__

double log2(double x) {
  return log(x)/M_LN2;
}


#endif

#ifndef __linux__
void *memalign (size_t ignored, size_t nbytes)
{
  return malloc (nbytes);
}
#endif

#if defined(__APPLE__) && defined(_LP64)

#warning "warn: using bugfix sgemm for Mac 64 bit"

#define real float
#define integer int



int sgemm_bugfix (char *transa, char *transb, integer * pm, integer *
            pn, integer * pk, real * palpha, const real * a, integer * plda,
            const real * b, integer * pldb, real * pbeta, real * c,
            integer * pldc) {
  assert(transa[0]=='T' && transb[0]=='N');

  int na=*pm,nb=*pn,d=*pk;
  int lda=*plda,ldb=*pldb,ldc=*pldc;
  float alpha=*palpha,beta=*pbeta;

  int i,j,k;
  
  for(i=0;i<na;i++) for(j=0;j<nb;j++) {
    double accu=0;
    for(k=0;k<d;k++) 
      accu+=a[k+lda*i]*b[k+ldb*j];
    c[i+j*ldc]=beta*c[i+j*ldc]+alpha*accu;
  }
  
}




#endif


#ifdef __linux__

typedef struct {
  void *ptr;
  size_t size;
} alloc_block_t;


static struct {
  int enabled;
  malloc_stats_t s;
  
  alloc_block_t *blocks;
  
  int n,na;  

  /* stored ref functions */

  void (*real_free) (void *__ptr, const __malloc_ptr_t);
  void *(*real_malloc) (size_t __size, const __malloc_ptr_t);
  void *(*real_realloc) (void *__ptr, size_t __size, const __malloc_ptr_t);
  void *(*real_memalign) (size_t __alignment, size_t __size, const __malloc_ptr_t);
  
} msc={0};


static void *collector_memalign (size_t alignment, size_t size, const __malloc_ptr_t f);
static void *collector_malloc (size_t size, const __malloc_ptr_t f);
static void collector_free (void *ptr, const __malloc_ptr_t f);
static void *collector_realloc (void *ptr_in, size_t size, const __malloc_ptr_t f);



#define SET_MALLOC_HOOK(fname) __##fname##_hook=collector_##fname
#define UNSET_MALLOC_HOOK(fname) __##fname##_hook=msc.real_##fname

#define SET_MALLOC_HOOKS \
  SET_MALLOC_HOOK(free);                        \
  SET_MALLOC_HOOK(malloc);                      \
  SET_MALLOC_HOOK(realloc);                     \
  SET_MALLOC_HOOK(memalign);


#define UNSET_MALLOC_HOOKS                      \
  UNSET_MALLOC_HOOK(free);                      \
  UNSET_MALLOC_HOOK(malloc);                    \
  UNSET_MALLOC_HOOK(realloc);                   \
  UNSET_MALLOC_HOOK(memalign);


static void *collector_memalign (size_t alignment, size_t size, const __malloc_ptr_t f) {
  UNSET_MALLOC_HOOKS;
  void *ptr; 

  if(alignment==1) {
    ptr=malloc(size);
  } else {
    ptr=memalign(alignment,size);
  }  
  msc.s.n_alloc++;
  msc.s.delta_alloc+=size;
  if(msc.s.delta_alloc>msc.s.max_alloc) 
    msc.s.max_alloc=msc.s.delta_alloc;
  
  if(msc.n>=msc.na) {
    msc.na=msc.na<8 ? 8 : msc.na*2;
    msc.blocks=realloc(msc.blocks,sizeof(*msc.blocks)*msc.na);
    assert(msc.blocks);
  }
  
  msc.blocks[msc.n].ptr=ptr;
  msc.blocks[msc.n].size=size;
  msc.n++;  

  SET_MALLOC_HOOKS;
  return ptr;
}

static void *collector_malloc (size_t size, const __malloc_ptr_t f) {
  return collector_memalign(1,size,f);
}


static void collector_free (void *ptr, const __malloc_ptr_t f) {
  
  UNSET_MALLOC_HOOKS;

  free(ptr);

  msc.s.n_free++;
  
  /* find where the block is */
  int i;
  for(i=msc.n-1;i>=0;i--) 
    if(msc.blocks[i].ptr==ptr) break;

  if(i<0) {
    msc.s.n_untracked_frees++;
  } else {
    msc.s.delta_alloc-=msc.blocks[i].size;
    msc.blocks[i]=msc.blocks[msc.n-1];
    msc.n--;
  }

  SET_MALLOC_HOOKS;

}



static void *collector_realloc (void *ptr_in, size_t size, const __malloc_ptr_t f) {
  
  UNSET_MALLOC_HOOKS

  void *ptr=realloc(ptr_in,size);

  /* find where the block is */
  int i;
  for(i=msc.n-1;i>=0;i--) 
    if(msc.blocks[i].ptr==ptr_in) break;

  msc.s.n_realloc++;
  
  if(i<0) {
    msc.s.n_untracked_frees++;
  } else {
    msc.s.delta_alloc-=msc.blocks[i].size+size;

    if(msc.s.delta_alloc>msc.s.max_alloc) 
      msc.s.max_alloc=msc.s.delta_alloc;

    msc.blocks[i].ptr=ptr;
  }

  SET_MALLOC_HOOKS

  return ptr;
}


#define GET_MALLOC_HOOK(fname) msc.real_##fname=__##fname##_hook; SET_MALLOC_HOOK(fname)



void malloc_stats_begin() {
  assert(!msc.enabled || "malloc_stats_begin: collector enabled already");
  msc.enabled=1;

  memset(&msc.s,0,sizeof(malloc_stats_t));
  msc.n=msc.na=0;
  msc.blocks=NULL;
  
  printf("initial hooks: %p %p %p %p\n",
         __free_hook,
         __malloc_hook,
         __realloc_hook,
         __memalign_hook);
         

  GET_MALLOC_HOOK(free);
  GET_MALLOC_HOOK(malloc);
  GET_MALLOC_HOOK(realloc);
  GET_MALLOC_HOOK(memalign);
      
}


malloc_stats_t malloc_stats_end() {
  assert(msc.enabled || "malloc_stats_begin: collector not enabled");
  msc.enabled=0;
  
  UNSET_MALLOC_HOOKS;
  
  free(msc.blocks);

  return msc.s;
}

#else

void malloc_stats_begin() {
  /* not implemented */
}

malloc_stats_t malloc_stats_end() {
  malloc_stats_t s;
  memset(&s,0,sizeof(malloc_stats_t));
  return s;
}


#endif 
