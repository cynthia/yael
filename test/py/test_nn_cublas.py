import time


from yael import yael

na=10000
nb=700
d=128


yael.common_srandom(1)

a=yael.fvec_new_rand(na*d)
b=yael.fvec_new_rand(nb*d)

assign=yael.ivec(na)

# warm up
yael.nn(na,nb,d,b,a,assign,None)

t0=time.time()

yael.nn(na,nb,d,b,a,assign,None)

print "t=%.3f ms"%((time.time() - t0)*1000)

print "1st res = ",[assign[i] for i in range(20)]


##void nn (int n, int nb, int d, 
##         const float *b, const float *v,
##         int *assign,                                              
##         void (*peek_fun) (void *arg,double frac),
##         void *peek_arg);


