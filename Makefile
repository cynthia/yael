.SUFFIXES: .c .o .swg 

include makefile.inc

CC=gcc 

all: test_pca test_binheap test_kmeans test_eigs test_spectral_clustering \
     test_gmm \
     libyael.a libyael.$(SHAREDEXT) _yael.so




ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif


#############################
# Various  

LIBOBJ=matrix.o vector.o nn.o clustering.o kmeans.o gmm.o eigs.o \
	spectral_clustering.o sorting.o binheap.o machinedeps.o vlad.o

libyael.a: $(LIBOBJ)
	ar r libyael.a $^

libyael.$(SHAREDEXT): $(LIBOBJ)
	gcc $(LDFLAGS) $(YAELSHAREDFLAGS) -o libyael.$(SHAREDEXT) $^ $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@ $(FLAGS) $(EXTRACFLAGS)

nn.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(THREADCFLAGS)

matrix.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(THREADCFLAGS) $(EXTRAMATRIXFLAG)

clustering.o: EXTRACFLAGS=$(THREADCFLAGS)

kmeans.o: EXTRACFLAGS=$(THREADCFLAGS)

gmm.o: EXTRACFLAGS=$(THREADCFLAGS) $(LAPACKCFLAGS)

eigs.o: EXTRACFLAGS=$(LAPACKCFLAGS) $(LAPACKCFLAGS)

yael_wrap.o: EXTRACFLAGS=$(PYTHONCFLAGS) $(NUMPYCFLAGS)

test_pca: test_pca.o vector.o matrix.o sorting.o binheap.o machinedeps.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG)

test_binheap: test_binheap.o binheap.o sorting.o vector.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) 

test_kmeans: test_kmeans.o libyael.a
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) 

test_gmm: test_gmm.o libyael.a
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) 

test_eigs: test_eigs.o libyael.a
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) 

test_spectral_clustering: test_spectral_clustering.o libyael.a
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG)

mmap_and_lock: mmap_and_lock.c
	$(CC) $(CFLAGS) -o $@ $<



#############################
# Python wrapper 


yael_wrap.c : yael.swg common.swg nn.h clustering.h kmeans.h gmm.h eigs.h \
	sorting.h matrix.h vector.h machinedeps.h vlad.h
	$(SWIG) -I.. $(NUMPYSWIGFLAGS) $< 

_yael.so: yael_wrap.o libyael.$(SHAREDEXT)
	$(CC) $(LDFLAGS) -o $@ $(WRAPLDFLAGS) $^ $(PYTHONLDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(YAELLDFLAGS)

#############################
# Dependencies  

# for i in *.c ; do cpp -I.. -MM $i; done

clustering.o: clustering.c clustering.h vector.h nn.h machinedeps.h \
  sorting.h
kmeans.o: kmeans.c vector.h kmeans.h nn.h
gmm.o: gmm.c vector.h matrix.h kmeans.h nn.h sorting.h
fvecfile.o: fvecfile.c
machinedeps.o: machinedeps.c machinedeps.h
matrix.o: matrix.c vector.h matrix.h machinedeps.h sorting.h
nn.o: nn.c vector.h nn.h sorting.h machinedeps.h
sorting.o: sorting.c sorting.h machinedeps.h
test_pca.o: test_pca.c vector.h matrix.h
timer.o: timer.c timer.h
vector.o: vector.c vector.h
binheap.o: binheap.c binheap.h sorting.h
spectral_clustering.o: spectral_clustering.c nn.h eigs.h vector.h \
  matrix.h
test_spectral_clustering.o: test_spectral_clustering.c \
  spectral_clustering.h vector.h
test_binheap.o: test_binheap.c binheap.h 
test_kmeans.o: test_kmeans.c kmeans.h 
test_gmm.o: test_gmm.c vector.h machinedeps.h gmm.h
test_eigs.o: test_eigs.c eigs.h 
vlad.o: vlad.c vlad.h vector.h nn.h sorting.h


indent: generic_hash_table.c fast_alloc.c
	$(INDENT) $^

clean:
	rm -f libyael.a libyael.$(SHAREDEXT)* *.o yael.py *.pyc yael_wrap.c _yael.so \
		test_pca test_binheap test_kmeans test_gmm test_eigs test_spectral_clustering fvecfile

