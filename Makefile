.SUFFIXES: .c .o .swg 

include makefile.inc

CC=gcc 

all: libyael.a libyael.$(SHAREDEXT) _yael.so




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

binheap.o: binheap.c binheap.h sorting.h
clustering.o: clustering.c clustering.h vector.h nn.h machinedeps.h \
  sorting.h kmeans.h
eigs.o: eigs.c vector.h sorting.h machinedeps.h
gmm.o: gmm.c vector.h matrix.h kmeans.h nn.h gmm.h sorting.h
kmeans.o: kmeans.c vector.h kmeans.h nn.h
machinedeps.o: machinedeps.c machinedeps.h
matrix.o: matrix.c vector.h matrix.h sorting.h machinedeps.h
nn.o: nn.c machinedeps.h vector.h nn.h binheap.h sorting.h
sorting.o: sorting.c sorting.h machinedeps.h binheap.h vector.h
spectral_clustering.o: spectral_clustering.c nn.h eigs.h vector.h \
  matrix.h kmeans.h spectral_clustering.h
vector.o: vector.c vector.h
vlad.o: vlad.c nn.h vector.h sorting.h


clean:
	rm -f libyael.a libyael.$(SHAREDEXT)* *.o yael.py *.pyc yael_wrap.c _yael.so 
