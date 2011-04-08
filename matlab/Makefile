# makefile to build the mex-file of the yael matlab interface
# Require to construct a yael archive specifically adapted to yael 

CC=gcc 


include ../makefile.inc

# Here you can chose between Matlab and Octave
TARGET=matlab
#TARGET=octave


ifeq "$(TARGET)" "matlab"


    # Matlab on Linux 64 bit is linked with a BLAS version where 
    # Fortran integers are 64 bit, so Yael has to be recompiled

    ifeq "$(YAELCONF)" "linux64"
        CFLAGS=-O3 -DFINTEGER=long -D_YAEL4MATLAB_ -fPIC -m64 -g -I..
    else 
    ifeq "$(YAELCONF)" "mac64"
        CFLAGS=-O3 -DFINTEGER=long -D_YAEL4MATLAB_ -fPIC -m64 -g -I..
    else 
        CFLAGS=-O3 -DFINTEGER=int -D_YAEL4MATLAB_ -fPIC -m64 -g -I.. # should be int in 32-bit mode
    endif
    endif


    YAELAR=yael4matlab.a
    MEX=mex

    LDFLAGS=-g $(YAELAR) -lmwblas -lmwlapack -lmwarpack

else

    # modify this script to compile code to a octave-compilant mex file
    MEX=./octave_mex.sh
    YAELAR=yael4octave.a
    LDFLAGS=$(YAELAR)
endif


all: $(YAELAR) yael_kmeans yael_kmeans_init_user yael_gmm yael_fisher yael_nn yael_kmin yael_kmax yael_fvecs_normalize yael_L2sqr


LIBOBJ=vector.o matrix.o eigs.o nn.o kmeans.o gmm.o sorting.o binheap.o machinedeps.o vlad.o
YAELSRC=../yael/binheap.c ../yael/eigs.c ../yael/gmm.c ../yael/kmeans.c ../yael/machinedeps.c ../yael/matrix.c ../yael/nn.c ../yael/sorting.c ../yael/vector.c ../yael/vlad.c

binheap.o: ../yael/binheap.c ../yael/binheap.h
	gcc -c $(CFLAGS) ../yael/binheap.c

eigs.o: ../yael/eigs.c ../yael/eigs.h
	gcc -c $(CFLAGS) ../yael/eigs.c

gmm.o: ../yael/gmm.c ../yael/gmm.h
	gcc -c $(CFLAGS) ../yael/gmm.c

kmeans.o: ../yael/kmeans.c ../yael/kmeans.h
	gcc -c $(CFLAGS) ../yael/kmeans.c

machinedeps.o: ../yael/machinedeps.c ../yael/machinedeps.h
	gcc -c $(CFLAGS) ../yael/machinedeps.c

matrix.o: ../yael/matrix.c ../yael/matrix.h
	gcc -c $(CFLAGS) ../yael/matrix.c

nn.o: ../yael/nn.c ../yael/nn.h
	gcc -c $(CFLAGS) ../yael/nn.c 

sorting.o: ../yael/sorting.c ../yael/sorting.h
	gcc -c $(CFLAGS) ../yael/sorting.c

vector.o: ../yael/vector.c ../yael/vector.h
	gcc -c $(CFLAGS) ../yael/vector.c

vlad.o: ../yael/vlad.c ../yael/vlad.h
	gcc -c $(CFLAGS) ../yael/vlad.c

yael4matlab.a: $(LIBOBJ)
	ar r yael4matlab.a $(LIBOBJ)
	rm *.o

yael4octave.a: 
	ar r $@ $(addprefix ../yael/,$(LIBOBJ))


yael_kmeans: yael_kmeans.c $(YAELAR)
	$(MEX) yael_kmeans.c -I.. $(LDFLAGS)

yael_kmeans_init_user: yael_kmeans_init_user.c $(YAELAR)
	$(MEX) yael_kmeans_init_user.c -I.. $(LDFLAGS)

yael_gmm: yael_gmm.c $(YAELAR)
	$(MEX) yael_gmm.c -I.. $(LDFLAGS)

yael_fisher: yael_fisher.c $(YAELAR)
	$(MEX) yael_fisher.c -I.. $(LDFLAGS)

yael_nn: yael_nn.c $(YAELAR)
	$(MEX) yael_nn.c -I.. $(LDFLAGS)

yael_kmin: yael_kmin.c $(YAELAR)
	$(MEX) yael_kmin.c -I.. $(LDFLAGS)

yael_kmax: yael_kmax.c $(YAELAR)
	$(MEX) yael_kmax.c -I..  $(LDFLAGS)

yael_fvecs_normalize: yael_fvecs_normalize.c $(YAELAR)
	$(MEX) yael_fvecs_normalize.c -I.. $(LDFLAGS)

yael_L2sqr: yael_L2sqr.c $(YAELAR)
	$(MEX) yael_L2sqr.c -I.. $(LDFLAGS)

clean:
	rm -f  *.a *.mexa64 *.mexmaci64 *.mex *.o *~
