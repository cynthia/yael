# makefile to build the mex-file of the yael matlab interface
# Require to construct a yael archive specifically adapted to yael 

CC=gcc 

all: yael4matlab.a yael_kmeans yael_nn

include ../makefile.inc

ifeq "$(YAELCONF)" "linux64"
    CFLAGS=-O3 -DFINTEGER=long -fPIC -m64 -I..
else 
    CFLAGS=-O3 -DFINTEGER=int -fPIC -m64 -I..
endif

LDFLAGS=-lmwblas


LIBOBJ=vector.o nn.o kmeans.o sorting.o binheap.o machinedeps.o vlad.o

yael4matlab.a: 
	gcc -c $(CFLAGS) ../yael/binheap.c
	gcc -c $(CFLAGS) ../yael/kmeans.c
	gcc -c $(CFLAGS) ../yael/machinedeps.c
	gcc -c $(CFLAGS) ../yael/nn.c
	gcc -c $(CFLAGS) ../yael/sorting.c
	gcc -c $(CFLAGS) ../yael/vector.c
	gcc -c $(CFLAGS) ../yael/vlad.c
	ar r yael4matlab.a $(LIBOBJ)
	rm *.o


yael_kmeans: yael_kmeans.c
	mex yael_kmeans.c -I.. yael4matlab.a $(LDFLAGS)

yael_nn: yael_nn.c
	mex yael_nn.c -I.. yael4matlab.a $(LDFLAGS)

clean:
	rm *.a *.mexa64 *.mex *.o