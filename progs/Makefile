.SUFFIXES: .c .o .swg 

include ../makefile.inc

CC=gcc 

all: kmeans fvecfile



ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif


.c.o:
	$(CC) $(CFLAGS) -c $< -o $@ $(FLAGS) $(EXTRACFLAGS) $(YAELCFLAGS)

kmeans: kmeans.o 
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(YAELLDFLAGS)

fvecfile: fvecfile.o 
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) 



#############################
# Dependencies  

# for i in *.c ; do cpp -I.. -MM $i; done

fvecfile.o: fvecfile.c
kmeans.o: kmeans.c ../yael/vector.h ../yael/machinedeps.h \
  ../yael/kmeans.h
clean:
	rm -f  *.o kmeans fvecfile

