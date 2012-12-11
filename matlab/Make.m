

setenv ('YAELCFLAGS', '-msse4 -fopenmp -I/usr/include -I.. -DFINTEGER=long -DHAVE_ARPACK');
setenv ('YAELLDFLAGS', '-lmwblas -lmwlapack -lmwarpack -fopenmp');

mex -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmeans.c ../yael/kmeans.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_nn.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmax.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c

mex -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmin.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c

mex -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_fvecs_normalize.c ../yael/vector.c

!rm *.o
