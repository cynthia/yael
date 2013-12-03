% Compilation parameters
openmp = false;   % De-activated by default since latest MacOS needs hacking to have it working
sse4 = true;
arpack = true;
bitvecsize = 128;
extraflags = '-Wall';

% Set flags
cflags = '-I/usr/include -I.. -DFINTEGER=long ';
ldflags = '-lmwblas -lmwlapack -lmwarpack ';

if openmp
  cflags = ['-fopenmp ' cflags ];
  ldflags = ['-fopenmp ' ldflags];
end

if sse4
  cflags = ['-msse4 ' cflags ];
end

if arpack
  cflags = [cflags '-DHAVE_ARPACK '];
  ldflags = [ldflags ' -lmwarpack'];
end

  
cflags = [cflags extraflags ' -DBITVECSIZE=' num2str(bitvecsize)];

  
fprintf ('cflags = %s\n', cflags);
fprintf ('ldflags = %s\n', ldflags);
setenv ('YAELCFLAGS', cflags);
setenv ('YAELLDFLAGS', ldflags);


mex -largeArrayDims -v -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmeans.c ../yael/kmeans.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_gmm.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_fisher.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_fisher_elem.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_nn.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_fvecs_normalize.c ../yael/vector.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmax.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_kmin.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_L2sqr.c ../yael/binheap.c ../yael/nn.c ../yael/vector.c  ../yael/machinedeps.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_cross_distances.c ../yael/binheap.c ../yael/nn.c ../yael/vector.c  ../yael/machinedeps.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_cross_distances.c ../yael/binheap.c ../yael/nn.c ../yael/vector.c  ../yael/machinedeps.c ../yael/sorting.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_hamming.c ../yael/hamming.c

mex -largeArrayDims -g CFLAGS="\$CFLAGS \$YAELCFLAGS" LDFLAGS="\$LDFLAGS \$YAELLDFLAGS" yael_ivf.c ../yael/ivf.c ../yael/hamming.c

mex siftgeo_read.c

delete *.o


