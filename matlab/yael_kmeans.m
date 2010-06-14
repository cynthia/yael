% This function performs the clustering of a set of vector in k clusters
% 
% C = yael_kmeans (V, k)
% [C, I] = yael_kmeans (v, k)
% [C, D, I] = yael_kmeans (v, k)
% [C, D, I, Nassign] = yael_kmeans (v, k)
% returns a set of k centroids, stored column-wise in C
% The input vectors are given in the matrix V (one vector per column)
% 
% Optionnally the function can returns
%   I: the cluster index associated with each input vector, 
%   D: the square distance D between each vector and its centroid,
%   Nassign: the total number of centroids assigned to each cluster
% 
% Options: 
% C = yael_kmeans (V, k, 'redo', redo, 'verbose', verbose, 'nt', nt, 'seed', seed)
% where 
%    redo       number of times the k-means is run (best clustering returned)
%    verbose    the verbosity level. 0: no output, 1 (default), 2: detailled
%    nt         number of threads. For octave users. 
%               Warning: nt=1 (default) should provide 
%               multi-threading depending on matlab version or architecture. 
%               Warning: do not use nt>1 in that case, at it will cause 
%               memory leaks
%    seed       0 by default. Specify a value !=0 to randomize initalization
