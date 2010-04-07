addpath ('../matlab')

k = 100;                   % number of cluster to create
d = 128;                   % dimensionality of the vectors
n = 100000;                % number of vectors
nt = 2;                    % number of threads to use
v = single (randn (d, n)); % random set of vectors 

[centroids, dis, assign] = yael_kmeans (v, 100, 'nt', 2, 'niter', 25);

