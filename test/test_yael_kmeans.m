addpath ('../matlab')

k = 100;                   % number of cluster to create
d = 128;                   % dimensionality of the vectors
n = 100000;                % number of vectors
nt = 2;                    % number of threads to use
v = single (randn (d, n)); % random set of vectors 
niter = 30;

[centroids, dis, assign] = yael_kmeans (v, k, 'nt', nt, 'niter', niter);

