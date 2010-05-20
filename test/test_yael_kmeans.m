addpath ('../matlab')

k = 1000;                   % number of cluster to create
d = 128;                    % dimensionality of the vectors
n = 1000000;                % number of vectors
nt = 4;                     % number of threads to use
v = rand (d, n, 'single');  % random set of vectors 
niter = 30;

tic
[centroids, dis, assign] = yael_kmeans (v, k, 'nt', nt, 'niter', niter);
toc
