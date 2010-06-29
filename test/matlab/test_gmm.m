addpath ('../../matlab')

k = 100;                   % number of cluster to create
d = 128;                   % dimensionality of the vectors
n = 10000;                 % number of vectors
v = single(rand (d, n));   % random set of vectors 
niter = 30;                % maximum number of iterations
verbose = 2;               % verbosity level
seed = 3;                  % 0: no seeding, values > 0 are used as seed
nt = 1;                    % to force multi-threading if not done by Matlab/octave
                           % check if multithreaded actived with nt=1 before 
			   % changing this variable
			   

tic
[w, mu, sigma] = yael_gmm (v, k, 'niter', niter, ...
			   'verbose', verbose, 'seed', seed);
toc
