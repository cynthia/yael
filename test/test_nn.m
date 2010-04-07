% compute the nearest neighbors 
% If the mex-compiled version exists, it is automatically called. 
% Otherwise, the pure matlab implementation is used instead. 

addpath ('../matlab')

n = 100000;              % number of database vectors
nq = 100;                % number of queries
d = 128;                 % dimension of the vectors
k = 5;                   % number of nearest neighbors to retrieve
nt = 2;                  % number of processor cores to be used
v = single (rand(d,n));  % base vectors
q = single (rand(d,nq)); % query vectors

tic
[idx, dis] = yael_nn (v, q, k, nt);
toc
