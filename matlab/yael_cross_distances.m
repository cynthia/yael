% Compute all the distances between two sets of vectors
%
% Usage: [dis] = dis_cross_distances(q, v, distype, nt)
%
% Parameters:
%   q, v        sets of vectors (1 vector per column)
%   distype     1=L1, 2=L2sqrt, 3=chi-square (default: 2)
%   nt          number of threads (not used for L2 distance)
%
% Returned values
%   dis         the corresponding distances
%               vectors of q corresponds to row, and columns for v
function dis = yael_cross_distances (q, v, distype, nt)

error ('This function is available only if compiled (Mex-file)');
