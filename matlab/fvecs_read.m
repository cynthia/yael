% Read a set of vectors stored in the fvec format (int + n * float)
% The function returns a set of output vector (one vector per column)
%
% Syntax: 
%   v = fvecs_read (filename)     -> read all vectors
%   v = fvecs_read (filename, n)  -> read n vectors 
function v = fvecs_read (filename, maxn)

% open the file and count the number of descriptors
fid = fopen (filename, 'rb');
 
if fid == -1
  error ('I/O error : Unable to open the file %s\n', filename)
end

% Read the vector size
d = fread (fid, 1, 'int');

if nargin < 2
  fseek (fid, 0, 1);
  n = ftell (fid) / (1 * 4 + d * 4);
else
  n = maxn;
end
  
fseek (fid, 0, -1);

% first read the meta information associated with the descriptor
v = fread (fid, (d + 1) * n, 'float=>single');
v = reshape (v, d + 1, n);
v = v (2:end, :);

fclose (fid);
