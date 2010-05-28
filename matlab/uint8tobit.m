% This function takes a uint8 vector as input and construct a binary vector in output
% The vectors are line-stored
function b = uint8tobit (v)

n = size (v, 2);
dbytes = size (v, 1);
d = dbytes * 8;

b = zeros(d, n, 'uint8');

for i = 1:n
  for j = 1:dbytes
  b((j-1)*8+1:j*8 ,i) = bitget (v(j, i), 1:8);  
  end
end
 