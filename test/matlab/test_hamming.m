% Test program for Hamming distance computation
% The mex-file yael_hamming.c should be compiled, otherwise this test program
% invokes the yael_hamming.m function, which is super-slow

nbits = 64;
d = nbits / 8;
na = 1000;
nb = 1000000;

% Generate some random bit vectors uniformely
a = uint8 (randi (256, d, na) - 1);
b = uint8 (randi (256, d, nb) - 1);

tic
dis = yael_hamming (a, b);
toc
