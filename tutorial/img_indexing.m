addpath ('~/src/yael/matlab/');

k = 2048;                 % In practice, we would rather choose k=100k
nbits = 128;              % Typical values are 32, 64 or 128 bits

% Images and descriptors are assumed stored in the following directory
dir_data = './holidays_100/';

% Retrieve the image list and load the images and SIFT
img_list = dir ([dir_data '/*.jpg']);
nimg = numel(img_list);


tic 
imgs = arrayfun (@(x) (imread([dir_data x.name])), img_list, 'UniformOutput', false) ;
fprintf ('* Loaded images in %.3f seconds\n', toc); tic

sifts = arrayfun (@(x) (siftgeo_read([dir_data strrep(x.name, '.jpg', '.siftgeo')])), ...
                        img_list, 'UniformOutput', false) ; 
fprintf ('* Loaded descriptors in %.3f seconds\n', toc); tic

sifts = cellfun (@(x) (yael_vecs_normalize(sign(x).*sqrt(abs(x)))), ...
                        sifts, 'UniformOutput', false) ;
fprintf ('* Convert to RootSIFT in %.3f seconds\n', toc); tic


% Here we learn it on Holidays itself to avoid 
% requiring another dataset. but note
% that this is generally considered as bad practice and should be avoided.

vtrain = [sifts{:}];
vtrain = vtrain (:, 1:2:end);


C = yael_kmeans (vtrain, k, 'niter', 10); 
fprintf ('* Learned a visual vocabulary C in %.3f seconds\n', toc); tic

% We provide the codebook and the function that performs the assignment,  
% here it is the exact nearest neighbor function yael_nn

ivfhe = yael_ivf_he (k, nbits, vtrain, @yael_nn, C);
fprintf ('* Learned the Hamming Embedding structure in %.3f seconds\n', toc); tic

% Construct a query 



% Save ivf
fivf_name = cfg.ivf_fname;
fprintf ('* Save the inverted file to %s\n', fivf_name);
ivfhe.save (ivfhe, fivf_name);

fprintf ('* Free the inverted file\n');
% Free the variables associated with the inverted file
yael_ivf ('free');
clear ivfhe;


figure(1);

hax = axes('Position', [.35, .35, .3, .3]);
imagesc(imgs{42}); axis off image


subplot(2,5,[1 2]), imagesc(imgs{1}); title ('Query'); axis off image

subplot(2,5,7), imagesc(imgs{2}); axis off image
subplot(2,5,3), imagesc(imgs{3}); axis off image
subplot(2,5,4), imagesc(imgs{5}); axis off image
subplot(2,5,5), imagesc(imgs{5}); axis off image


subplot(2,5,6), imagesc(imgs{10}); axis off image

figure(2)

wh = 0.18;
whi = 0.19;

axes('Position', [0, 0, wh, wh]);
 imagesc(imgs{1}); axis off image
 
axes('Position', [1-wi*4, 0, wh, wh]);
 imagesc(imgs{2}); axis off image
 
axes('Position', [1-wi*3, 0, wh, wh]);
 imagesc(imgs{3}); axis off image

 axes('Position', [1-wi*2, 0, wh, wh]);
 imagesc(imgs{2}); axis off image
 
axes('Position', [1-wi, 0, wh, wh]);
 imagesc(imgs{3}); axis off image
 



