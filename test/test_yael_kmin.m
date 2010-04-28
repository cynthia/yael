% Test program for function test_yael_kmin, and timings comparison
d = 100000;
n = 100; 
k = 100;

a = single (rand (d, n)); 

% Just the minimal value
tic ; [v, i] = min(a) ; tmin = toc ;

% use the yael function to find the minimum only
tic; [val,idx] = yael_kmin (a, 1) ; tymin = toc;

% use to find the k smallest values
tic; [val,idx] = yael_kmin (a, k) ; tykmin = toc; 

% A standard matlab full sort (not good to find knn)
tic ; [val,idx] = sort(a); tsort = toc ;


fprintf (['Timings:\n  Matlab min search: %.3f\n  Yael min search:   %.3f\n' ...
	  '  Yael k-min search: %.3f\n  Matlab sort:       %.3f\n'], ... 
	 tmin, tymin, tykmin, tsort);
