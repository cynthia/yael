addpath('../../matlab')


a=rand(100,100);

a=a+a'; % hope it's definite positive

[vecs_ref, vals_ref ] = eigs(a,8)
[vecs_new, vals_new ] = yael_eigs(single(a),8)


