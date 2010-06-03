#!/bin/bash

# usage ./octave_mex.sh source.c [compile and link options]


# Yael requires an Octave version that supports float
# arrays. You can check this with 
#
#    a=single(4); class(a)
# 
# shoud report single (and not double).
# 
# Compiling and linking is done explicitly, because mkoctfile adds way
# too many useless libraries.
# 
# Modify octavedir and octavevers below to point to the Octave installation 


# set octave install directory (/usr if you are lucky)
octavedir=/home/lear/douze/scratch2/installed_fc10_64

# version number
octavevers=3.2.3



n=$1
n=${n%.c}
shift
os="$@"
    
cmd="gcc -c -fPIC -I$octavedir/include/octave-$octavevers -I$octavedir/include/octave-$octavevers/octave -g -O2  $n.c $os"
echo $cmd
$cmd
    
cmd="g++ -shared -Wl,-Bsymbolic -o $n.mex $n.o $os"
echo $cmd
$cmd
