#!/bin/bash

# need an octave version that supports float arrays

octavedir=/home/lear/douze/scratch2/installed_fc10_64

function mkoct_disable () {
    n=$1
    shift
    os="$@"
    $octavedir/bin/mkoctfile -v --mex  $n.c -I../yael $os
}

function mkoct () {
    n=$1
    shift
    os="$@"
    
    # link only with what is necessary

    set -x 
    gcc -c -fPIC -I$octavedir/include/octave-3.2.3 -I$octavedir/include/octave-3.2.3/octave -g -O2 -I../yael $n.c
    
    g++ -shared -Wl,-Bsymbolic -o $n.mex $n.o $os
    set +x
}


mkoct kmeans_fast ../yael/{binheap,nn,vector,machinedeps,kmeans,sorting}.o
mkoct nn ../yael/{binheap,nn,vector,machinedeps,sorting}.o

