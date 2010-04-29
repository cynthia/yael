#!/bin/bash

# Customize makefile.inc

# Detect architecture
if [ `uname` == Darwin ] ; then
  conf=mac
elif [ `uname -m` == 'x86_64' ] ; then
  conf=linux64
elif  [ `uname -m` == 'i686' ] || [ `uname -m` == 'i386' ]  ; then
  conf=linux32
fi


# Defaults parameters (includes ones set on cmd line)
cflags="-fPIC -Wall -g -O3 $CFLAGS"
ldflags="-g -fPIC $LDFLAGS"

yaelprefix=$PWD

lapackldflags='-lblas -llapack'

# for some reason, the atlas libs are not accessible via -lblas
# -llapack on Ubuntu and Fedora Core Linux.

if [ -e /usr/lib/libblas.so.3gf ]; then
   # special names for libs on ubuntu
   lapackldflags="/usr/lib/libblas.so.3gf /usr/lib/liblapack.so.3gf"
elif [ -d /usr/lib64/atlas/ ]; then 
   lapackldflags="/usr/lib64/atlas/libblas.so.3 /usr/lib64/atlas/liblapack.so.3"
elif [ -d /usr/lib/atlas/ ]; then
   lapackldflags="/usr/lib/atlas/libblas.so.3 /usr/lib/atlas/liblapack.so.3"
else
   echo -n "using default locations for blas and lapack"
fi


# by default Fortran integer is int (may be long for MKL used with mexa64)
lapackcflags="-DFINTEGER=int"

usearpack=no
arpackldflags=/usr/lib64/libarpack.so.2

useopenmp=""

# dynamic libs: force an install path so that the user does not need
# to set the LD_LIBRARY_PATH for yael

if [ $conf == mac ]; then
  wrapldflags="-Wl,-F. -bundle -undefined dynamic_lookup"
  sharedext=dylib
  sharedflags="-dynamiclib"
  yaelsharedflags="$sharedflags -install_name $yaelprefix/yael/libyael.dylib"
else
  wrapldflags="-shared"
  sharedflags="-shared"
  yaelsharedflags="$sharedflags"
  sharedext=so  
fi

function usage () {
    cat <<EOF 1>&2
usage: $0 
  [--debug] 
  [--yael=yaelprefix] 
  [--swig=swigpath]
  [--lapack=ld_flags_for_lapack_and_blas]
  [--enable-arpack]
  [--arpack=ld_flags_for_arpack]
  [--python-cflags=flags_to_compile_with_python_c_api]
  [--enable-numpy]
  [--numpy-cflags=includes-for-numpy]

Examples that work: 

Compile for 64-bit mac, with optimization
CFLAGS=-m64 LDFLAGS=-m64 ./configure.sh --python-cflags=-I$HOME/local64/include/python2.6


EOF
    exit $1
}


# Search latest python version (not 3.x !!!!)
for pysubver in {6,5,4,x} ; do
    if [ -f "/usr/include/python2.${pysubver}/Python.h" ] ; then
	echo "Found python development version 2.$pysubver"
	break
    fi
done

pythoncflags=-I/usr/include/python2.$pysubver

if [ "$pysubver" == "x" ] ; then
    echo "# Warn: no python directory (python-dev) found"
fi



# Parse command line arguments
while [ $# -gt 0 ] ; do

    a=$1
    shift

    # echo $a

    case $a in 
	-h | --help) usage 0 ;; 

	--debug)    cflags="${cflags/ -O3/}" ;;
	
	--yael=*)   yaelprefix=${a#*=};;
        --swig=*)   swig=${a#*=} ;;
	--lapack=*) lapackldflags=${a#*=} ;;
        --enable-arpack) usearpack=yes;;
	--arpack=*) arpackldflags=${a#*=} ;;
	--fortran-64bit-int) 
            lapackcflags="$lapackcflags -DFINTEGER=long" ;;       

        --python-cflags=*)     
            pythoncflags=${a#*=}
            ;;        

        --enable-numpy) 
            usenumpy=yes
            numpycflags="-I$( python -c 'import numpy; print numpy.get_include()' )"
            numpyswigflags="-DHAVE_NUMPY"
            ;;

        --numpy-cflags) 
            numpycflags=-I${a#*=}
            ;;

	*)  echo "unknown option $a" 1>&2; exit 1;;
    esac
done

yaellib=${yaelprefix}/yael
yaelinc=${yaelprefix}


if [ -z "$swig" ]; then
  if which swig ; then
    swig=swig
  else 
    echo "Error: no swig executable found. Provide one with --swig"
    exit 1
  fi
fi


cat <<EOF | tee makefile.inc

CFLAGS=$cflags
LDFLAGS=$ldflags

PYTHONCFLAGS = $pythoncflags

YAELCONF=$conf
YAELCFLAGS=-I$yaelinc
YAELLDFLAGS=-L$yaellib -Wl,-rpath,$yaellib -lyael


SWIG=$swig -python

WRAPLDFLAGS=$wrapldflags
LAPACKLDFLAGS=$lapackldflags
LAPACKCFLAGS=$lapackcflags

USEARPACK=$usearpack
ARPACKLDFLAGS=$arpackldflags

USETHREADS=yes
THREADCFLAGS=-DHAVE_THREADS
THREADLDFLAGS=-lpthread

SHAREDEXT=$sharedext
SHAREDFLAGS=$sharedflags
YAELSHAREDFLAGS=$yaelsharedflags

USENUMPY=$usenumpy
NUMPYCFLAGS=$numpycflags
NUMPYSWIGFLAGS=$numpyswigflags

EOF

