#! /bin/bash

set -e
DIRNAME_INPUT=$(dirname $1)
TEMPNAME=$DIRNAME_INPUT/$(basename $(mktemp -u XXXXXXXX))

h5repack -v -f GZIP=5 $1 $TEMPNAME

#mv $1 $1.old
rm $1
mv $TEMPNAME $1
