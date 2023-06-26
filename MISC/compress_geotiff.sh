#! /bin/bash

set -e
DIRNAME_INPUT=$(dirname $1)
TEMPNAME=$DIRNAME_INPUT/$(basename $(mktemp -u XXXXXXXX)).tif

gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" $1 $TEMPNAME

rm $1
mv $TEMPNAME $1
