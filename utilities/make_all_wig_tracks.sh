#!/usr/bin/bash
# shell script to create a UCSC custrom tracks file from 6 wiggle files 

STEM=$1
SPECIES=$2

echo "${STEM}_$SPECIES"

OUTFNAME="$STEM_$SPECIES.track"

echo "browser" > $OUTFNAME

for F in `seq 0 2`; do
    for S in "plus" "minus"; do
	echo "track type=wig name=\"ClaMSA f=$F strand=$S\" description=\"evolutionary coding signal\" visibility=2 color=0,128,128" >> $OUTFNAME
	cat ${STEM}_${SPECIES}_${F}-${S}.wig >> $OUTFNAME
    done
done
