#!/usr/bin/bash
# shell script to create a UCSC custrom tracks file from 6 wiggle files 

STEM=$1
POS=$2

OUTFNAME="$STEM.track"

echo "browser position $POS" > $OUTFNAME

for F in `seq 0 5`; do
  if [ $F -lt 3 ]; then
     STRAND="+"
     FRAME=$F
  else
     STRAND="-"
     FRAME=$(($F-3))
  fi
  echo "track type=wig name=\"ClaMSA f=$FRAME strand=$STRAND\" description=\"evolutionary coding signal\" visibility=2 color=0,128,0" >> $OUTFNAME
  cat $STEM$F.wig >> $OUTFNAME 
done
