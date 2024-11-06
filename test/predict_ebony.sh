#!/usr/bin/bash

# This is an example prediction for splice sites in ../examples/ebony/splice_sites.aug.out with pre-trained ebony on splice sites
# Predictions are output in CSV format


../clamsa.py predict augustus ../examples/ebony/splice.predict.aug \
     --clades ../examples/ebony/diverse32mammals.nwk \
     --saved_weights_basedir ../saved_weights/ \
     --model_ids '{ "ebony" : "ebony_splice" }' \
     --num_classes 2 \
     | tee ebony.out
