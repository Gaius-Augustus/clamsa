#!/usr/bin/bash
#
# This is an example prediction for the sitewise classification of selection of the 10 Alignments in ..examples/dNdS_class_alignments/
# with pre-trained weights.
# Predictions are output in a csv and a pkl file.

# Models are specified with --model_ids { "label1" : "ID1", "label2" : "ID2, ...},
# where
# - The label is arbitrary, it will merely become the tile of the column in the output table
# - The ID is from training
#         - if you use pre-trained parameters, it could be "default", for example
#         - if you trained yourself, it is the time stamp used for storing weights (see train.sh)
# - A common choice may be --model_ids '{ "clamsa" : "default" }' for a single prediction each
#   with the default parameters.



python ../clamsa.py predict fasta \
   amniota_class_data.lst \
   --clades ../examples/amniota.nwk \
   --use_codons \
   --model_ids '{ "tcmc_dNdS_class" : "2023.09.21--12.07.25" }' \
   --out_csv dNdS_class_amniota.csv \
   --sitewise \
   --classify

# takes about 30 seconds on 3000 MSAs

#
# After you have trained ClaMSA yourself as in train.sh
# look up the timestamp from the log file or saved_weights directory and predict with the
# self trained run like this
# --model_ids '{ "my_train_run" : "2021.03.18--15.17.25" }'


# A warning like
# WARNING:tensorflow:Skipping loading of weights for layer P_sequence_columns due to mismatch in shape ((21,) vs (53,)).
# can be ignored. This is about a mismatch between the number of leaves in the prediction clade and the training forest ( = set of training trees)
