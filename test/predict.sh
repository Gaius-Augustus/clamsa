#!/usr/bin/bash
#
# This script contains an example run of ClaMSA to classify MSAs as coding or not.
# It computes for each alignment given in the list of fasta files vertebrate.test.msa.lst
# and for each ClaMSA model the probability that the MSA is coding under the model.
# Models are specified with --model_ids { "label1" : "ID1", "label2" : "ID2, ...},
# where
# - The label is arbitrary, it will become the tile of the column in the output table
# - The ID is from training
#         - if you use pre-trained parameters, it could be "default", for example
#         - if you trained yourself, it is the time stamp used for storing weights (see train.sh)
# - A common choice may be --model_ids '{ "clamsa" : "default" }' for a single prediction each
#   with the default parameters.

SAVED_WEIGHTS_DIR=../saved_weights  # ../saved_weights/

../clamsa.py predict fasta \
   vertebrate.test.msa.lst \
   --clades ../examples/vertebrate.nwk \
   --use_codons \
   --saved_weights_basedir $SAVED_WEIGHTS_DIR \
   --model_ids '{
      "logreg" : "logreg"
   }' \
  --out_csv clamsa_vert.csv
#       "mixed_rnn"    : "2021.03.18--15.17.25",
#       "mixed_logreg" : "2021.03.18--15.24.39"


# A warning like
# WARNING:tensorflow:Skipping loading of weights for layer P_sequence_columns due to mismatch in shape ((21,) vs (53,)).
# can be ignored. This is about a mismatch between the number of leaves in the prediction clade and the training forest ( = set of training trees)
