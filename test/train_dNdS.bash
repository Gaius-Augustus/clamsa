#!/usr/bin/bash
#
# A quick test run for the training of the tcmc_dNds model.
# The small test training data (only 10 Alignments) are the tfrecord files in ../data/train_dNdS created with clamsa convert from fasta alignments.


EPOCHS=3          # 
BATCHES_PER_EPOCH=5 # 


python ../clamsa.py train ../data/train_dNdS/ \
	     `# basenames are the clades to be used in training` \
		 --basenames amniota \
	     `# clades specifies a list of tree files, more than one if multiple clades are trained together` \
        --clades ../examples/amniota.nwk\
	--merge_behaviour .33 .33 .33 \
        --split_specification '{
              "train": {"name": "train", "wanted_models": [0], "interweave_models": true, "repeat_models": [true, true]},
              "val"  : {"name": "val",   "wanted_models": [0], "interweave_models": true, "repeat_models": [false, false]},
              "test" : {"name": "test",  "wanted_models": [0], "interweave_models": true, "repeat_models": [false, false]}
        }' \
        --use_codons \
        --model_hyperparameters '{
              "tcmc_dNdS" : {
                  "tcmc_models": [8],
                  "dense_dimension": [16]
              }
          }' \
        --batch_size 2 \
        --batches_per_epoch $BATCHES_PER_EPOCH \
        --epochs $EPOCHS \
        --saved_weights_basedir ../saved_weights \
		--sitewise \
		--sample_weights \
	--verbose \
        | tee dNdS_train.log


# Remarks:
#
# 1) The JSON format of --model_hyperparameters allows a "grid search".
# E.g. with
# --model_hyperparameters '{ "tcmc_rnn" : \
#   { "tcmc_models": [2,4,8], "rnn_type": ["gru", "lstm"], "rnn_units": [16,32], "dense_dimension": [16] },
# one actually starts training runs for 3 x 2 x 2 combinations of hyper parameters for this one model class tcmc_rnn alone.
#
# 2) The training call determined the IDs of the models with time stamps.
#

