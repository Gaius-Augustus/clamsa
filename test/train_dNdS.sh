#!/usr/bin/bash
#
# This script contains an example training run of ClaMSA.
# It trains two models in one call of clamsa
#    1)  "tcmc_rnn" is as in the paper of Mertsch and Stanke (2021).
#        It uses a recurrent neural network and neural nets as prediction layers,
#        M=8 rate matrices (tcmc_models), the GRU has 32 units and the dense layer 16.
#    2) "tcmc_mean_log" is a logistic regression model with M=3 rate matrices.
# As training data MSAs of 3 clades are used and mixed in a ratio of 1/3, 1/3, 1/3 with each other. 
# The negatives ({fly,vertebrate,yeast}-train-m0.tfrecord.gz) and positives ({fly,vertebrate,yeast}-train-m0.tfrecord.gz)
# are mixed in a ratio of 2/3 to 1/3 as in the paper.
# The training data are the tfrecord files in ../data/train created with clamsa convert from fasta alignments.

#for convert: python3 clamsa.py convert fasta examples/msa4.fa examples/msa3.fa --tf_out_dir examples/ --clades examples/example_tree.nwk --use_codons --dNdS
#		      python3 clamsa.py convert fasta examples/diptera/*.fasta --basename diptera --splits '{"train-m0": 0.6, "test-m0": 0.2, "val-m0": 0.2}' --tf_out_dir examples/diptera --clades examples/DipteraTree_rooted.txt --use_codons --dNdS
# Reduce the number of epochs and batches per epoch, e.g. to 2 and 10, for a quick test run
EPOCHS=2            # 200 was used in the paper
BATCHES_PER_EPOCH=10 # 100 was used in the paper


python ../clamsa.py train ../examples/diptera/ \
	     `# basenames are the clades to be used in training` \
		 --basenames diptera \
	     `# clades specifies a list of tree files, more than one if multiple clades are trained together` \
        --clades ../examples/DipteraTree_rooted.txt \
	--merge_behaviour .33 .33 .33 \
        --split_specification '{
              "train": {"name": "train", "wanted_models": [0], "interweave_models": true, "repeat_models": [true, true]},
              "val"  : {"name": "val",   "wanted_models": [0], "interweave_models": true, "repeat_models": [false, false]},
              "test" : {"name": "test",  "wanted_models": [0], "interweave_models": true, "repeat_models": [false, false]}
        }' \
        --use_codons \
        --model_hyperparameters '{
              "tcmc_dNddS" : {
                  "tcmc_models": [8],
                  "dense_dimension": [16]
              }
          }' \
        --batch_size 10 \
        --batches_per_epoch $BATCHES_PER_EPOCH \
        --epochs $EPOCHS \
        --saved_weights_basedir ../saved_weights \
		--dNdS \
	--verbose \
        | tee fly_vert_yeast_train.log


# Remarks:
#
# 1) The JSON format of --model_hyperparameters allows a "grid search".
# E.g. with
# --model_hyperparameters '{ "tcmc_rnn" : \
#   { "tcmc_models": [2,4,8], "rnn_type": ["gru", "lstm"], "rnn_units": [16,32], "dense_dimension": [16] },
# one actually starts training runs for 3 x 2 x 2 combinations of hyper parameters for this one model class tcmc_rnn alone.
#
# 2) The training call determined the IDs of the models with time stamps. Find the IDs of the models in the log file with:
#

grep -A 1 "set of hyperparameters" fly_vert_yeast_train.log

# Current set of hyperparameters: {'tcmc_models': 8, 'rnn_type': 'gru', 'rnn_units': 32, 'dense_dimension': 16}
# Training information will be stored in: saved_weights/fly_vertebrate_yeast/tcmc_rnn/2021.03.18--15.17.25
# --
# Current set of hyperparameters: {'tcmc_models': 3, 'sequence_length_as_feature': False, 'dense1_dimension': 0, 'dense2_dimension': 0}
# Training information will be stored in: saved_weights/fly_vertebrate_yeast/tcmc_mean_log/2021.03.18--15.24.39
#
# In this example the time stamps 2021.03.18--15.17.25 and 2021.03.18--15.24.39 identify the respective models and are to be specified
# when predicting with clamsa.
