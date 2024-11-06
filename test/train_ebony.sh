#!/usr/bin/bash

# Example training run for ebony
# the training data only contains 100 samples

EPOCHS=5
BATCHES=2
BATCHSIZE=10

../clamsa.py train ../data/train_ebony/ \
    --basenames diverse32mammals \
    --clades ../examples/ebony/diverse32mammals.nwk \
    --split_specification '{
        "train": {"name": "train", "wanted_models": [0, 1], "interweave_models": [0.75, 0.25], "repeat_models": [true, true]},
        "val"  : {"name": "val",   "wanted_models": [0, 1], "interweave_models": true, "repeat_models": [false, false]},
        "test" : {"name": "test",  "wanted_models": [0, 1], "interweave_models": true, "repeat_models": [false, false]}
        }' \
    --model_hyperparameters '{
        "tcmc_ebony" : {
            "tcmc_models": [8],
            "num_positions":[100],
            "gaps_as_feature": [true], 
            "conv_filters": [32], 
            "conv_kernelsize": [8],
            "conv2_filters": [16], 
            "conv2_kernelsize": [4], 
            "maxpool_size": [6], 
            "dense_dimension": [16] 
            }
        }' \
    --epochs $EPOCHS \
    --batches_per_epoch $BATCHES \
    --batch_size $BATCHSIZE \
    --saved_weights_basedir ../saved_weights \
    --verbose 
