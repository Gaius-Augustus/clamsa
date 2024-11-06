#!/usr/bin/bash

# Example call for converting training data for ebony
# converts samples in examples/ebony/splice.train.aug
# and puts them in data/train_ebony/

../clamsa.py convert augustus ../examples/ebony/splice.train.aug \
    --clades ../examples/ebony/diverse32mammals.nwk \
    --basename diverse32mammals  \
    --splits '{"train": 0.7, "val": 0.1, "test": 0.2}' \
    --split_models 0 1  \
    --tf_out_dir ../data/train_ebony/ \
    --fixed_sequence_length 100
