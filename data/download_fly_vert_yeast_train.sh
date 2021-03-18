#!/usr/bin/bash
#
# Download the training MSAs from fly, vertebrate and yeast.
# These sets are disjoint from the test MSA set.

if [ ! -f fly_vert_yeast_train_tf.tgz ]; then
    echo "Downloading training data ..."
    wget http://bioinf.uni-greifswald.de/bioinf/downloads/data/clamsa/fly_vert_yeast_train_tf.tgz
else
    echo "Skipping download and using previously downloaded training data."
fi

if [ ! -d train ]; then
    echo "Extracting ..."
    tar xzf fly_vert_yeast_train_tf.tgz
else
    echo "Skipping extraction, directory 'train' already exists."
fi
