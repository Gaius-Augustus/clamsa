#!/usr/bin/bash
#
# Download the 3000 vertebrate test alignments from the paper.

if [ ! -f vertebrate-test-3000.tar.gz ]; then
    echo "Downloading test data ..."
    wget http://bioinf.uni-greifswald.de/bioinf/downloads/data/clamsa/vertebrate-test-3000.tar.gz
else
    echo "Skipping download and using previously downloaded test data."
fi

if [ ! -d test ]; then
    echo "Extracting ..."
    tar xzf vertebrate-test-3000.tar.gz
else
    echo "Skipping extraction, directory 'test' already exists."
fi
