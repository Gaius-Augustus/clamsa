# ClaMSA - **Cla**ssify **M**ultiple **S**equence **A**lignments 
ClaMSA is tool that uses machine learning to classify sequences that are related to each other via a tree. It takes as input a tree and a multiple sequence alignment (MSA) of sequences and outputs probabilities that the MSA belongs to given classes.
It is currently trained and tested to classify sequences of codons (= triplets of DNA characters) into coding (1) or non-coding (0).
It builds on TensorFlow and a custom layer for Continuous-Time Markov Chains (CTMC) and **trains a set of rate matrices** for a classification task.

![MSA example](examples/msa1-2.png)

Above image shows two toy example input MSAs. Synonymous codons, which code for the same amino acid, have the same color.

# Requirements
  Python modules
 - tensorflow >= 2.0
 - biopython
 - regex
 - newick
 - tqdm
 - pandas
 - protobuf3-to-dict
 - matplotlib
 - seaborn

Install requirements with
```console
pip3 install tensorflow biopython regex newick tqdm pandas protobuf3-to-dict matplotlib seaborn
```

# Installation

Download ClaMSA with
```console
git clone --recurse-submodules https://github.com/Gaius-Augustus/clamsa.git
```

# Example Classification
The commands
```console
cd clamsa

./clamsa.py predict fasta examples/msa.lst --clades examples/example_tree.nwk --use_codons
```
output the table
```
path                    clamsa
examples/msa1.fa        0.9539
examples/msa2.fa        0.1667
```
Here, the two toy example alignments `msa1`, `msa2` pictured above are predicted to precoding with probabilities 0.9539 and 0.1667, respectively.  
See the [usage of prediction](docs/usage-predict.md) for an explanation of the command line structure.  
See [test/predict.sh](test/predict.sh) for more explanations and a realistical application.

# Input Tree Construction

For codon MSA classification we recommend that you construct a tree the following way:
  1. Construct a set of codon MSAs just as you would do for prediction. You only need **positive** examples, i.e. alignments of actual coding sequences. One option to compile such a set is [AUGUSTUS-CGP](https://github.com/Gaius-Augustus/Augustus/blob/master/docs/EXONCAND-MSAS-CGP.md).
  2. Construct a tree with MrBayes using a *codon model* as described in the [supplementary material to below paper](https://www.biorxiv.org/content/biorxiv/early/2021/03/10/2021.03.09.434414/DC1/embed/media-1.pdf?download=true).

Other trees may work, but a good performance should only be expected if the tree is scaled to **1 expected codon mutation per time unit**.

# Train and Test Example Data
Obtain
  1. codon alignment training data from a fly, vertebrate and yeast clade in tfrecords format and
  2. codon alignment test data from vertebrates in fasta format with
  
```konsole
cd data
./download_fly_vert_yeast_train.sh
./download_vert_test.sh
```

# Training
ClaMSA can be trained for a classification task on a training set of labeled MSAs.  
See [test/train.sh](test/train.sh) for more explanations and the command line that ClaMSA was trained with.


# Usages
  - [clamsa predict](docs/usage-predict.md)
  - [clamsa train](docs/usage-train.md)
  - [clamsa convert](docs/conversion.md) (MSA conversion)

# Reference
Most of ClaMSA was written by Darvin Mertsch.  

Please cite:  
[End-to-end Learning of Evolutionary Models to Find Coding Regions in Genome Alignments](https://www.biorxiv.org/content/10.1101/2021.03.09.434414v1), Darvin Mertsch and Mario Stanke, *bioRxiv* 2021.03.09.434414
