# ClaMSA - **Cla**ssify **M**ultiple **S**equence **A**lignments 

# Requirements
  Python modules
 - tensorflow >= 2.0
 - regex
 - newick
 - tqdm
 - protobuf3-to-dict

Install requirements with
```console
pip install tensorflow regex newick tqdm protobuf3-to-dict
```

# Installation

Download ClaMSA with
```console
git clone git clone --recurse-submodules https://github.com/Gaius-Augustus/clamsa.git
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
examples/msa1.fa        0.9585
examples/msa2.fa        0.2802
```
Here, the two toy example alignments `msa1`, `msa2` are predicted as likely coding (0.9585) and rather non-coding (0.2802).

# Utilities
  - [MSA conversion](docs/conversion.md)

# Reference

[End-to-end Learning of Evolutionary Models to Find Coding Regions in Genome Alignments](https://www.biorxiv.org/content/10.1101/2021.03.09.434414v1), Darvin Mertsch and Mario Stanke, *bioRxiv* 2021.03.09.434414
