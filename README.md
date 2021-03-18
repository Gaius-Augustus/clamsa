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

```console
cd clamsa

./clamsa.py predict fasta examples/msa2.lst  \
   --clades examples/fly.nwk  \
   --use_codons \
   --saved_weights_basedir saved_weights/  \
    --model_ids '{"clamsa" :  "default"}' \
    --out_csv /dev/stdout
```


# Utilities
  - [MSA conversion](docs/conversion.md)

