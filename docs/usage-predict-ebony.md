# Predicting splice sites in MSAs

Ebony is a position-specific model developed to detect splice site in multiple sequence alignments.
It requires the input sequences to be of a fixed length to enable the position-specificity.
It outputs predicted probabilities for each class.

The ebony weights in `saved_weights/` were trained on 32 diverse mammals.
The trained model is strand-specific with a fixed sequence length of 100 and 2 classes (splice site and not splice site).
The splice site is required to be right in the middle of the MSA, i.e. one side of 50 nucleotides is coding and the other non-coding.
The input should be a pre-selection of potential splice sites, e.g. as produced by Augustus CGP.

Example data is in `examples/ebony/` 
```
cd examples/ebony/
```
Run ebony predictions with 

```
../../clamsa.py predict augustus splice.predict.aug.gz \
     --clades diverse32mammals.nwk \
     --saved_weights_basedir ../../saved_weights/ \
     --model_ids '{ "ebony" : "ebony_splice" }' \
     --num_classes 2
```

The tree `diverse32mammals.nwk` is scaled to 1 expected codon mutation per time unit and contains the same species names as the input Augustus file `splice.predict.aug.gz`.
The output can be compared to `expected_output.csv`.

`test/convert_ebony.sh` and `test/train_ebony.sh` show examples of how to convert training data with a fixed sequence length and how to train ebony.

### Model architecture

Ebony can be divided into functional blocks.
The main block, called sequence block, consists of a CTMC (continuous-time Markov chain) layer, followed by CNN, MaxPool, DropOut, and a second CNN layer.
Additionally, a few optional feature blocks are available: gaps, depth, and frames (which was specifically developed for an input encoding of overlapping tuples of length 3).
