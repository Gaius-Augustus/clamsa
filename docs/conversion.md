# Conversion (Not ready for release yet)
Example conversion parameters for our usual flie dataset from augustus:
```console
./clamsa.py convert augustus msa/train.out --clades clades/flies.nwk --splits '{"train": 0.9, "test": 0.05, "val": 0.05}' --margin_width 10 --basename augustus_flies --use_codons --split_models 0 1
```
Where we assume that the file `train.out` is stored in the folder `msa` and the Newick file specifying the phylogenetic tree is stored in `clades/flies.nwk`.

Synchronous export to tensorflow records and PhyloCSF files:

```
./clamsa.py convert augustus sample2.out \
   --subsample_lengths  \
   --ratio_neg_to_pos 2 \
   --tf_out_dir tf_out \
   --phylocsf_out_dir PhyloCSF \
   --splits '{"train": -1, "test": 100, "val1": 30, "val2": 10}' \
   --use_codons \
   --split_models 0 1 \
   --margin_width 0 \
   --basename aug-fly \
   --clades tree.nwk \
   | tee convert.out

Parsing AUGUSTUS file(s): 100%|██████████████████████████████████████████████████████▉| 27.3M/27.3M [00:01<00:00, 16.0Mb/s]
Writing TensorFlow record: 100%|███████████████████████████████████████████████████████| 297/297 [00:05<00:00, 57.37 MSA/s]
Writing PhyloCSF dataset: 100%|██████████████████████████████████████████████████████| 297/297 [00:00<00:00, 7720.77 MSA/s]
species= [['dmoj', 'dvir', 'dgri', 'dmel', 'dsim', 'dsec', 'dere', 'dyak', 'dana', 'dpse', 'dper', 'dwil']] reference_clades= ['tree.nwk']
69 MSAs were dropped as they had fewer than 2 rows.
Subsampling based on lengths has reduced the number of alignents from 16941 to 753
subsampling with ratio 2.0
Reducing number of negatives from 654 to 198 .
Number of filtered alignments available to be written:  297
Split totals: [157 100  30  10]
Writing to tfrecords...
number of tf records written [rows: split bin s, column: model/label m]:
 [[109.  48.]
 [ 63.  37.]
 [ 19.  11.]
 [  7.   3.]]
The datasets have sucessfully been saved in tfrecord files.
Writing to PhyloCSF flat files...
number of PhyloCSF records written [rows: split bin s, column: model/label m]:
 [[109.  48.]
 [ 63.  37.]
 [ 19.  11.]
 [  7.   3.]]
The datasets have sucessfully been saved in PhyloCSF files.

```
