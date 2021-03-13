# clamsa
ClaMSA (**Cla**ssify **M**ultiple **S**equence **A**lignments). 

# Needed packages
Assuming a working conda environment, the following commands will install all needed packages for clamsa
```console
conda install numpy tensorflow-gpu regex biopython
conda install -c bioconda python-newick
conda install -c conda-forge tqdm protobuf3-to-dict
```
# Example prediction

```console
./clamsa.py predict fasta  examples/msa2.lst  \
   --clades examples/{fly,vertebrate,yeast}.nwk  \
   --use_codons  \
   --log_basedir logs/  --saved_weights_basedir saved_weights/  \
    --model_ids '{
      "fly_vert_yeast-logreg-M3" :  "2021.02.14--20.47.20"
   }'   --out_csv /dev/stdout
```


# Example Conversion
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

We can use these Tensorflow .tfrecords files to train a set of models with multiple hyperparameter configurations in the following way:
```
./clamsa.py train tf_out \
	--basename aug-fly \
	--clades tree.nwk \
	--split_specification '{
		"train": {"name": "train", "wanted_models": [0, 1], "interweave_models": [0.67, 0.33], "repeat_models": [true, true]},
		"val": {"name": "val1", "wanted_models": [0, 1], "interweave_models": true, "repeat_models": [false, false]},
		"test": {"name": "test", "wanted_models": [0, 1], "interweave_models": true, "repeat_models": [false, false]}
	}' \
	--used_codons \
	--model_hyperparameters '{
		"tcmc_rnn": {
			"tcmc_models": [1,4,8],
			"rnn_type": ["lstm","gru"],
			"rnn_units": [32,64],
			"dense_dimension": [16]
		}, 
		"tcmc_mean_log": {
			"tcmc_models": [1,4,8],
				"sequence_length_as_feature": [false,true],
				"dense1_dimension": [0,8,16],
				"dense2_dimension": [0,16]
			}
	}' \
	--model_training_callbacks '{
		"tcmc_rnn": {},
		"tcmc_mean_log": {}
	}' \
	--batch_size 30 \
	--batches_per_epoch 100 \
	--epochs 40 \
	--save_model_weights \
	--log_basedir 'logs' \
	--saved_weights_basedir 'saved_weights' \
	--verbose \
	| tee aug-fly_train.log

```
