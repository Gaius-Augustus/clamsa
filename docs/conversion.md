# Format Conversion of MSAs

ClaMSA trains most efficiently on binary files in a particular TensorFlow Records (```tfrecords```) format. For our manuscript we used the following command to convert alignments in the text format produced by [AUGUSTUS-CGP](https://github.com/Gaius-Augustus/Augustus/blob/master/docs/EXONCAND-MSAS-CGP.md) simultaneously to tfrecords format and (PhyloCSF compatible) FASTA format.

```
clamsa.py convert augustus  vertebrate.f.out.gz \
                --subsample_lengths \
                --ratio_neg_to_pos 2 \
                --tf_out_dir . \
                --phylocsf_out_dir phylocsf \
                --refid 11 \
                --splits '{"train": -1, "test": 3000, "val": 1500}' \
                --use_codons \
                --split_models 0 1 \
                --basename vertebrate \
                --clades fly.nwk vertebrate.nwk yeast.nwk
```

```vertebrate.f.out.gz``` contains the training MSAs.  
```--tf_out_dir``` and ```--phylocsf_out_dir phylocsf``` specify output locations. You can omit one, if you only the the other type. We used both as we tested different programs with different input requirements.

```--splits``` chose 3000 random MSAs as test set, 1500 as validation set and the rest as training set.

With ```--clades``` you list trees that you may use during training.
In some of our experiments we *mixed training examples* from vertebrate with fly and yeast. If you train only on a single clade you only need to specify that tree that you will use during training.



## Usage

Type
```konsole
./clamsa.py convert -h
```
to obtain the current usage:

```
usage: clamsa.py [-h] [--tf_out_dir OUTPUT_FOLDER] [--basename BASENAME] [--phylocsf_out_dir PHYLOCSF_OUT_DIR] [--refid R] [--write_nexus NEX_FILENAME] [--nexus_sample_size N] [--splits SPLITS_JSON] [--clades CLADES [CLADES ...]]
                 [--margin_width MARGIN_WIDTH] [--tuple_length TUPLE_LENGTH] [--ratio_neg_to_pos RATIO] [--use_codons] [--phylocsf_out_use_codons] [--orig_fnames] [--use_amino_acids] [--use_compression] [--subsample_lengths]
                 [--subsample_lengths_relax SUBSAMPLE_LENGTHS_RELAX] [--verbose] [--split_models SPLIT_MODELS [SPLIT_MODELS ...]]
                 INPUT_TYPE INPUT_FILE [INPUT_FILE ...]

Convert an input multiple sequence alignment dataset to be used by clamsa.

positional arguments:
  INPUT_TYPE            Choose which type of input file(s) should be converted. Supported are: {augustus, fasta, phylocsf}
  INPUT_FILE            Input file(s) in .out(.gz) format from AUGUSTUS, in FASTA (.fs) format or a (.zip) file from PhyloCSF

optional arguments:
  -h, --help            show this help message and exit
  --tf_out_dir OUTPUT_FOLDER
                        Folder in which the converted MSA database should be stored. By default the folder "msa/" is used.
  --basename BASENAME   The base name of the output files to be generated. By default a concatination of the input files is used.
  --phylocsf_out_dir PHYLOCSF_OUT_DIR
                        Specifies that the MSA database should (also) be converted to PhyloCSF format.
  --refid R             The index of the reference species that should be in the first MSA row.
  --write_nexus NEX_FILENAME
                        A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.
  --nexus_sample_size N
                        The sample size (=number of alignments) of the nexus output. The sample is taken uniformly from among all positive alignments in random order.
  --splits SPLITS_JSON  The imported MSA database will be splitted into the specified pieces. SPLITS_JSON is assumed to be a a dictionairy in JSON notation. The keys are used in conjunction with the base name to specify an output path. The values
                        are assumed to be either positive integers or floating point numbers between zero and one. In the former case up to this number of examples will be stored in the respective split. In the latter case the number will be
                        treated as a percentage number and the respective fraction of the data will be stored in the split. A value of -1 specifies that the remaining entries are distributed among the splits of negative size. All (filtered)
                        examples are used in this case.
  --clades CLADES [CLADES ...]
                        Provide a paths CLADES to clade file(s) in Newick (.nwk) format. The species found in the input file(s) are assumed to be contained in the leave set of exactly one these clades. If so, the sequences will be aligned in the
                        particular order specified in the clade. The names of the species in the clade(s) and in the input file(s) need to coincide.
  --margin_width MARGIN_WIDTH
                        Whether the input MSAs are padded by a MARGIN_WIDTH necleotides on both sides.
  --tuple_length TUPLE_LENGTH
                        The MSAs will be exported as n-tupel-aligned sequences instead of nucleotide alignments where n is the tuple_length. This flag works only with the INPUT_TYPE fasta and not in combination with the --use_codons flag!
  --ratio_neg_to_pos RATIO
                        Undersample the negative samples (Model ID 0) or positive examples (Model ID 1) of the input file(s) to achieve a ratio of RATIO negative per positive example.
  --use_codons          The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.
  --phylocsf_out_use_codons
                        The PhyloCSF output MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.
  --orig_fnames         The original input relative filename paths will be used for outputs. Can be used only for phylocsf input type.
  --use_amino_acids     Use amino acids instead of nucleotides as alphabet. This flag works only with the INPUT_TYPE fasta.
  --use_compression     Whether the output files should be compressed using GZIP or not. By default compression is used.
  --subsample_lengths   Negative examples of overrepresented length are undersampled so that the length distributions of positives and negatives are similar. Defaults to false.
  --subsample_lengths_relax SUBSAMPLE_LENGTHS_RELAX
                        Factor for length subsampling probability of negatives. If > 1, the subsampling delivers more data but the negative length distribution fits not as closely that of the positives. Default=1.0
  --verbose             Whether some logging of the import and export should be performed.
  --split_models SPLIT_MODELS [SPLIT_MODELS ...]
                        Whether the dataset should be divided into multiple chunks depending on the models of the sequences. By default no split is performed. Say one wants to split models 0 and 1 then one may achive this by "--split_models 0 1".
```
