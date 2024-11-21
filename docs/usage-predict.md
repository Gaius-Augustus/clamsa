# Usage of Prediction
Type
```konsole
./clamsa.py predict -h
```
to obtain the current usage:

```
usage: clamsa.py [-h] [--clades CLADES [CLADES ...]] [--tuple_length TUPLE_LENGTH] [--tuples_overlap] [--use_amino_acids] 
                 [--use_codons] [--remove_stop_rows] [--batch_size BATCH_SIZE] [--log_basedir LOG_BASEDIR] 
                 [--saved_weights_basedir SAVED_WEIGHTS_BASEDIR] [--model_ids MODEL_IDS] [--out OUTPUT_FILE] [--out_csv OUT_CSV] 
                 [--name_translation TRANSTBL [TRANSTBL ...]] [--num_classes NUM_CLASSES] [--sitewise] [--classify] 
                 [--output_all_species] [--logits] [--sliding_window] [--ebony] INPUT_TYPE INPUT [INPUT ...]

Predict the class of multiple sequence alignments with one or more models.

positional arguments:
  INPUT_TYPE            Specif the input file type. Supported are: {fasta, tfrecord, maf, augustus}
  INPUT                 If INPUT_TYPE == fasta: A space separated list of paths to text files containing themselves paths to MSA files.
                        Each MSA file contains a single alignment.
                        If INPUT_TYPE == tfrecord  or INPUT_TYPE == augustus: A space separated list of paths to tfrecord/augustus files.
                        If INPUT_TYPE == maf: A (gzipped) MAF file. A .wig file with coding probabilities of codons in all six-frames is
                        output. Requires --sitewise, --ebony or --sliding_window option.
options:
  -h, --help            show this help message and exit
  --clades CLADES [CLADES ...]
                        Path(s) to the clades files (.nwk files, with branch lengths) used in the converting process.
                        CAUTION: The same ordering as in the converting process must be used!
  --tuple_length TUPLE_LENGTH
                        The MSAs will be exported as n-tupel-aligned sequences instead of nucleotide alignments where n is the 
                        tuple_length. If n = 3, you can use the flag --use_codons instead.
  --tuples_overlap      The tuples will overlap tuple_length-1 characters. This flag only works in combination with --tuple_length > 1.
  --use_amino_acids     Use amino acids instead of nucleotides as alphabet.
  --use_codons          The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.
  --remove_stop_rows    Alignment rows that contain an in-frame stop are completely removed.
  --batch_size BATCH_SIZE
                        Number of MSAs to evaluate per computation step. Higher batch sizes increase the speed of evaluation, 
                        though require more RAM / VRAM in the case of CPU / GPU evaluation.
  --log_basedir LOG_BASEDIR
                        Folder in which the Tensorboard training logs are stored. Defaults to "./logs/"
  --saved_weights_basedir SAVED_WEIGHTS_BASEDIR
                        Folder in which the weights for the best performing models are stored.
                        Defaults "./saved_weights/"
  --model_ids MODEL_IDS
                        Trial-IDs of trained models residing in the LOG_BASEDIR folder with weights stored in SAVED_WEIGHTS_BASEDIR.
  --out OUTPUT_FILE     Output file name for the predictions. CSV for MAF input with --sliding_window and FASTA input, wig for 
                        MAF input with --sitewise, GTF for MAF input with --ebony.
  --out_csv OUT_CSV     For backwards compatibility: Output file name for the *.csv file containing the predictions. Use --out instead.
  --name_translation TRANSTBL [TRANSTBL ...]
                        Path to a file that contains an optional translation table.
                        The sequence names in the fasta MSA input are translated to clade ids as used in the clade .nwk files.
                        In the tab-separated 2-column file, the first column holds the seqence name, the second the taxon id.
                        The first column cannot contain duplicates. A space separated list of paths is allowed, too.
                        Example:
                        dm       dmel
                        droAna   dana
                        dm3.chr1 dmel
  --num_classes NUM_CLASSES
                        Number of predicted classes.
  --sitewise            Predict sitewise values (needs a trained sitewise model). Currently only works on fasta files
  --classify            Predict sitewise classes (needs a trained sitewise classification model). Currently only works on fasta files
  --output_all_species  When input is in MAF format, output wiggle files for all species, not just the reference. 
                        Not compatible with --ebony or --sliding_window.
  --logits              Wiggle output are logits (positiv and negative) rather than probabilities.
  --sliding_window      Predict in a sliding window manner. Works only for position specific models on MAF files.
  --ebony               Score potential exon boundaries. Works only for exon boundary models on MAF files.
  ```
