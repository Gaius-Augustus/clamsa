#!/usr/bin/env python3

import sys
import os
import argparse, textwrap
import configparser
import json
import numbers
import newick
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
from collections import OrderedDict
import warnings
import utilities.msa_converter as mc
import utilities.wiggle as wg

# with tf 2.4 there are UserWarnings about converting sparse IndexedSlices to a dense
# Tensor of unknown shape, that may consume a large amount of memory.
# Turn such warnings off.
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def file_exists(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"The file {path} does not exist!")
    return path

def folder_exists_and_is_writable(arg):
    if not os.path.isdir(arg) or not os.access(arg, os.W_OK):
        raise argparse.ArgumentTypeError(f"The folder {arg} does not exist or is not writable!")
    return arg

def folder_is_writable_if_exists(arg):
    if arg == None:
        return arg
    if not os.path.isdir(arg) or not os.access(arg, os.W_OK):
        raise argparse.ArgumentTypeError(f"The folder {arg} does not exist or is not writable!")
    return arg

def is_valid_split(arg):
    try:
        splits = json.loads(arg, object_pairs_hook=OrderedDict) # so the input order is kept
        if not isinstance(splits, dict):
            argparse.ArgumentTypeError(f'The provided split {arg} does not represent a dictionary!')
        for split in splits:
            if not isinstance(splits[split], numbers.Number):
                raise argparse.ArgumentTypeError(f'The provided value "{splits[split]}" for the split "{split}" is not a number!')
        return splits
    except ValueError:
        raise argparse.ArgumentTypeError(f'The provided split "{arg}" is not a valid JSON string!')

def is_valid_json(arg):
    try:
        obj = json.loads(arg)
        return obj
    except ValueError:
        raise argparse.ArgumentTypeError(f'The provided split "{arg}" is not a valid JSON string!')

class ClaMSA(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='''
Discriminative evolutionary classification of alignments.
Authors: Darvin Mertsch, Mario Stanke
''',
            usage = '''clamsa.py <command> [<args>]

Use one of the following commands:
   convert    Create an MSA dataset ready for usage in clamsa
   train      Train clamsa on an MSA dataset with given models
   predict    Infer probability for an MSA to be a coding exon
''')
        parser.add_argument('command', help='Subcommand to run')

        # parse the command
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        
        getattr(self, args.command)()

    def convert(self):
        parser = argparse.ArgumentParser(
                description='Convert an input multiple sequence alignment dataset to be used by clamsa.')

        parser.add_argument('in_type', 
                choices=['augustus', 'fasta', 'phylocsf'],
                metavar='INPUT_TYPE',
                help='Choose which type of input file(s) should be converted. Supported are: {augustus, fasta, phylocsf}')

        parser.add_argument('input_files',
            metavar="INPUT_FILE",
            nargs='+',
            type=file_exists, 
            help="Input file(s) in .out(.gz) format from AUGUSTUS, in FASTA (.fs) format or a (.zip) file from PhyloCSF")

        parser.add_argument('--tf_out_dir',
                metavar='OUTPUT_FOLDER',
                help='Folder in which the converted MSA database should be stored. By default the folder "msa/" is used.',
                type = folder_is_writable_if_exists)

        parser.add_argument('--basename',
                metavar = 'BASENAME',
                help = 'The base name of the output files to be generated. By default a concatenation of the input files is used.')

        parser.add_argument('--phylocsf_out_dir',
                help = 'Specifies that the MSA database should (also) be converted to PhyloCSF format.',
                type = folder_is_writable_if_exists)

        parser.add_argument('--refid',
                            metavar = 'R',
                            help = 'The index of the reference species that should be in the first MSA row.',
                            type = int,
                            default = None)

        parser.add_argument('--write_nexus',
                metavar = 'NEX_FILENAME',
                help = 'A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.')

        parser.add_argument('--nexus_sample_size',
                metavar = 'N',
                help = 'The sample size (=number of alignments) of the nexus output. The sample is taken uniformly from among all positive alignments in random order.',
                type = int,
                            default = 500)

        parser.add_argument('--splits', 
                help = 'The imported MSA database will be splitted into the specified pieces. SPLITS_JSON is assumed to be a a dictionairy in JSON notation. The keys are used in conjunction with the base name to specify an output path. The values are assumed to be either positive integers or floating point numbers between zero and one. In the former case up to this number of examples will be stored in the respective split. In the latter case the number will be treated as a percentage number and the respective fraction of the data will be stored in the split. A value of -1 specifies that the remaining entries are distributed among the splits of negative size. All (filtered) examples are used in this case.',
                metavar = 'SPLITS_JSON',
                type = is_valid_split)

        # TODO: implement newick check
        parser.add_argument('--clades', 
                help = 'Provide a paths CLADES to clade file(s) in Newick (.nwk) format. The species found in the input file(s) are assumed to be contained in the leave set of exactly one these clades. If so, the sequences will be aligned in the particular order specified in the clade. The names of the species in the clade(s) and in the input file(s) need to coincide.',
                metavar = 'CLADES',
                type = file_exists,
                nargs = '+')

        parser.add_argument('--margin_width',
                help = 'Whether the input MSAs are padded by a MARGIN_WIDTH necleotides on both sides.',
                metavar = 'MARGIN_WIDTH',
                type = int,
                default = 0)        

        parser.add_argument('--tuple_length', 
                help = 'The MSAs will be exported as n-tupel-aligned sequences instead of nucleotide alignments where n is the tuple_length. This flag works only with the INPUT_TYPE fasta and not in combination with the --use_codons flag!',
                metavar = 'TUPLE_LENGTH',
                type = int,
                default = 1)

        parser.add_argument('--ratio_neg_to_pos',
                help = 'Undersample the negative samples (Model ID 0) or positive examples (Model ID 1) of the input file(s) to achieve a ratio of RATIO negative per positive example.',
                metavar = 'RATIO',
                type = float)

        parser.add_argument('--use_codons', 
                help = 'The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.',
                action = 'store_true')

        parser.add_argument('--phylocsf_out_use_codons',
                help = 'The PhyloCSF output MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.',
                action = 'store_true')

        parser.add_argument('--orig_fnames',
                help = 'The original input relative filename paths will be used for outputs. Can be used only for phylocsf input type.',
                action = 'store_true')

        parser.add_argument('--use_amino_acids', 
                help = 'Use amino acids instead of nucleotides as alphabet. This flag works only with the INPUT_TYPE fasta.',
                action = 'store_true')

        parser.add_argument('--use_compression',
                help = 'Whether the output files should be compressed using GZIP or not. By default compression is used.',
                action = 'store_false')

        parser.add_argument('--subsample_lengths',
                help = 'Negative examples of overrepresented length are undersampled so that the length distributions of positives and negatives are similar. Defaults to false.',
                action = 'store_true')

        parser.add_argument('--subsample_depths_lengths',
                help = 'Negative and positive examples a undersampled, so the joint distribution of MSA depth and length are are similar. If specified, do not specificy --subsample_lengths simultaneously. Defaults to false.',
                action = 'store_true')

        parser.add_argument('--subsample_lengths_relax',
                help = 'Factor for length subsampling probability of negatives. If > 1, the subsampling delivers more data but the negative length distribution fits not as closely that of the positives. Default=1.0', type=float, default=1.0)

        parser.add_argument('--min_sequence_length',
                help = 'Minum length of alignment', type=int, default=1)

        parser.add_argument('--verbose',
                help = 'Whether some logging of the import and export should be performed.',
                action = 'store_true')

        parser.add_argument('--split_models',
                help = 'Whether the dataset should be divided into multiple chunks depending on the models of the sequences. By default no split is performed. Say one wants to split models 0 and 1 then one may achive this by "--split_models 0 1".',
                type = int,
                nargs = '+')
        
        parser.add_argument('--sitewise',
                help = 'Whether the dataset is for training a sitewise (=columnwise) model, one label for each site is expected (currently only supports fasta as input).',
                action = 'store_true')
        
        parser.add_argument('--subsample_small_omega',
                help = 'Undersample examples with small dNdS to achieve the desired OMEGA_RATIO of omegas >= s to omega < s (default: s = 1.0)',
                metavar = 'OMEGA_RATIO',
                type = float)
        
        parser.add_argument('--subsample_separator',
                help = 'Value s for separating small from large omegas (default: 1.0)',
                metavar = 'SEPARATOR',
                type = float,
                default=1.0)

        parser.add_argument('--no_codon_alignment',
                help = 'Assume that the input MSA is already codon-aligned.',
                action = 'store_true')

        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])

        if args.basename == None:
            args.basename = '_'.join(Path(p).stem for p in args.input_files)

        if args.sitewise and args.in_type != 'fasta':
            print("Datasets for the sitewise model are currently only supported in fasta format")
            return

        if args.in_type == 'fasta':
            T, species = mc.import_fasta_training_file(args.input_files,
                                                       reference_clades = args.clades,
                                                       margin_width = args.margin_width,
                                                       tuple_length = args.tuple_length,
                                                       use_amino_acids = args.use_amino_acids,
                                                       use_codons = args.use_codons,
                                                       sitewise = args.sitewise)

        if args.in_type == 'augustus':
            T, species = mc.import_augustus_training_file(args.input_files,
                                                          reference_clades = args.clades,
                                                          margin_width = args.margin_width,
                                                          use_codons = args.use_codons)
                                                          

        if args.in_type == 'phylocsf':
            T, species = mc.import_phylocsf_training_file(args.input_files,
                                                          reference_clades = args.clades,
                                                          margin_width = args.margin_width,
                                                          use_codons = args.use_codons)

        # harmonize the length distributions if requested
        if args.subsample_lengths:
            if args.sitewise:
                print("Unsupported option: subsample_lengths - for sitewise training data")
                return
            
            T = mc.subsample_lengths(T, min_sequence_length = args.min_sequence_length, relax=args.subsample_lengths_relax)

        if args.subsample_depths_lengths:
            if args.sitewise:
                print("Unsupported option: subsample_depths_lengths - for sitewise training data")
                return
            
            T = mc.subsample_depths_lengths(T, min_sequence_length = args.min_sequence_length,
                                            relax=args.subsample_lengths_relax,
                                            pos_over_neg_mod=6.0) # favor less abundant positive examples

        # achieve the requested ratio of negatives to positives
        if args.ratio_neg_to_pos:
            if args.sitewise:
                print("Unsupported option: ratio_neg_to_pos - for sitewise training data (maybe use subsample_small_omegas)")
                return
            
            T = mc.subsample_labels(T, args.ratio_neg_to_pos)
            if args.subsample_depths_lengths:
                print ("Creating histogram")
                mc.plot_depth_length_scatter(T, id = "sub-subsampled")
                mc.plot_lenhist(T, id = "sub-subsampled")
        
        # achieve the requested ratio of omegas >= s to omegas < s
        if args.subsample_small_omega:
            T = mc.subsample_omegas(T, args.subsample_small_omega, args.subsample_separator)
        
        
        
        print ("Number of filtered alignments available to be written: ", len(T))
        
        if len(T) > 0:
            # write NEXUS format for tree construction
            if args.write_nexus:
                mc.export_nexus(T, species, nex_fname = args.write_nexus,
                                n = args.nexus_sample_size, use_codons = args.use_codons)
            
            # compute actual split sizes: how many alignments to write in test, validation, training sets
            splits, split_models, split_bins, n_wanted \
            = mc.preprocess_export(T, species,
                                   args.splits,
                                   args.split_models,
                                   args.verbose)
            
            # store MSAs in tfrecords, if requested
            if args.tf_out_dir:
                num_skipped = mc.persist_as_tfrecord(T,
                        args.tf_out_dir,
                        args.basename,
                        species,
                        splits, split_models, split_bins, n_wanted,
                        use_compression = args.use_compression,
                        sitewise = args.sitewise,
                        verbose = args.verbose,
                        no_codon_alignment = args.no_codon_alignment)

                print(f'The datasets have sucessfully been saved in tfrecord files.')
            
            # store MSAs in PhyloCSF format, if requested
            if args.phylocsf_out_dir:
                mc.write_phylocsf(T,
                        args.phylocsf_out_dir,
                        args.basename,
                        species,
                        splits, split_models, split_bins, n_wanted,
                        refid = args.refid,
                        orig_fnames = args.orig_fnames
                )
                
                print(f'The datasets have sucessfully been saved in PhyloCSF files.')

                
    def train(self):
        
        parser = argparse.ArgumentParser(
                description='Train a series of models and hyperparameter configurations on an input multiple sequence alignment dataset generated by clamsa.')

        parser.add_argument('input_dir', 
                            metavar='INPUT_DIR',
                            help='Folder in which the converted MSA database should be stored. By default the folder "msa/" is used.',
                            type = folder_is_writable_if_exists,
        )
        
        parser.add_argument('--basenames',
                            metavar = 'BASENAMES',
                            help = 'The base name of the input files.',
                            nargs='+',
        )
        
        
        parser.add_argument('--clades', 
                            help='Path(s) to the clades files (.nwk files, with branch lengths) used in the converting process. CAUTION: The same ordering as in the converting process must be used!',
                            metavar='CLADES',
                            type=file_exists,
                            nargs='+',
        )
        
        parser.add_argument('--merge_behaviour',
                            metavar='MERGE_BEHAVIOUR',
                            help='In which ratio the respective splits for each basename shall be merged. The possible modes are: "evenly", "w_1 ... w_n". Where "evenly" means all basenames have the same weight. A set of custom weights can be given directly. Default is "evenly".',
                            nargs='+',
        ) 
        # possible extensions:  "columns", "sequences"
        # In the mode "columns" the total number of alignment columns for each basename is counted and the weights are adjusted accordingly. In mode "sequences" the total number of sequences for each basename is counted and the weights are adjusted accordingly.
        
        parser.add_argument('--tuple_length', 
                            help = 'The MSAs will be exported as n-tupel-aligned sequences instead of nucleotide alignments where n is the tuple_length. If n = 3, you can use the flag --use_codons instead.',
                            metavar = 'TUPLE_LENGTH',
                            type = int,
                            default = 1)
        
        
        parser.add_argument('--split_specifications', 
                            help = 'see test/train.sh for an example',
                            metavar='SPLIT_SPECIFICATIONS',
                            type=is_valid_json,
        )
        
        parser.add_argument('--use_amino_acids', 
                            help = 'Use amino acids instead of nucleotides as alphabet.',
                            action = 'store_true',
        )

        parser.add_argument('--use_codons', 
                            help = 'The MSAs were exported as codon-aligned codon sequences instead of nucleotide alignments.',
                            action = 'store_true',
        )
        
        parser.add_argument('--model_hyperparameters', 
                            help='see test/train.sh for an example',
                            metavar='MODEL_HYPERPARAMETERS',
                            type=is_valid_json,
        )
        
        
        parser.add_argument('--model_training_callbacks', 
                            help = '', # TODO: Write help'
                            metavar='MODEL_TRAINING_CALLBACKS',
                            type=is_valid_json,
        )
        
        
        parser.add_argument('--batch_size',
                            help='Number of MSAs per training batch.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=30,
        )
        
        
        parser.add_argument('--batches_per_epoch',
                            help='Number of training batches in each epoch.',
                            metavar='BATCHES_PER_EPOCH',
                            type=int,
                            default=100,
        )
        
        
        parser.add_argument('--epochs',
                            help='Number of epochs per hyperparameter configuration.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=40,
        )
        
        
        #parser.add_argument('--save_model_weights', 
        #                    help = 'Whether the weights of the best performing models shall be saved.',
        #                    action = 'store_true',
        #)
        
        
        parser.add_argument('--log_basedir',
                            metavar = 'LOG_BASEDIR',
                            help = 'Folder in which the Tensorboard training logs should be stored. Defaults to "./logs/"',
                            # default : same as saved_weights_dir
                            type = folder_is_writable_if_exists,
        )

        parser.add_argument('--saved_weights_basedir',
                            metavar='SAVED_WEIGHTS_BASEDIR',
                            help='Folder in which the weights for the best performing models should be stored. Defaults "./saved_weights/"',
                            type = folder_is_writable_if_exists,
        )
        
        parser.add_argument('--sitewise',
                            help = 'Whether the training is for estimating sitewise values (uses different datasets for training)',
                            action = 'store_true')
        
        parser.add_argument('--sample_weights',
                            help = 'Whether sample weights should be used for training (currently creates weights 0.05, 1.0, 3.5 for dNdS values omega <= 0.8, 0.8 < omega < 1.2, 1.2 <= omega) ',
                            action = 'store_true')
        
        parser.add_argument('--classify',
                            help = 'Whether the training is for classification of sitewise classes (tcmc_dNdS_class). If sitewise but classify is not specified, a regression model is expected (tcmc_dNdS).',
                            action = 'store_true')
                
        parser.add_argument('--verbose', 
                            help = 'Whether training information should be printed to console.',
                            action = 'store_true',
        )
        
        
        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])

        # default the log_dir to the saved_weights_dir
        if args.log_basedir is None:
            args.log_basedir = args.saved_weights_basedir
            
        from utilities.training import train_models
        
        train_models(args.input_dir, 
                     args.basenames,
                     args.clades,
                     args.merge_behaviour if args.merge_behaviour else 'evenly',
                     args.split_specifications,
                     args.tuple_length,
                     args.use_amino_acids,
                     args.use_codons,
                     args.model_hyperparameters,
                     args.model_training_callbacks,
                     args.batch_size,
                     args.batches_per_epoch,
                     args.epochs,
                     True, # args.save_model_weights,
                     args.log_basedir,
                     args.saved_weights_basedir,
                     args.sitewise,
                     args.classify,
                     args.sample_weights,
                     args.verbose,
        )
        
    
    def predict(self):
        
        
        parser = argparse.ArgumentParser(
            description='Predict the class of multiple sequence alignments with one or more models.',
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('in_type',
                choices=['fasta', 'tfrecord', 'maf'],
                metavar='INPUT_TYPE',
                help='Specif the input file type. Supported are: {fasta, tfrecord, maf}',
        )
        

        parser.add_argument('input', 
                            metavar='INPUT',
                            help='If INPUT_TYPE == fasta: A space separated list of paths to text files containing themselves paths to MSA files.\nEach MSA file contains a single alignment.\nIf INPUT_TYPE == tfrecord: A space separated list of paths to tfrecord files.\nIf INPUT_TYPE == maf: A (gzipped) MAF file. A .wig file with coding probabilities of codons in all six-frames is output. Requires --sitewise option',
                            type=file_exists,
                            nargs='+',
        )
        
        parser.add_argument('--clades', 
                            help='Path(s) to the clades files (.nwk files, with branch lengths) used in the converting process.\nCAUTION: The same ordering as in the converting process must be used!',
                            metavar='CLADES',
                            type=file_exists,
                            nargs='+',
        )

        parser.add_argument('--tuple_length', 
                            help = 'The MSAs will be exported as n-tupel-aligned sequences instead of nucleotide alignments where n is the tuple_length. If n = 3, you can use the flag --use_codons instead.',
                            metavar = 'TUPLE_LENGTH',
                            type = int,
                            default = 1)
                
        parser.add_argument('--use_amino_acids', 
                            help = 'Use amino acids instead of nucleotides as alphabet.',
                            action = 'store_true',
        )
        
        parser.add_argument('--use_codons', 
                            help = 'The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.',
                            action = 'store_true',
        )
        
        parser.add_argument('--remove_stop_rows',
                            help = 'Alignment rows that contain an in-frame stop are completely removed.',
                            action = 'store_true',
        )

        
        parser.add_argument('--batch_size',
                            help='Number of MSAs to evaluate per computation step.\nHigher batch sizes increase the speed of evaluation, though require more RAM / VRAM in the case of CPU / GPU evaluation.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=30,
        )
        
        parser.add_argument('--log_basedir',
                            metavar='LOG_BASEDIR',
                            help='Folder in which the Tensorboard training logs are stored. Defaults to "./logs/"',
#                            default = './logs/',
                            type = folder_is_writable_if_exists,
        )
        
        
        parser.add_argument('--saved_weights_basedir',
                            metavar='SAVED_WEIGHTS_BASEDIR',
                            help='Folder in which the weights for the best performing models are stored.\nDefaults "./saved_weights/"',
                            type = folder_is_writable_if_exists,
        )
        
        
        parser.add_argument('--model_ids',
                            metavar='MODEL_IDS',
                            help='Trial-IDs of trained models residing in the LOG_BASEDIR folder with weights stored in SAVED_WEIGHTS_BASEDIR.',
                            default='{ "clamsa" : "default" }',
                            type=is_valid_json,
                           )
        
        
        parser.add_argument('--out', 
                           metavar='OUTPUT_FILE',
                           help='Output file name for the predictions. CSV for FASTA input, wig for MAF input.',
                           type = str
        )
              
        parser.add_argument('--out_csv', 
                           metavar='OUT_CSV',
                           help='For backwards compatibility: Output file name for the *.csv file containing the predictions. Use --out instead.',
        )

        parser.add_argument('--name_translation', 
                            help='''Path to a file that contains an optional translation table.
The sequence names in the fasta MSA input are translated to clade ids as used in the clade .nwk files.
In the tab-separated 2-column file, the first column holds the seqence name, the second the taxon id.
The first column cannot contain duplicates. A space separated list of paths is allowed, too.
Example:
dm       dmel
droAna   dana
dm3.chr1 dmel''',
                            metavar='TRANSTBL',
                            type=file_exists,
                            nargs='+',
        )

        parser.add_argument('--num_classes',
                            help='Number of predicted classes.',
                            metavar='NUM_CLASSES',
                            type=int,
                            default=2,
        )
        
        parser.add_argument('--sitewise',
                            help='Predict sitewise values (needs a trained sitewise model). Currently only works on fasta files',
                            action='store_true',
        )
        parser.add_argument('--classify',
                            help='Predict sitewise classes (needs a trained sitewise classification model). Currently only works on fasta files',
                            action='store_true',
        )
        parser.add_argument('--output_all_species',
                            help='When input is in MAF format, output wiggle files for all species, not just the reference',
                            action='store_true',
        )
        parser.add_argument('--logits',
                            help='Wiggle output are logits (positiv and negative) rather than probabilities.',
                            action='store_true',
        )

        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])

        if args.saved_weights_basedir is None:
            pathname = os.path.dirname(sys.argv[0])
            args.saved_weights_basedir = os.path.join(os.path.abspath(pathname), "saved_weights")


        # default the log_dir to the saved_weights_dir
        if args.log_basedir is None:
            args.log_basedir = args.saved_weights_basedir

        if args.out is not None and args.out_csv is not None:
            print("Only one of --out and --out_csv can be specified. Please use --out only.")
            return
        if args.out_csv is not None:
            args.out = args.out_csv
            print("Warning: --out_csv is deprecated. Please use --out instead.")

        # import on demand (importing tf is costly)
        import utilities.model_evaluation as me

        if args.in_type == 'fasta' or args.in_type == 'maf': # text MSA input
            # read name->taxon_id translation tables into dictionary if specified
            trans_dict = {}
            if not args.name_translation is None:
                for trfn in args.name_translation:
                    with open(trfn) as f:
                        for line in f.read().splitlines():
                            a = line.split('\t')
                            if len(a) != 2:
                                raise Exception(f"Translation file {trfn} contains an error in line {line}. Must have 2 tab-separated fields.")
                            (fasta_name, taxon_id) = a
                            if fasta_name in trans_dict and trans_dict[fasta_name] != taxon_id:
                                raise Exception(f"Translation file {trfn} contains conflicting duplicates: {fasta_name} -> {trans_dict[fasta_name]}, {taxon_id}")
                            trans_dict[fasta_name] = taxon_id
            model_ids = OrderedDict(args.model_ids) # to fix the models order as on the command line

            if args.in_type == 'fasta':
                preds = me.predict_on_fasta_files( \
                    trial_ids = model_ids, saved_weights_dir = args.saved_weights_basedir,
                    log_dir = args.log_basedir, clades = args.clades, input_files = args.input,
                    use_amino_acids = args.use_amino_acids, use_codons = args.use_codons,
                    tuple_length = args.tuple_length, batch_size = args.batch_size,
                    trans_dict = trans_dict, remove_stop_rows = args.remove_stop_rows,
                    num_classes = args.num_classes, sitewise = args.sitewise,
                    classify = args.classify)
            else: # in_type == 'maf'
                if args.num_classes != 2:
                    raise Exception("maf input is currently only supported for binary classification")
                if len(args.model_ids) > 1:
                    raise Exception("maf input is currently only supported for a single model.")
                preds, aux = me.predict_on_maf_files( \
                    trial_ids = model_ids, saved_weights_dir = args.saved_weights_basedir,
                    log_dir = args.log_basedir, clades = args.clades,
                    paths = args.input, use_codons = args.use_codons,
                    tuple_length = args.tuple_length, batch_size = args.batch_size,
                    trans_dict = trans_dict, remove_stop_rows = args.remove_stop_rows,
                    output_all_species = args.output_all_species)

        elif args.in_type == 'tfrecord':
            if args.sitewise:
                print("Sitewise prediction currently does not work on tfrecord files")
                return
            
            preds = me.predict_on_tfrecord_files(trial_ids=args.model_ids,
                                                 saved_weights_dir=args.saved_weights_basedir,
                                                 log_dir=args.log_basedir,
                                                 clades=args.clades,
                                                 tfrecord_paths = args.input,
                                                 use_amino_acids = args.use_amino_acids,
                                                 use_codons = args.use_codons,
                                                 tuple_length = args.tuple_length,
                                                 batch_size = args.batch_size,
                                                 num_classes = args.num_classes
            )
        else:
            raise Exception(f"Unsupported input type: {args.in_type}")

        # determine output format and output the preds
        if args.sitewise:
            if args.in_type == 'fasta':
                # write dictionary to file
                # for a classification task: probabilitys of both classes are output in succession
                if args.out is None:
                    print(preds, end = "")
                else:
                    with open (args.out, mode = 'w') as f:
                        for path, values in preds.items():
                            f.write('%s:%s\n' % (path, values))
                
                pickle_file = open("clamsa_out.pkl", "wb")
                pickle.dump(preds, pickle_file)
                pickle_file.close()
            elif args.in_type == 'maf':
                # write wig file
                wg.write_preds_to_wig(preds, aux, args.out,
                                      output_all_species=args.output_all_species,
                                      logits=args.logits)
            
        else:
            # construct a dataframe from the predictions
            df = pd.DataFrame.from_dict(preds)
    
            from io import StringIO
            output = StringIO()
    
            df.to_csv(output, sep='\t',
                      float_format = '%.4f', # output precision
                      index = False,
                      header = True,
                      mode = 'w' )
            outputstr = output.getvalue()
    
            if args.out is None:
                print(outputstr, end = "")
            else:
                with open(args.out, mode='w') as f:
                    print(outputstr , end = "", file = f)


def main():
    ClaMSA()
    exit(0)

if __name__ == "__main__":
    main()
