import tensorflow as tf
import numpy as np
from collections import defaultdict
import os
import regex as re
from functools import partial
from . import msa_converter


def num_leaves(forest):
    '''
        Given a list of Newick files gather the number of leaves occuring in this forest

        Args:
            forest (list(str)): List of paths to the considered Newick tree files

        Returns:
            (list(int)): Number of leaves encountered in any tree.
    '''
    num_leaves = [len(msa_converter.leaf_order(t)) for t in forest]

    return num_leaves


def find_exported_tfrecord_files(folder, basename, by_splits = False):
    '''
    Lists all files generated by `persist_as_tfrecord` in a given folder matching a basename
    and separates them via split-model flags 
    
    Args:
        folder (str): Path to the folder to be searched
        basename (str): Filename prefix
        by_splits (bool): Whether a dict with tuples as keys or a nested dict should be returned
        
    Returns:
        (dict) A dict with keys given either by 
                    
                    tuples `(split_name, model_name)` when `by_splits`
                    is false and entries given by lists of file names matching these split names
                    and models 
                    
                    strings `split_name` with values given by dicts with string keys `model_name` 
                    and entries given by lists of files names matching these split names and 
                    model names.
    '''
    files = os.listdir(folder)
    export_re = re.compile(f'{basename}(?:-([a-zA-Z]+))?(?:-m([0-9]+))?')
    wanted_files = filter(export_re.match, files)
    
    splits = defaultdict(list)
    for f in wanted_files:
        split, model = export_re.findall(f)[0]
        key = (split, model)
        splits[key].append(folder + f)
        
    if by_splits:
        split_names = set(s for (s,m) in splits.keys())
        splits_by_name = {name: {} for name in split_names}
        for s,m in splits:
            model = int(m) if str.isdigit(m) else m           
            splits_by_name[s][model] = splits[(s,m)]
            
        splits = splits_by_name
    
    return dict(splits)






@tf.function
def parse_tfrecord_entry(entry, num_leaves, alphabet_size):  
    """Parse function for tf.data.Dataset.map to process tfrecords files generated by `msa_converty.py`

    Args:
        entry (tf.train.SequenceExample) Sequences Example generated by `msa_converter.persist_as_tfrecord` function.
        num_leaves (list(int)) Number of leaves occuring in the trees in the forest
        alphabet_size (int) Number of characters in the alphabet (e.g. `3` for codons or `1` for nucleotide sequences)

    Returns:
        (tf.tuple) Tuples of tensors `(model, clade_id, sequence_length, sequence_onehot)`
    """

    s = alphabet_size
    
    # declare the structure of the tfrecord
    context_features = {
        'model': tf.io.FixedLenFeature([], dtype = tf.int64),
        'clade_id': tf.io.FixedLenFeature([], dtype = tf.int64),
        'sequence_length': tf.io.FixedLenFeature([], dtype = tf.int64),
    }
    sequence_features = {
        'sequence_onehot': tf.io.FixedLenSequenceFeature([], dtype = tf.string)
    }
    
    # obtain one example of the given structure
    context, sequence = tf.io.parse_single_sequence_example(
      entry, 
      context_features = context_features,
      sequence_features = sequence_features
    )
    
    # perform some transformation
    model = context['model']
    clade_id = context['clade_id']

    # number of leaves in the specific tree
    num_leaves = tf.constant(num_leaves)
    n = num_leaves[clade_id]
    N = tf.math.reduce_max(num_leaves)

    sequence_length = context['sequence_length']
    sequence_onehot = tf.reshape(
        tf.io.decode_raw(sequence['sequence_onehot'], tf.int32),
        (sequence_length, n, s)
    )

    # pad to `N` leaves
    paddings = tf.scatter_nd([[1,1]], (N-n)[None], shape = [3,2])
    sequence_onehot = tf.pad(sequence_onehot, paddings, constant_values=1)
    
    sequence_onehot = tf.cast(sequence_onehot, tf.float64)
    clade_id = tf.cast(clade_id, tf.int32)
    model = tf.cast(model, tf.int32)
    
    # return the transformed example
    return tf.tuple([model, clade_id, sequence_length, sequence_onehot])



class DatasetSplitSpecification(object):
    '''
    A raw class to specify settings used in the `get_datasets` function.
    '''

    def __init__(self, name, wanted_models, interweave_models = None, repeat_models = None):
        self.name = name
        self.wanted_models = wanted_models
        self.interweave_models = interweave_models
        self.repeat_models = repeat_models




def get_datasets(folder, basename, wanted_splits, num_leaves, alphabet_size, seed = None, buffer_size = 1000, used_compression = True, should_shuffle=False):
    '''
    Reads all tfrecord files in a folder with a given base name and returns them in splits.

    Args:
        folder (str): Folder to be searched
        basename (str): Base name of files to be considered
        wanted_splits (list(DatasetSplitSpecification)): All splits and models of those splits that should be read given configurations.
        num_leaves (list(int)): Number of leaves occuring in the forest to be studied for each tree.
        alphabet_size (int): Number of characters in the alphabet the files have been written with (e.g. `3` for codon sequences or `1` for nucleotide sequences)
        seed (int): Random seed to be used.
        buffer_size (int): Caching parameter for the Tensorflow datasets.
        used_compression (bool): Whether the datasets have been persisted with GZIP compression or not.

    Returns:
        (dict): Dict with keys given by strings `split_name` and values given eithet by 
                    
                    tf.data.Dataset if the models of the split had to be interleaved 

                    or dict with keys given by strings `model_name` and values given 
                    by tf.data.Dataset objects
    '''
    
    # get all files as a nested dict. by their split names
    files = find_exported_tfrecord_files(folder, basename, by_splits = True)
    
    # set up an entry parse function for the datasets
    parser = partial(parse_tfrecord_entry, num_leaves = num_leaves, alphabet_size=alphabet_size)
    
    compression_type = 'GZIP' if used_compression else None
    
    datasets = {}
    
    for split in wanted_splits:
        
        split_ds = {}
        
        for mid, model in enumerate(split.wanted_models):
            split_ds[model] = None
            for filename in files[split.name][model]:
                dataset = tf.data.TFRecordDataset(filename, 
                                        compression_type = compression_type, 
                                        buffer_size = buffer_size) \
                .map(parser, num_parallel_calls = 2)
                
                # all datasets of the same model in this split are concatinated
                split_ds[model] =  split_ds[model].concatenate(dataset) if split_ds[model] != None else dataset
                
            if split.repeat_models != None and split.repeat_models[mid]:
                split_ds[model] = split_ds[model].repeat()
        
        if split.interweave_models != None:
            if split.interweave_models == True:
                sd = None
                for m in split_ds:
                    sd = sd.concatenate(split_ds[m]) if sd != None else split_ds[m]
                split_ds = sd
            else:
                split_ds = tf.data.experimental.sample_from_datasets(list(split_ds.values()), weights = split.interweave_models, seed = seed) # percentages of neg and pos
                
            if should_shuffle:
                split_ds = split_ds.shuffle(buffer_size = buffer_size, seed = seed)
        
        
        datasets[split.name] = split_ds
    
    return datasets



def concatenate_dataset_entries(models, clade_ids, sequence_lengths, sequences):
    """
    Preprocessing function to concatenate a zero-padded batch of
    variable-length sequences into a single sequence.
    """
    
    concat_sequences = tf.cast(
        tf.boolean_mask(sequences, tf.sequence_mask(sequence_lengths)), 
        dtype = tf.float64)
    
    models_onehot = tf.one_hot(models, depth = 2)
    
    X = (concat_sequences, tf.repeat(clade_ids, sequence_lengths, axis=0), sequence_lengths)
    y = models_onehot
    
    return (X,y)


# TODO: These two functions behave nearly the same. Unify them!
def concat_sequences(clade_ids, sequence_lengths, sequences):
    concat_sequences = tf.cast(
        tf.boolean_mask(sequences, tf.sequence_mask(sequence_lengths)), 
        dtype = tf.float64)
    
    X = (concat_sequences, tf.repeat(clade_ids, sequence_lengths, axis=0), sequence_lengths)
    
    return (X, None)


def padded_batch(dataset, batch_size, num_leaves, alphabet_size):
    """
    Retrieve a zero-padded batch of variable length sequences.
    """
    padded_shapes = ([], [], [], [None, max(num_leaves), alphabet_size])
    dataset = dataset.padded_batch(batch_size, 
                                   padded_shapes = padded_shapes, 
                                   padding_values = (
                                       tf.constant(0, tf.int32),
                                       tf.constant(0, tf.int32), 
                                       tf.constant(0, tf.int64), 
                                       tf.constant(1.0, tf.float64))
                                  )
    
    return dataset


