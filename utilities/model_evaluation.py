import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2
from protobuf_to_dict import protobuf_to_dict
from importlib import import_module
from pathlib import Path
import pathlib
import inspect
from Bio import SeqIO, AlignIO
import numpy as np
import collections
from functools import partial
import itertools
import gzip

import sys
sys.path.append("..")

from utilities import database_reader
from utilities import msa_converter



# On some versions of CuDNN the default LSTM implementation
# raises a warning. The following code deals with these cases
# See [here](https://github.com/tensorflow/tensorflow/issues/36508)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def get_hyperparameter(path):
    """
        Reads the tf.Event files generated by `hp.hparams` in order to retrieve model hyperparameters
        
        Args:
            path (str): Path to the `events.out.tfevents.*.v2` file
            
        Returns:
            Dict: A dict. with keys given by the names of the hyperparameters and their values
    """

    si = summary_iterator(path)

    for event in si:
        for value in event.summary.value:
            
            proto_bytes = value.metadata.plugin_data.content
            plugin_data = plugin_data_pb2.HParamsPluginData.FromString(proto_bytes)
            
            if plugin_data.HasField("session_start_info"):
                
                hp = plugin_data.session_start_info.hparams
                
                # convert protocol buffer to dict.
                hp = {k: list(protobuf_to_dict(hp[k]).values())[0] for k in hp}
                
                return hp
    return False


def get_model_information(trial_id, log_dir):
    """
    Find the hyperparameters and model name used in a training run of `utilities.train_models`

    Args:
        trial_id (str): `Trial ID` displayed in the `HParam` panel of Tensorboard of the wanted model
        log_dir (str): Base log directory

    Returns:
        Dict: A dict. of hyperparameters for the wanted model
        str: The name of the model
    """
    
    for path in Path(log_dir).rglob('*.v2'):
        tid = path.parts[-2]
        
        if tid == trial_id:
            
            model_name = path.parts[-3]
            return get_hyperparameter(str(path)), model_name

    return None, None
        
        
def get_weights_path(trial_id, saved_weights_dir):
    """
    Finds the weights `*.h5`-file of a trained by model by its Trial ID 

    Args:
        trial_id (str): `Trial ID` displayed in the `HParam` panel of Tensorboard of the wanted model
        saved_weights_dir (str): Base saved weights directory

    Returns:
        str: Path to the `*.h5` file
    """
    
    for path in Path(saved_weights_dir).rglob('*.h5'):
        tid = path.stem
        
        if tid == trial_id:
            
            return str(path)
        
        
def recover_model(trial_id, forest, alphabet_size, log_dir, saved_weights_dir):
    """
    Loads a trained model by its `Trial ID`

    Args:
        trial_id (str): `Trial ID` displayed in the `HParam` panel of Tensorboard of the wanted model
        forest (List[str]): List of path's to Newick files used in the training run
        alphabet_size (int): Number characters in the alphabet used in the training run
        log_dir (str): Base log directory of the training run
        saved_weights_dir (str): Base saved weights directory of the training run

    Returns:
        tf.keras.Model: Trained model
    """
    
    hps, model_name = get_model_information(trial_id, log_dir)
    if hps is None or model_name is None:
        print (f"Error: Could not find model {trial_id} in directory {log_dir}.", file = sys.stderr)
        sys.exit(1)

    print (f"Recovering model {model_name} with ID {trial_id} in directory {log_dir}.")
    try:
        model_module = import_module(f"models.{model_name}", package=__name__)
    except ModuleNotFoundError as err:
        raise Exception(f'The module "models/{model_name}.py" for the model "{model_name}" does not exist.') from err
    try:
        create_model = getattr(model_module, "create_model")
    except AttributeError as err:
        raise Exception(f'The model "{model_name}" has no creation function "create_model" in "models/{model_name}.py".')
        
    
    # obtain the types for the default hyperparameter of the create_model function
    signature = inspect.signature(create_model)
    default_hps = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_types = {k: type(default_hps[k]) for k in default_hps}
    
    # convert the loaded hyperparameters to the correct types
    hps = {k: default_types[k](hps[k]) for k in hps}
    
    # obtain the model with the correct weights
    model = create_model(forest, alphabet_size, **hps)
    weights_path = get_weights_path(trial_id, saved_weights_dir)
    # the following call can print a warning like: Skipping loading of weights for layer P_sequence_columns due to mismatch in shape
    # it cannot be turned off with logging.set_verbosity(0), however.
    model.load_weights(weights_path, by_name = True, skip_mismatch = True)
    
    return model


def predict_on_fasta_files(trial_ids, # OrderedDict of model ids with keys like 'tcmc_rnn'
                           saved_weights_dir,
                           log_dir,
                           clades,
                           input_files, # list of files, each with a with a list of fasta file paths
                           use_amino_acids = False,
                           use_codons = False,
                           tuple_length = 1,
                           batch_size = 30,
                           trans_dict = None,
                           remove_stop_rows = False,
                           num_classes = 2,
                           sitewise = False,
                           classify = False):
    # import the list of fasta file paths
    fasta_paths = []

    for fl in input_files:
        with open(fl) as f:
            fasta_paths.extend(f.read().splitlines())

    # calculate model properties
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length if not use_amino_acids else 20 ** tuple_length
    num_leaves = database_reader.num_leaves(clades)
    
    trans_dict = trans_dict if not trans_dict is None else {}
    
    # import the fasta files and filter out empty codon aligned sequences
    path_ids_without_reference_clade = set()
    path_ids_with_empty_sequences = set()
    aux = []
    def sequence_generator():
        for f in fasta_paths:
            if f == "":
                path_ids_with_empty_sequences.add(f)
                continue
            # filter fasta files that have no valid reference clade
            tensor_msas = msa_converter.parse_text_MSA(f, clades, trans_dict=trans_dict, 
                                                        remove_stop_rows=remove_stop_rows, 
                                                        use_amino_acids = use_amino_acids, tuple_length = tuple_length,
                                                        use_codons = use_codons)
            for (cid, sl, S, auxdata) in tensor_msas: 
                if cid == -1:
                    path_ids_without_reference_clade.add(f)
                    continue
                if cid == -2:
                    path_ids_with_empty_sequences.add(f)
                    continue
                # store auxiliary data in parallel list for results reporting
                aux.append(auxdata)
                # only the tensor data is yielded in the generator
                yield cid, sl, S

    
    # load the wanted models and compile them
    models = collections.OrderedDict( (name, recover_model(trial_ids[name], clades, alphabet_size, log_dir, saved_weights_dir)) for name in trial_ids)
    accuracy_metric = 'accuracy'
    auroc_metric = tf.keras.metrics.AUC(num_thresholds = 1000, dtype = tf.float32, name='auroc')
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.0005)

    for n in models:
        models[n].compile(optimizer = optimizer,
                          loss = loss,
                          metrics = [accuracy_metric, auroc_metric])
        # export TCMC parameters, experimental, make this an option
        # evo_layer = models[n].get_layer(index=2)
        # evo_layer.export_matrices("rates-Q.txt", "rates-pi.txt")
        # D = models[n].get_layer(index=5)
        # print (D.get_weights())

    # construct a `tf.data.Dataset` from the fasta files    
    # generate a dataset for these files
    dataset = tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.int32, tf.int64, tf.float64))

    # batch and reshape sequences to match the input specification of tcmc
    #ds = database_reader.padded_batch(ds, batch_size, num_leaves, alphabet_size)


    padded_shapes = ([], [], [None, max(num_leaves), alphabet_size])
    dataset = dataset.padded_batch(batch_size, 
                                   padded_shapes = padded_shapes, 
                                   padding_values = (
                                       tf.constant(0, tf.int32), 
                                       tf.constant(0, tf.int64), 
                                       tf.constant(1.0, tf.float64)
                                   ))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(database_reader.concat_sequences, num_parallel_calls = 4)

    # predict on each model
    preds = collections.OrderedDict()

    for n in models:
        model = models[n]
        try:
            pred = model.predict(dataset)
            if sitewise:
                preds[n] = pred.flatten()
            else:
                if num_classes > 2:
                    for c in range(num_classes):
                        n_c = n + "_class_" + str(c)
                        preds[n_c] = pred[:, c]
                else:
                    preds[n] = pred[:, 1]
        except UnboundLocalError:
            pass # happens in tf 2.3 when there is no valid MSA
        del model

    wellformed_msas = [a["fasta_path"] for a in aux]
    preds['path'] = wellformed_msas
    preds.move_to_end('path', last = False) # move MSA file name to front
    
    for p in path_ids_without_reference_clade:
        print(f'The species in "{p}" are not in included in a reference clade. Ignoring it.')
        
    for p in path_ids_with_empty_sequences:
        print(f'The MSA "{p}" is empty (after) codon-aligning it. Ignoring it.')
        
    if sitewise:
        final_preds = {}
        
        for path in preds['path']:
            prev_seq_len = -1
            records = SeqIO.parse(path, "fasta")
            
            for record in records:
                seq_len = int(len(record.seq)/3)
                if classify:
                    # 2 outputs per site for 2 classes
                    seq_len = 2*seq_len
                #check if the sequence lengths in one file are different
                if seq_len != prev_seq_len and prev_seq_len != -1:
                    print("All sequences in one MSA should have the same length! Values could possibly be not correctly assigned to the MSA")
                prev_seq_len = seq_len

            # divide the predictions into parts corresponding to the right sequence 
            # (currently only works for 1 model)
            for n in models:
                final_preds[path] = preds[n][:seq_len]
                preds[n] = preds[n][seq_len:]
                
        return final_preds
    else:
        return preds




def predict_on_tfrecord_files(trial_ids, # OrderedDict of model ids with keys like 'tcmc_rnn'
                              saved_weights_dir,
                              log_dir,
                              clades,
                              tfrecord_paths,
                              use_amino_acids = False,
                              use_codons = False,
                              tuple_length = 1,
                              batch_size = 30,
                              num_classes = 2
):

    # calculate model properties
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length if not use_amino_acids else 20 ** tuple_length
    num_leaves = database_reader.num_leaves(clades)
    buffer_size = 1000
    
    
    
    # load the wanted models and compile them
    models = collections.OrderedDict( (name, recover_model(trial_ids[name], clades, alphabet_size, log_dir, saved_weights_dir)) for name in trial_ids)
    accuracy_metric = 'accuracy'
    auroc_metric = tf.keras.metrics.AUC(num_thresholds = 1000, dtype = tf.float32, name='auroc')
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.0005)

    for n in models:
        models[n].compile(optimizer = optimizer,
                          loss = loss,
                          metrics = [accuracy_metric, auroc_metric])


    # construct a `tf.data.Dataset` from the fasta files    
    # generate a dataset for these files
    parser = partial(database_reader.parse_tfrecord_entry, num_leaves = num_leaves, alphabet_size=alphabet_size)
    datasets = {p: tf.data.TFRecordDataset(p, compression_type = 'GZIP' if p.endswith('.gz') else None, buffer_size = buffer_size) \
                .map(parser, num_parallel_calls = 2) for p in tfrecord_paths}
    

    # batch and reshape sequences to match the input specification of tcmc
    #ds = database_reader.padded_batch(ds, batch_size, num_leaves, alphabet_size)


    padded_shapes = ([], [], [], [None, max(num_leaves), alphabet_size])
    
    for p in tfrecord_paths:
        
        dataset = datasets[p]
        dataset = dataset.padded_batch(batch_size, 
                                       padded_shapes = padded_shapes, 
                                       padding_values = (
                                           tf.constant(0, tf.int32), 
                                           tf.constant(0, tf.int32), 
                                           tf.constant(0, tf.int64), 
                                           tf.constant(1.0, tf.float64)
                                       ))

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        #dataset = dataset.map(database_reader.concatenate_dataset_entries, num_parallel_calls = 4)
        
        # TODO: Pass the variable "num_classes" to database_reader.concatenate_dataset_entries().
        if num_classes == 2:
            dataset = dataset.map(database_reader.concatenate_dataset_entries, num_parallel_calls = 4)
        elif num_classes == 3:
            dataset = dataset.map(database_reader.concatenate_dataset_entries2, num_parallel_calls = 4)
        else:
            raise Exception(f'Currently we only support two and three output classes. Your number of classes:{num_classes}')

        datasets[p] = dataset


    # function to extract meta data from the sequences
    from tf_tcmc.tcmc.tensor_utils import BatchedSequences
    bs_layer = BatchedSequences(feature_size = max(num_leaves), dtype=tf.float64, name="batched_sequences")    
    def sequence_data(X, y):
        sequences, clade_ids, sequence_lengths = X
        #S = tf.transpose(sequences, perm = [1, 0, 2])
        sl = tf.expand_dims(sequence_lengths, axis=-1)

        nontrivial_entries = tf.logical_not(tf.reduce_all(sequences == tf.ones(sequences.shape[-1], dtype=tf.float64), axis=-1))
        nontrivial_entries = tf.cast(nontrivial_entries, dtype=tf.float64)
        nontrivial_entries_batched = bs_layer(nontrivial_entries, sl)
        nontrivial_entries_batched = tf.cast(nontrivial_entries_batched, dtype=tf.bool)
        aligned_sequences = tf.reduce_any(nontrivial_entries_batched, axis=1)
        aligned_sequences = tf.reduce_sum(tf.cast(aligned_sequences, dtype=tf.int64), axis=-1)

        label = tf.cast(tf.argmax(y, axis=1), dtype=tf.int64)
        return (aligned_sequences, sequence_lengths, label)


    # predict on each model
    preds = collections.OrderedDict()
    num_seq = {}

    # wanted sequence meta data
    aligned_sequences = None
    sequence_lengths = None
    Y = None

    for p in tfrecord_paths:

        dataset = datasets[p]

        # evaluate the models
        for n in models:
            model = models[n]
            try:
                pred = model.predict(dataset)
                if num_classes > 2:
                    for c in range(num_classes):
                        n_c = n + "_class_" + str(c)
                        pred_c = pred[:, c]
                        preds[n_c] = np.concatenate((preds[n_c], pred_c)) if n_c in preds else pred_c
                else:
                    pred = pred[:, 1]
                    preds[n] = np.concatenate((preds[n], pred)) if n in preds else pred
                num_seq[p] = pred.shape[0]
            except UnboundLocalError:
                pass # happens in tf 2.3 when there is no valid MSA
            del model

        # extract the meta data
        for num_ali, sl, y in dataset.map(sequence_data).as_numpy_iterator():
            aligned_sequences = np.concatenate((aligned_sequences, num_ali)) if not aligned_sequences is None else num_ali
            sequence_lengths = np.concatenate((sequence_lengths, sl)) if not sequence_lengths is None else sl
            Y = np.concatenate((Y, y)) if not Y is None else y


    filenames = [[p for _ in range(num_seq[p])] for p in tfrecord_paths]
    indices = [list(range(num_seq[p])) for p in tfrecord_paths]
    preds['file'] = list(itertools.chain.from_iterable(filenames))
    preds['index'] = list(itertools.chain.from_iterable(indices))
    preds['aligned_sequences'] = aligned_sequences
    preds['sequence_length'] = sequence_lengths
    preds['y'] = Y


    preds.move_to_end('aligned_sequences', last = False)
    preds.move_to_end('sequence_length', last = False)
    preds.move_to_end('y', last = False)
    preds.move_to_end('index', last = False)
    preds.move_to_end('file', last = False) 


    return preds



 
    
def predict_on_maf_files(trial_ids, # OrderedDict of model ids with keys like 'tcmc_rnn'
                           saved_weights_dir,
                           log_dir,
                           clades,
                           paths,
                           use_codons = True,
                           tuple_length = 1,
                           batch_size = 30,
                           trans_dict = None,
                           remove_stop_rows = False):
    """
     This case is only implemented for 2 classes (binary classification).
    """
    # calculate model properties
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length
    num_leaves = database_reader.num_leaves(clades)
    
    trans_dict = trans_dict if not trans_dict is None else {}
    aux = []
    
    def sequence_generator():
         # conditionally open a .maf or .maf.gz file for input
        for maffile in paths:
            opener = open # for regular text files
            if '.gz' in Path(maffile).suffixes:
                opener = gzip.open
            with opener(maffile, "rt") as msas_file:
                for msa in AlignIO.parse(msas_file, "maf"): 
                    # print ("seqgen MSA", msa)
                    tensor_msas = msa_converter.parse_text_MSA(
                        msa, clades, trans_dict = trans_dict,
                        remove_stop_rows = remove_stop_rows, use_amino_acids = False,     tuple_length = tuple_length, use_codons = use_codons)
                    for (cid, sl, S, auxdata) in tensor_msas: 
                        # filter bad MSAs (trivial or missing reference)
                        if cid < 0:
                            continue
                        aux.append(auxdata)
                        if sl != auxdata['numSites']:
                            sys.die("length mismatch", auxdata)
                        yield cid, sl, S
    
    # load the wanted models and compile them
    models = collections.OrderedDict( (name, recover_model(trial_ids[name], clades, alphabet_size, log_dir, saved_weights_dir)) for name in trial_ids)
    accuracy_metric = 'accuracy'
    auroc_metric = tf.keras.metrics.AUC(num_thresholds = 1000, dtype = tf.float32, name='auroc')
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.0005)

    model = next(iter(models.values())) # only one model is supported for now
    model.compile(optimizer = optimizer, loss = loss, metrics = [accuracy_metric, auroc_metric])

    # construct a `tf.data.Dataset` from the fasta files    
    # generate a dataset for these files
    dataset = tf.data.Dataset.from_generator(sequence_generator, output_types=(tf.int32, tf.int64, tf.float64))

    # batch and reshape sequences to match the input specification of tcmc
    #ds = database_reader.padded_batch(ds, batch_size, num_leaves, alphabet_size)


    padded_shapes = ([], [], [None, max(num_leaves), alphabet_size])
    dataset = dataset.padded_batch(batch_size, 
                                   padded_shapes = padded_shapes, 
                                   padding_values = (
                                       tf.constant(0, tf.int32), 
                                       tf.constant(0, tf.int64), 
                                       tf.constant(1.0, tf.float64)
                                   ))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(database_reader.concat_sequences, num_parallel_calls = 4)

    # predict on each model
    preds = collections.OrderedDict()
        

    try:
        preds = model.predict(dataset)
        # print ("preds", preds.shape)
    except UnboundLocalError:
        pass # happens in tf 2.3 when there is no valid MSA
    
    return preds, aux