import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2
from protobuf_to_dict import protobuf_to_dict
from importlib import import_module
from pathlib import Path
import inspect
from Bio import SeqIO
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
        
        
def recover_model(trial_id, forest, alphabet_size, log_dir, saved_weights_dir, model_pred_config = None):
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

    if model_pred_config: # add (overwrite) parameters to kwargs dictionary
        hps = {**hps, **model_pred_config}

    # obtain the model with the correct weights
    model = create_model(forest, alphabet_size, **hps)
    weights_path = get_weights_path(trial_id, saved_weights_dir)
    # the following call can print a warning like: Skipping loading of weights for layer P_sequence_columns due to mismatch in shape
    # it cannot be turned off with logging.set_verbosity(0), however.
    model.load_weights(weights_path, by_name = True, skip_mismatch = True)
    
    return model




def parse_fasta_file(fasta_path, clades, margin_width=0):
    
    species = [msa_converter.leaf_order(c,use_alternatives=True) for c in clades] if clades != None else []
    
    with gzip.open(fasta_path, 'rt') if fasta_path.endswith('.gz') else open(fasta_path, 'r') as fasta_file:            
        entries = [rec for rec in SeqIO.parse(fasta_file, "fasta")]
    # parse the species names
    spec_in_file = [e.id.split('|')[0] for e in entries]

    # compare them with the given references
    ref_ids = [[(r,i) for r in range(len(species))  for i in range(len(species[r])) if s in species[r][i] ] for s in spec_in_file]

    # check if these are contained in exactly one reference clade
    n_refs = [len(x) for x in ref_ids]

    if 0 == min(n_refs) or max(n_refs) > 1:
        return None

    ref_ids = [x[0] for x in ref_ids]

    if len(set(r for (r,i) in ref_ids)) > 1:
        return None

    # the first entry of the fasta file has the header informations
    header_fields = entries[0].id.split("|")

    # read the sequences and trim them if wanted
    sequences = [str(rec.seq).lower() for rec in entries]
    sequences = sequences[margin_width:-margin_width] if margin_width > 0 else sequences

    msa = msa_converter.MSA(
        model = None,
        chromosome_id = None, 
        start_index = None,
        end_index = None,
        is_on_plus_strand = True if len(header_fields) < 5 or header_fields[4] != 'revcomp' else False,
        frame = int(header_fields[2][-1]),
        spec_ids = ref_ids,
        offsets = [],
        sequences = sequences
    )
    # Use the correct onehot encoded sequences
    coded_sequences = msa.coded_codon_aligned_sequences if msa.use_codons or msa.tuple_length > 1 else msa.coded_sequences
    
    # Infer the length of the sequences
    sequence_length = len(coded_sequences[1])  
    
    if sequence_length == 0:
        return None

    # cardinality of the alphabet that has been onehot-encoded
    s = coded_sequences.shape[-1]
    
    # get the id of the used clade and leaves inside this clade
    clade_id = msa.spec_ids[0][0]
    num_species = max([len(specs) for specs in species])
    leaf_ids = [l for (c,l) in msa.spec_ids]
    
    
    # embed the coded sequences into a full MSA for the whole leaf-set of the given clade
    S = np.ones((num_species, sequence_length, s), dtype = np.int32)
    S[leaf_ids,...] = coded_sequences
    
    # make the shape conform with the usual way datasets are structured,
    # namely the columns of the MSA are the examples and should therefore
    # be the first axis
    S = np.transpose(S, (1,0,2))
    
    return clade_id, sequence_length, S

    
    
    
    
def predict_on_fasta_files(trial_ids, # OrderedDict of model ids with keys like 'tcmc_rnn'
                           saved_weights_dir,
                           log_dir,
                           clades,
                           fasta_paths,
                           use_amino_acids = False,
                           use_codons = False,
                           tuple_length = 1,
                           batch_size = 30,
                           trans_dict = None,
                           remove_stop_rows = False,
                           num_classes = 2,
                           model_pred_config = None
):
    # calculate model properties
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length if not use_amino_acids else 20 ** tuple_length
    num_leaves = database_reader.num_leaves(clades)
    
    trans_dict = trans_dict if not trans_dict is None else {}
    
    # import the fasta files and filter out empty codon aligned sequences
    path_ids_without_reference_clade = set()
    path_ids_with_empty_sequences = set()
    
    def sequence_generator():
        for f in fasta_paths:
            if f == "":
                path_ids_with_empty_sequences.add(f)
                continue
            # filter fasta files that have no valid reference clade
            cid, sl, S = msa_converter.parse_fasta_file(f, clades, trans_dict=trans_dict, remove_stop_rows=remove_stop_rows, 
                                                        use_amino_acids = use_amino_acids, tuple_length = tuple_length, use_codons = use_codons)
            if cid == -1:
                path_ids_without_reference_clade.add(f)
                continue
            if cid == -2:
                path_ids_with_empty_sequences.add(f)
                continue

            yield cid, sl, S

    # load the wanted models and compile them
    models = collections.OrderedDict( (name, recover_model(trial_ids[name], clades, alphabet_size, log_dir, saved_weights_dir, model_pred_config)) for name in trial_ids)
    accuracy_metric = 'accuracy'
    auroc_metric = tf.keras.metrics.AUC(num_thresholds = 1000, dtype = tf.float32, name='auroc')
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(0.0005)

    for n in models:
        model = models[n]
        model.compile(optimizer = optimizer,
                      loss = loss,
                      metrics = [accuracy_metric, auroc_metric])
        
        """ For the very special purpose when the rate matrices are to used with an external program, output
        the parameters upon request into flat files.
        """
        if model_pred_config and "get_flat_pars" in model_pred_config and model_pred_config["get_flat_pars"]:
        # export TCMC parameters
            print ("Exporting model parameters to text files...")
            print (model.summary())
            evo_layer = model.get_layer(index=2)
            qFileName = "rates-Q.txt"
            piFileName = "rates-pi.txt"
            print ("Writing full rate matrices to ", qFileName, " and stationary distributions to ", piFileName)
            evo_layer.export_matrices("rates-Q.txt", "rates-pi.txt")
            D = models[n].get_layer(index=5)
            Theta = D.get_weights()
            print ("Writing logistic regression parameters of last layer:\n", Theta)
            thetaFileName = "Theta.txt"
            biasFileName = "biases.txt"
            Theta[0].tofile(thetaFileName, sep='\n')
            Theta[1].tofile(biasFileName, sep='\n')

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
            res = model.predict(dataset)
            if type(res) is list:
                pred = res[0]
                mean_log_P = res[1]
            else:
                pred = res
                mean_log_P = None
            if num_classes > 2:
                for c in range(num_classes):
                    n_c = n + "_class_" + str(c)
                    preds[n_c] = pred[:, c]
            else: # for 2 classes output only the probability of being positive
                preds[n] = pred[:, 1]
            # log likelihoodif s if requested
            if mean_log_P is not None:
                M = mean_log_P.shape[1]
                for m in range(M):
                    ll_m = n + "ll" + str(m)
                    preds[ll_m] = mean_log_P[:, m]
        except UnboundLocalError:
            pass # happens in tf 2.3 when there is no valid MSA
        del model

    wellformed_msas = [f for f in fasta_paths if f not in path_ids_with_empty_sequences and f not in path_ids_without_reference_clade]
    preds['path'] = wellformed_msas
    preds.move_to_end('path', last = False) # move MSA file name to front
    
    for p in path_ids_without_reference_clade:
        print(f'The species in "{p}" are not in included in a reference clade. Ignoring it.')
        
    for p in path_ids_with_empty_sequences:
        print(f'The MSA "{p}" is empty (after) codon-aligning it. Ignoring it.')
    
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
                              num_classes = 2,
                              model_pred_config = None
):

    # calculate model properties
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length if not use_amino_acids else 20 ** tuple_length
    num_leaves = database_reader.num_leaves(clades)
    buffer_size = 1000
    
    
    
    # load the wanted models and compile them
    models = collections.OrderedDict( (name, recover_model(trial_ids[name], clades, alphabet_size, log_dir, saved_weights_dir, model_pred_config)) for name in trial_ids)
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

        model = tf.cast(tf.argmax(y, axis=1), dtype=tf.int64)
        return (aligned_sequences, sequence_lengths, model)


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
