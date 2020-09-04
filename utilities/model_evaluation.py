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

import sys
sys.path.append("..")

from utilities import database_reader
from utilities import msa_converter

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
    
    print ("Getting information for model ", trial_id, "in directory", log_dir)
    hps, model_name = get_model_information(trial_id, log_dir)
    print ("The models name is", model_name)
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
    model.load_weights(weights_path)
    
    return model







def parse_fasta_file(fasta_path, clades, margin_width=0):
    
    species = [msa_converter.leaf_order(c,use_alternatives=True) for c in clades] if clades != None else []
    
    entries = [rec for rec in SeqIO.parse(fasta_path, "fasta")]
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
    # Use the correct onehot encded sequences
    coded_sequences = msa.coded_codon_aligned_sequences if use_codons else msa.coded_sequences
    
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
                           use_codons = True,
                           batch_size = 30,
                           trans_dict = dict()
):
    # calculate model properties
    word_len = 3 # codon size or other tuples
    entry_length = word_len if use_codons else 1
    alphabet_size = 4 ** entry_length
    num_leaves = database_reader.num_leaves(clades)
    
    
    # import the fasta files and filter out empty codon aligned sequences
    fasta_sequences = [msa_converter.parse_fasta_file(f, clades, use_codons, trans_dict=trans_dict) for f in fasta_paths]
    
    # filter fasta files that have no valid reference clade
    malformed = []
    malformed1 = [] # due to id mismatch
    malformed2 = [] # due to length
    wellformed_msas = []
    for i, s in enumerate(fasta_sequences):
        if s is None:
            malformed.append(i)
            malformed2.append(i)
            continue

        (cid, alen, seqs) = s
        if alen < word_len:
            malformed.append(i)
            malformed2.append(i)
        elif cid < 0:
            malformed.append(i)
            malformed1.append(i)
        else:
            wellformed_msas.append({'path': fasta_paths[i], 'sequence': seqs, 'clade_id': cid, 'sequence_length': alen})
            
    # print those paths where no valid reference clade was found
    for i in malformed1:
        print(f'The species in "{fasta_paths[i]}" are not in included in a reference clade. Ignoring it.')
    # print paths to empty too short MSAs
    for i in malformed2:
        print(f'The MSA "{fasta_paths[i]}" has a length < {word_len}. Ignoring it.')

    if len(wellformed_msas) == 0:
        return collections.OrderedDict({'path':[]})
    
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
    dataset = tf.data.Dataset.from_generator(lambda: [(seq['clade_id'], seq['sequence_length'], seq['sequence']) for seq in wellformed_msas], (tf.int32, tf.int64, tf.float64))

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
    preds['path'] = [s['path'] for s in wellformed_msas]
    
    for n in models:        
        model = models[n]
        preds[n] = model.predict(dataset)[:,1]
        del model
        
    return preds
