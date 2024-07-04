import sys
sys.path.append("..")

import numpy as np
from Bio import SeqIO
import os
import json
import tensorflow as tf
import datetime
import itertools
from pathlib import Path
from matplotlib import pyplot as plt

from tf_tcmc.tcmc.tcmc import TCMCProbability
from tf_tcmc.tcmc.tensor_utils import segment_ids
from tf_tcmc.tcmc.tensor_utils import BatchedSequences
import utilities.onehot_tuple_encoder as ote
from utilities import database_reader
from utilities import msa_converter
from importlib import import_module
import models
from tensorboard.plugins.hparams import api as hp


# On some versions of CuDNN the default LSTM implementation
# raises a warning. The following code deals with these cases
# See [here](https://github.com/tensorflow/tensorflow/issues/36508)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
def train_models(input_dir, 
          basenames,
          clades,
          merge_behaviour = 'evenly',
          split_specifications  = {
              'train': {'name': 'train', 'wanted_models': [0, 1], 'interweave_models': [.67, .33], 'repeat_models': [True, True]},
              'val': {'name': 'val', 'wanted_models': [0, 1], 'interweave_models': True, 'repeat_models': [False, False]},
              'test': {'name': 'test', 'wanted_models': [0, 1], 'interweave_models': True, 'repeat_models': [False, False]},
          },
          tuple_length = 1,
          use_amino_acids = False,
          use_codons = False,
          model_hyperparameters = {
              "tcmc_rnn": {
                  "tcmc_models": [8,],
                  "rnn_type": ["gru",],
                  "rnn_units": [32,],
                  "dense_dimension": [16,],
              }, 
              "tcmc_mean_log": {
                  "tcmc_models": [8,],
                  "sequence_length_as_feature": [False,],
                  "dense1_dimension": [16,],
                  "dense2_dimension": [16,],
              },
          },
          model_training_callbacks = {
              "tcmc_rnn": {},
              "tcmc_mean_log": {},
          },
          batch_size = 30,
          batches_per_epoch = 100,
          epochs = 40,
          save_model_weights = True,
          log_basedir = 'logs',
          saved_weights_basedir = 'saved_weights',
          sitewise = False,
          classify = False,
          sample_weights = False,
          verbose = True,
         ):
    """
    TODO: Write Docstring
    """
    # calculate some features from the input
    num_leaves = database_reader.num_leaves(clades)
    tuple_length = 3 if use_codons else tuple_length
    alphabet_size = 4 ** tuple_length if not use_amino_acids else 20 ** tuple_length

    # evaluate the split specifications
    splits = {'train': None, 'val': None, 'test': None}
    num_classes = 0

    for k in split_specifications:
        if k in splits.keys():
            try:
                splits[k] = database_reader.DatasetSplitSpecification(**split_specifications[k])
                num_classes = len(splits[k].wanted_models) if num_classes < len(splits[k].wanted_models) else num_classes
            except TypeError as te:
                raise Exception(f"Invalid split specification for '{k}': {split_specifications[k]}") from te


    # read the datasets for each wanted basename
    input_dir = os.path.join(input_dir, '') # append '/' if not already there

    wanted_splits = [split for split in splits.values() if split != None ]
    try:
        unmerged_datasets = {b: database_reader.get_datasets(input_dir, b, wanted_splits, num_leaves = num_leaves, alphabet_size = alphabet_size, seed = None, buffer_size = 1000, should_shuffle=True, sitewise = sitewise) for b in basenames}
    except Exception as e:
        print(f'Error while reading the tfrecord datasets for basenames: {basenames}')
        raise(e)

    if any(['train' not in unmerged_datasets[b] for b in basenames]):
        raise Exception("A 'train' split must be specified!")

    # merge the respective splits of each basename
    datasets = {}
    
    merge_behaviour = merge_behaviour if len(merge_behaviour) > 1 else merge_behaviour[0]
    
    weights = len(basenames) * [1/len(basenames)] # evenly is default
    
    if isinstance(merge_behaviour, str):
        if merge_behaviour != "evenly":
            print(f'Unknown merge beheaviour, merging evenly.')
    else:
        if len(merge_behaviour) > 1:
            # check whether the custom weights are correct        
            # expecting a list of weights
            try:
                merge_behaviour = [float(w) for w in merge_behaviour]
            except ValueError:
                print(f'Expected a list of floats in merge_behaviour. However merge_behaviour = {merge_behaviour}')
                print(f'Will use even merge_behaviour = {weights} instead.')
                
            if len(merge_behaviour) == len(basenames) \
               and all([isinstance(x, float) for x in merge_behaviour]) \
               and all([x >= 0 for x in merge_behaviour]) \
               and sum(merge_behaviour) == 1:
                weights = merge_behaviour


    merge_ds = tf.data.experimental.sample_from_datasets
    datasets = {s: merge_ds([unmerged_datasets[b][s] for b in basenames], weights) for s in splits.keys()}

    # prepare datasets for batching
    for split in datasets:

        ds = datasets[split]
        # batch and reshape sequences to match the input specification of tcmc
        ds = database_reader.padded_batch(ds, batch_size, num_leaves, alphabet_size, sitewise)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        #ds = ds.map(database_reader.concatenate_dataset_entries, num_parallel_calls = 4)

        # TODO: Pass the variable "num_classes" to database_reader.concatenate_dataset_entries().
        if sitewise and not classify:
            if sample_weights:
                ds = ds.map(database_reader.concatenate_dataset_entries4, num_parallel_calls = 4)
            else:
                ds = ds.map(database_reader.concatenate_dataset_entries3, num_parallel_calls = 4)
        elif sitewise and classify:
            if sample_weights:
                ds = ds.map(database_reader.concatenate_dataset_entries6, num_parallel_calls = 4)
            else:
                ds = ds.map(database_reader.concatenate_dataset_entries5, num_parallel_calls = 4)
            
        elif num_classes == 2:
            ds = ds.map(database_reader.concatenate_dataset_entries, num_parallel_calls = 4)
        elif num_classes == 3:
            ds = ds.map(database_reader.concatenate_dataset_entries2, num_parallel_calls = 4)
        else:
            raise Exception(f'Currently we only support two and three output classes. Your number of classes:{num_classes}')
       
        datasets[split] = ds

    try:
        for t in datasets['train'].take(1):
            if sample_weights:
                (sequences, clade_ids, sequence_lengths), labels, s_weights = t
            else:
                (sequences, clade_ids, sequence_lengths), labels = t

        if verbose:
            print(f'Example batch of the "train" dataset:\n')
            # extract the clade ids per batch
            padding = [[1,0]]
            ind = tf.pad(tf.cumsum(sequence_lengths), padding)[:-1]
            clade_ids = tf.gather(clade_ids, ind)
            
            if sitewise:
                print("labels: ", labels)
                if sample_weights:
                    print("sample weights: ", s_weights)
            else:
                # extract the label ids
                label_ids = tf.argmax(labels, axis=1)

                print(f'label_ids: {label_ids}')
                
            print(f'clade_ids: {clade_ids}')
            print(f'sequence_length: {sequence_lengths}')
            print(f'sequence_onehot.shape: {sequences.shape}')

            # to debug the sequence first transform it to its
            # original shape
            S = tf.transpose(sequences, perm = [1, 0, 2])

            # decode the sequence and print some columns
            if use_amino_acids:
                alphabet = "ARNDCEQGHILKMFPSTWYV"           
                dec = ote.OnehotTupleEncoder.decode_tfrecord_entry(S.numpy(), alphabet = alphabet, tuple_length = tuple_length, use_bucket_alphabet = False)
            else:
                dec = ote.OnehotTupleEncoder.decode_tfrecord_entry(S.numpy(), tuple_length = tuple_length)
            print(f'first (up to) 8 alignment columns of decoded reshaped sequence: \n{dec[:,:8]}')
    except  tf.errors.InvalidArgumentError as e:
            print(e, file = sys.stderr)
            print("\nClaMSA: InvalidArgumentError during training. ", file = sys.stderr)
            print("This can be caused by training on tfrecord input with a different list of\n",
                  "clades (--clades) than used during conversion.\n",
                  "Use the same list with clamsa train as used with clamsa convert!",
                  file = sys.stderr, sep = "")
            print ("Aborting.", file = sys.stderr)
            sys.exit(1)

    # obtain the model creation functions for the wanted models
    model_creaters = {}
    model_training_callbacks = {}

    for model_name in model_hyperparameters.keys():

        try:
            model_module = import_module(f"models.{model_name}", package=__name__)
        except ModuleNotFoundError as err:
            raise Exception(f'The module "models/{model_name}.py" for the model "{model_name}" does not exist.') from err
        try:
            model_creaters[model_name] = getattr(model_module, "create_model")
        except AttributeError as err:
            raise Exception(f'The model "{model_name}" has no creation function "create_model" in "models/{model_name}.py".')
        try:
            model_training_callbacks[model_name] = getattr(model_module, "training_callbacks")
        except AttributeError as err:
            raise Exception(f'The model "{model_name}" has no training callbacks function "training_callbacks" in "models/{model_name}.py".')


    #prepare the hyperparams for training
    for model_name in model_hyperparameters:
        model_hps = model_hyperparameters[model_name]
        model_hps = {h: hp.HParam(h, hp.Discrete(model_hps[h])) for h in model_hps}
        model_hyperparameters[model_name] = model_hps

    accuracy_metric = 'accuracy'
    auroc_metric = tf.keras.metrics.AUC(num_thresholds = 500, dtype = tf.float32, name='auroc')

    
    
    for model_name in model_hyperparameters:
    
        if verbose:
            print('==================================================================================================')
            print(f'Current model name: "{model_name}"\n')

        # prepare model hyperparameters for iteration
        hps = model_hyperparameters[model_name]
        hp_names = list(hps.keys())
        hp_values = [hps[k].domain.values for k in hp_names]

        # log the wanted hyperparams and metrics 
        str_basenames = '_'.join(basenames)
        logdir = f'{log_basedir}/{str_basenames}/{model_name}'
        if verbose:
            print (f"Logging training in {logdir}")
        with tf.summary.create_file_writer(logdir).as_default():

            hp.hparams_config(
                hparams=list(hps.values()),
                metrics=[hp.Metric('accuracy', group = 'test', display_name='Accuracy'), 
                         hp.Metric('auroc', group = 'test', display_name="AUROC"),
                        ],
            )

        # iterate over all hyperparameter combinations for the current model
        for hp_config in itertools.product(*hp_values):

            # hp for tensorboard callback
            hp_current = {hps[k]: hp_config[i] for i,k in enumerate(hp_names)}

            # hp for model creation
            creation_params = {k: hp_config[i] for i,k in enumerate(hp_names)}


            # determine logging and saving paths
            now_str = datetime.datetime.now().strftime('%Y.%m.%d--%H.%M.%S')

            rundir = f'{logdir}/{now_str}'

            save_weights_dir = f'{saved_weights_basedir}/{str_basenames}/{model_name}'
            Path(save_weights_dir).mkdir(parents=True, exist_ok=True)
            # suffix .weights.h5 required by newer versions of tf.keras.callbacks.ModelCheckpoint
            save_weights_path = f'{save_weights_dir}/{now_str}.weights.h5'

            if verbose: 
                print(f"Current set of hyperparameters: {creation_params}")
                print(f"Weights are stored in: {rundir}")
                print(f"Weights for the best model will be stored in: {save_weights_path}")


            with tf.summary.create_file_writer(rundir).as_default():

                hp.hparams(hp_current, trial_id = now_str)


                create_model = model_creaters[model_name]

                model = create_model(clades, alphabet_size, **creation_params)

                if verbose:
                    print(f'Architecture of the model "{model_name}" with the current hyperparameters:')
                    model.summary()
                    #tf.keras.utils.plot_model(model, show_shapes=True)
                    

                # compile the model for training
                if sitewise and not classify:
                    # change loss function
                    class MeanSquaredLogarithmicError(tf.keras.losses.Loss):
                        
                        def call(self, y_true, y_pred):
                            return tf.reduce_mean(tf.math.square(tf.math.log(y_pred) - tf.math.log(y_true)), axis=-1)
                        
                    
                    class SelectionRecall(tf.keras.metrics.Metric):
                        """
                           classifies selection into 3 classes
                           and returns recall for a single class c.
                        """
                        def __init__(self, name="sel_recall", c = 0, **kwargs):
                            super(SelectionRecall, self).__init__(name=name+str(c), **kwargs)
                            self.s_correct = self.add_weight(name = "sc", initializer="zeros")
                            self.s_total = self.add_weight(name = "st", initializer="zeros")
                            self.c = c
                        
                        def update_state(self, y_true, y_pred, sample_weight=None):
                            
                            # write 2 for positive selection(omega>=1.2)
                            true_pos = tf.where(tf.math.greater_equal(y_true, 1.2), tf.constant(2.0, dtype=tf.float64), y_true)
                            pred_pos = tf.where(tf.math.greater_equal(y_pred, 1.2), tf.constant(2.0, dtype=tf.float64), y_pred)
                            
                            #write 0 for negative(omega <= 0.8)
                            true_pos_neg = tf.where(tf.math.less_equal(true_pos, 0.8), tf.constant(0.0, dtype=tf.float64), true_pos)
                            pred_pos_neg = tf.where(tf.math.less_equal(pred_pos, 0.8), tf.constant(0.0, dtype=tf.float64), pred_pos)
                            
                            #write 1 for neutral(1.2 >= omega >= 0.8)
                            neutral_sel_mask_true = tf.math.logical_and(
                                                    tf.math.not_equal(tf.constant(2.0, dtype=tf.float64), true_pos_neg),
                                                    tf.math.not_equal(tf.constant(0.0, dtype=tf.float64), true_pos_neg))
                            
                            neutral_sel_mask_pred = tf.math.logical_and(
                                                    tf.math.not_equal(tf.constant(2.0, dtype=tf.float64), pred_pos_neg),
                                                    tf.math.not_equal(tf.constant(0.0, dtype=tf.float64), pred_pos_neg))
                            
                            true_all = tf.where(neutral_sel_mask_true, tf.constant(1.0, dtype=tf.float64), true_pos_neg)
                            pred_all = tf.where(neutral_sel_mask_pred, tf.constant(1.0, dtype=tf.float64), pred_pos_neg)
                            
                            
                            
                            cf_mat = tf.math.confusion_matrix(tf.squeeze(true_all), tf.squeeze(pred_all), num_classes=3)
                            
                            #number of correct predictions for each class
                            diag = tf.linalg.tensor_diag_part(cf_mat)
                            
                            #total number of examples for each class
                            total_per_class = tf.reduce_sum(cf_mat, axis=1)
                            
                            
                            #only assign values for class c (metric results can't be arrays?)
                            self.s_correct.assign_add(tf.cast(tf.gather(diag, self.c), dtype = tf.float32))
                            self.s_total.assign_add(tf.cast(tf.gather(total_per_class, self.c), dtype = tf.float32))
                            
                        def reset_state(self):
                            self.s_correct.assign(0)
                            self.s_total.assign(0)
                            
                        def result(self):
                            #compute recall
                            return (self.s_correct / tf.maximum(1.0, self.s_total))
                            
                    class SelectionPrecision(tf.keras.metrics.Metric):
                        """
                           classifies selection into 3 classes
                           and returns precision for a single class c.
                        """
                        def __init__(self, name="sel_precision", c = 0, **kwargs):
                            super(SelectionPrecision, self).__init__(name=name+str(c), **kwargs)
                            self.s_correct = self.add_weight(name = "sc", initializer="zeros")
                            self.s_total = self.add_weight(name = "st", initializer="zeros")
                            self.c = c
                        
                        def update_state(self, y_true, y_pred, sample_weight=None):
                            
                            # write 2 for positive selection(omega>=1.2)
                            true_pos = tf.where(tf.math.greater_equal(y_true, 1.2), tf.constant(2.0, dtype=tf.float64), y_true)
                            pred_pos = tf.where(tf.math.greater_equal(y_pred, 1.2), tf.constant(2.0, dtype=tf.float64), y_pred)
                            
                            #write 0 for negative(omega <= 0.8)
                            true_pos_neg = tf.where(tf.math.less_equal(true_pos, 0.8), tf.constant(0.0, dtype=tf.float64), true_pos)
                            pred_pos_neg = tf.where(tf.math.less_equal(pred_pos, 0.8), tf.constant(0.0, dtype=tf.float64), pred_pos)
                            
                            #write 1 for neutral(1.2 >= omega >= 0.8)
                            neutral_sel_mask_true = tf.math.logical_and(
                                                    tf.math.not_equal(tf.constant(2.0, dtype=tf.float64), true_pos_neg),
                                                    tf.math.not_equal(tf.constant(0.0, dtype=tf.float64), true_pos_neg))
                            
                            neutral_sel_mask_pred = tf.math.logical_and(
                                                    tf.math.not_equal(tf.constant(2.0, dtype=tf.float64), pred_pos_neg),
                                                    tf.math.not_equal(tf.constant(0.0, dtype=tf.float64), pred_pos_neg))
                            
                            true_all = tf.where(neutral_sel_mask_true, tf.constant(1.0, dtype=tf.float64), true_pos_neg)
                            pred_all = tf.where(neutral_sel_mask_pred, tf.constant(1.0, dtype=tf.float64), pred_pos_neg)
                            
                            
                            
                            cf_mat = tf.math.confusion_matrix(tf.squeeze(true_all), tf.squeeze(pred_all), num_classes=3)
                            
                            #number of correct predictions for each class
                            diag = tf.linalg.tensor_diag_part(cf_mat)
                            
                            #total number of predictions for each class
                            total_per_class = tf.reduce_sum(cf_mat, axis=0)
                            
                            
                            #only assign values for class c (metric results can't be arrays?)
                            self.s_correct.assign_add(tf.cast(tf.gather(diag, self.c), dtype = tf.float32))
                            self.s_total.assign_add(tf.cast(tf.gather(total_per_class, self.c), dtype = tf.float32))
                            
                        def reset_state(self):
                            self.s_correct.assign(0)
                            self.s_total.assign(0)
                            
                        def result(self):
                            #compute precision
                            return (self.s_correct / tf.maximum(1.0, self.s_total))
                        
                        
                        
                    loss = MeanSquaredLogarithmicError()
                    optimizer = tf.keras.optimizers.Adam(0.0005)
                    model.compile(optimizer = optimizer,
                                  loss = loss,
                                  metrics = [SelectionRecall(c=0),SelectionRecall(c=1),SelectionRecall(c=2),
                                            SelectionPrecision(c=0), SelectionPrecision(c=1), SelectionPrecision(c=2)])
                    
                elif sitewise and classify:
                    loss = tf.keras.losses.CategoricalCrossentropy()
                    optimizer = tf.keras.optimizers.Adam(0.0005)

                    model.compile(optimizer = optimizer,
                                  loss = loss,
                                  metrics = [accuracy_metric],
                                  weighted_metrics = [],
                                  )
                else:
                    loss = tf.keras.losses.CategoricalCrossentropy()
                    optimizer = tf.keras.optimizers.Adam(0.0005)

                    model.compile(optimizer = optimizer,
                                  loss = loss,
                                  metrics = [accuracy_metric, auroc_metric],
                                  )
                
                # define callbacks during training
                checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath = save_weights_path, 
                                                                   monitor = 'val_loss', 
                                                                   mode = 'min', 
                                                                   save_best_only = True,
                                                                   save_weights_only = True,
                                                                   verbose = 1,
                )

                # Function to decrease learning rate by 'factor' 
                learnrate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                                    mode = 'min', 
                                                                    factor = 0.75, 
                                                                    patience = 4, 
                                                                    verbose = 1,
                )


                tensorboard_cb = tf.keras.callbacks.TensorBoard(rundir, histogram_freq=0, write_graph=True, write_images=True)

                #plot_cb = PlotLearning()

                callbacks = [tensorboard_cb,]

                if datasets['val'] != None:
                    callbacks = callbacks + [checkpoint_cb, learnrate_cb, ]
                    
                training_callbacks = model_training_callbacks[model_name]
                callbacks = callbacks + training_callbacks(model, rundir, wanted_callbacks=None)

                history = model.fit(datasets['train'], 
                          validation_data = datasets['val'], 
                          callbacks = callbacks,
                          epochs = epochs, 
                          steps_per_epoch = batches_per_epoch, 
                          verbose = verbose,
                )
                
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title("Model Loss")
                plt.ylabel("Loss")
                plt.xlabel("Epoche")
                plt.legend(["train", "val"], loc = "upper right")
                plt.savefig("loss_plot.png")
                #plt.show()
                plt.clf()
                
                if classify:
                    plt.plot(history.history['val_accuracy'])   
                    plt.title("Modell \"accuracy\"")
                    plt.ylabel("\"accuracy\"")
                    plt.xlabel("Epoche")
                    plt.legend(["\"accuracy\""], loc = "lower right")
                    plt.savefig("accuracy_plot.png")
                    plt.clf()
                    
                    #plt.plot(history.history['val_auroc'])   
                    #plt.title("Modell AUC")
                    #plt.ylabel("AUC")
                    #plt.xlabel("Epoche")
                    #plt.legend(["AUC"], loc = "lower right")
                    #plt.savefig("auroc_plot.png")
                    #plt.clf()

                else:
                    plt.plot(history.history['val_sel_recall0'])
                    plt.plot(history.history['val_sel_recall1'])
                    plt.plot(history.history['val_sel_recall2'])     
                    plt.title("Modell Sensitivität")
                    plt.ylabel("Sensitivität")
                    plt.xlabel("Epoche")
                    plt.legend(["Sensitivität0", "Sensitivität1", "Sensitivität2"], loc = "lower right")
                    plt.savefig("recall_plot.png")
                    plt.clf()
                    
                    plt.plot(history.history['val_sel_precision0'])
                    plt.plot(history.history['val_sel_precision1'])
                    plt.plot(history.history['val_sel_precision2'])     
                    plt.title("Modell Genauigkeit")
                    plt.ylabel("Genauigkeit")
                    plt.xlabel("Epoche")
                    plt.legend(["Genauigkeit0", "Genauigkeit1", "Genauigkeit2"], loc = "lower right")
                    plt.savefig("precision_plot.png")
                    plt.clf()

                # load 'best' model weights and eval the test dataset
                if datasets['test'] != None:
                    if verbose:
                        print("Evaluating the 'test' dataset:")
                    if sitewise and not classify:
                        test_loss, test_rec0, test_rec1, test_rec2, test_prec0, test_prec1, test_prec2 = model.evaluate(datasets['test'])
                        with tf.summary.create_file_writer(f'{rundir}/test').as_default():
                            tf.summary.scalar('loss', test_loss, step=1)
                            tf.summary.scalar('recall_0', test_rec0, step=1)
                            tf.summary.scalar('recall_1', test_rec1, step=1)
                            tf.summary.scalar('recall_2', test_rec2, step=1)
                            
                            tf.summary.scalar('precision_0', test_prec0, step=1)
                            tf.summary.scalar('precision_1', test_prec1, step=1)
                            tf.summary.scalar('precision_2', test_prec2, step=1)
                    elif sitewise and classify:
                        test_loss, test_acc = model.evaluate(datasets['test'])
                        with tf.summary.create_file_writer(f'{rundir}/test').as_default():
                            tf.summary.scalar('accuracy', test_acc, step=1)
                            #tf.summary.scalar('auroc', test_auroc, step=1)
                            tf.summary.scalar('loss', test_loss, step=1)
                    else:
                        test_loss, test_acc, test_auroc = model.evaluate(datasets['test'])
                        with tf.summary.create_file_writer(f'{rundir}/test').as_default():
                            tf.summary.scalar('accuracy', test_acc, step=1)
                            tf.summary.scalar('auroc', test_auroc, step=1)
                            tf.summary.scalar('loss', test_auroc, step=1)
                    
    return 0
