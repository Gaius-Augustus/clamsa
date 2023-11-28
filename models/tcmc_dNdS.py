# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:17:31 2022

@author: Daniel Meissner
"""

import sys
sys.path.append("..")
import tensorflow as tf
from functools import partial

from utilities import database_reader
from tf_tcmc.tcmc.tcmc import TCMCProbability

def create_model(forest, 
                 alphabet_size,
                 new_alphabet_size = 0,
                 tcmc_models=8,
                 dense_dimension=16,
                 name="clamsa_tcmc_dNdS",
                 sparse_rates = False):
    
    num_leaves = database_reader.num_leaves(forest)
    N = max(num_leaves)
    s = alphabet_size
    
    # define the inputs
    sequences = tf.keras.Input(shape=(N,s), name = "sequences", dtype=tf.float64)
    clade_ids = tf.keras.Input(shape=(), name = "clade_ids", dtype=tf.int32)
    sequence_lengths = tf.keras.Input(shape = (1,), name = "sequence_lengths", dtype = tf.int64) # keras inputs doesn't allow shape [None]


    # define the layers
    encoding_layer = Encode(new_alphabet_size, name='encoded_sequences', dtype=tf.float64) if new_alphabet_size > 0 else None
    tcmc_layer = TCMCProbability((tcmc_models,), forest, sparse_rates = sparse_rates, name="P_sequence_columns")
    log_layer = tf.keras.layers.Lambda(tf.math.log, name="log_P", dtype=tf.float64)
    norm_layer = normalize_by_depth(name = "normalize", dtype=tf.float64)
    
    # maybe change kernel initialization and activation functions
    dense_layer1 = tf.keras.layers.Dense(dense_dimension, kernel_initializer = "TruncatedNormal", activation = "relu", name="dense1")
    dense_layer2 = tf.keras.layers.Dense(dense_dimension, kernel_initializer = "TruncatedNormal", activation = "relu", name="dense2")
    reg_layer = tf.keras.layers.Dense(1, name = "z", kernel_initializer = "TruncatedNormal", dtype=tf.float64)
    exp_layer = tf.keras.layers.Lambda(tf.math.exp, name="exp_z", dtype=tf.float64)
    
    # assemble the computational graph
    Encoded_sequences = encoding_layer(sequences) if new_alphabet_size > 0 else sequences
    P = tcmc_layer(Encoded_sequences, clade_ids)
    log_P = log_layer(P)
    norm_log_P = norm_layer(sequences, log_P)
    dense1 = dense_layer1(norm_log_P)
    dense2 = dense_layer2(dense1)
    z = reg_layer(dense2)
    omega = exp_layer(z)
    

    model = tf.keras.Model(inputs = [sequences, clade_ids, sequence_lengths], outputs = omega, name = name)
    
    return model



def training_callbacks(model, logdir, wanted_callbacks):
    
    return []
    
    tcmc = model.get_layer("P_sequence_columns")
    
    file_writer = tf.summary.create_file_writer(f'{logdir}/images/aa')
    
    log_aa = partial(log_amino_acid_probability_distribution, tcmc=tcmc, file_writer=file_writer, model_id=0, t=0)
    
    aa_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_aa)
    
    return [aa_callback]



class Encode(tf.keras.layers.Layer):
    """Encoding the alphabet"""
    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Encode, self).__init__(**kwargs)

    def build(self, input_shape):
        self.resize_matrix = self.add_weight(shape = (input_shape[-1], self.new_size), name = "resize_matrix", dtype = tf.float64, initializer='uniform', trainable=True)
        super(Encode, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        prop_matrix = tf.keras.activations.softmax(self.resize_matrix)
        return tf.matmul(inputs, prop_matrix)

    def get_config(self):
        base_config = super(Encode, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    

def get_depths(sequences):
    
    #check which arrays in the third dimension contain all ones (corresponds to taxon missing in the alignment)
    all_ones_mask = tf.reduce_all(tf.equal(sequences, 1), axis=2)
    
    #delete missing rows 
    seq_trim = tf.ragged.boolean_mask(sequences, tf.logical_not(all_ones_mask))
    
    #count how many actual taxons are present in one alignment column
    depths = seq_trim.row_lengths(axis=1)
    
    return depths


class normalize_by_depth(tf.keras.layers.Layer):
    """Divide the output from the TCMC layer with the alignment depth (column-wise)"""
    
    def __init__(self, **kwargs):
        super(normalize_by_depth, self).__init__(**kwargs)
               
        
    def build(self, input_shape):
        super(normalize_by_depth, self).build(input_shape)
        
    
    @tf.function(input_signature=(
    tf.TensorSpec(shape=[None,None,None], dtype=tf.float64, name='alignment'),
    tf.TensorSpec(shape=[None,None], dtype=tf.float64, name='log_P_values'),
    ))
    def call(self, sequences, log_P_values):
        
        
        with tf.name_scope("get_depths"):
            depths = get_depths(sequences)
            
        norm_log_P_values = tf.transpose(tf.transpose(log_P_values) / tf.cast(depths, tf.float64))
            
        return norm_log_P_values
    
    
    def get_config(self):
        base_config = super(normalize_by_depth, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
