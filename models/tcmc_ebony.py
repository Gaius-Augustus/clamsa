import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from functools import partial

from utilities import database_reader
from tf_tcmc.tcmc.tcmc import TCMCProbability
from tf_tcmc.tcmc.tensor_utils import segment_ids

def create_model(forest, 
                 alphabet_size,
                 new_alphabet_size = 0,
                 tcmc_models = 8,
                 num_positions = 1,
                 sparse_rates = False,
                 frames_as_feature = False,
                 gaps_as_feature = False,
                 sequence_dense_dimension = 32,
                 dense_dimension = 16,
                 name = "clamsa_ebony",
                 num_classes = 2):

    num_leaves = database_reader.num_leaves(forest)
    N = max(num_leaves)
    s = alphabet_size

    # define the inputs
    sequences = tf.keras.Input(shape=(N,s), name = "sequences", dtype=tf.float64)
    clade_ids = tf.keras.Input(shape=(), name = "clade_ids", dtype=tf.int32)
    sequence_lengths = tf.keras.Input(shape = (1,), name = "sequence_lengths", dtype = tf.int64) # keras inputs doesn't allow shape [None]

    # define the layers
    encoding_layer = Encode(new_alphabet_size, name='encoded_sequences', dtype=tf.float64) if new_alphabet_size > 0 else None
    tcmc_layer = TCMCProbability((tcmc_models,), forest, num_positions = num_positions, sparse_rates = sparse_rates, name="P_sequence_columns") # shape (batch_size * num_positions, tcmc_models)
    log_layer = tf.keras.layers.Lambda(tf.math.log, name="log_P", dtype=tf.float64)
    reshape_layer = ReshapeBatch(num_positions, name = "reshape") # reshape to (batch_size, tcmc_models * num_positions)
    #mean_log_layer = tf.keras.layers.Lambda(lambda x: - tf.math.reduce_mean(x, axis=-1), name="neg_mean_log", dtype=tf.float64)  # for LL loss
    sequence_dense_layer = tf.keras.layers.Dense(sequence_dense_dimension, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="sequence_dense", dtype=tf.float64)

    concat_layer = None

    if gaps_as_feature:
        gap_encoding_layer = EncodeGaps(name = "encoded_gaps")
        gap_tcmc_layer = TCMCProbability((tcmc_models,), forest, num_positions = num_positions, name="P_gap_columns")
        gap_dense_layer = tf.keras.layers.Dense(16, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="gap_dense", dtype=tf.float64)
        concat_layer = tf.keras.layers.Concatenate(name = "sequence_and_features")

    if frames_as_feature:
        frame_mean_layer = FrameLogLikelihood(alphabet_size, num_positions, name = "frame_mean_log")  # shape (B, 6*M)
        frame_dense_layer = tf.keras.layers.Dense(16, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="frame_dense", dtype=tf.float64)
        if concat_layer == None:
            concat_layer = tf.keras.layers.Concatenate(name = "sequence_and_features")
    
    dense_layer = tf.keras.layers.Dense(dense_dimension, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="dense", dtype=tf.float64) \
        if dense_dimension > 0 else None

    guesses_layer = tf.keras.layers.Dense(num_classes, kernel_initializer = "TruncatedNormal", activation = "softmax", name = "guesses", dtype=tf.float64)

    # assemble the computational graph
    Encoded_sequences = encoding_layer(sequences) if new_alphabet_size > 0 else sequences
    P = tcmc_layer(Encoded_sequences, clade_ids)
    log_P = log_layer(P)
    X = reshape_layer(log_P)
    #LL = mean_log_layer(X)
    X = sequence_dense_layer(X)

    if gaps_as_feature:
        encoded_gaps = gap_encoding_layer(sequences)
        P_gaps = gap_tcmc_layer(encoded_gaps, clade_ids)
        log_P_gaps = log_layer(P_gaps)
        reshape_gaps = reshape_layer(log_P_gaps)
        gap_dense = gap_dense_layer(reshape_gaps)
        X = concat_layer([X, gap_dense])
        
    if frames_as_feature:
        frame_mean = frame_mean_layer(log_P)
        frame_dense = frame_dense_layer(frame_mean)
        X = concat_layer([X, frame_dense])

    if dense_layer != None:
        X = dense_layer(X)

    guesses = guesses_layer(X)
    
    model = tf.keras.Model(inputs = [sequences, clade_ids, sequence_lengths], outputs = [guesses], name = name)

    return model


def training_callbacks(model, logdir, wanted_callbacks):

    return []
    tcmc = model.get_layer("P_sequence_columns")

    file_writer_aa = tf.summary.create_file_writer(f'{logdir}/images/aa')
    file_writer_gen = tf.summary.create_file_writer(f'{logdir}/images/Q')

    log_aa = partial(log_amino_acid_probability_distribution, tcmc=tcmc, file_writer=file_writer_aa, model_id=0, t=1)
    log_gen = partial(log_generator, tcmc=tcmc, file_writer=file_writer_gen, model_id=0)

    aa_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_aa)

    gen_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_gen)

    return [aa_callback, gen_callback]


class NegativeLogLikelihood(tf.keras.layers.Layer):
    """Negative mean loglikelihood of each input MSA"""
    def __init__(self, k, **kwargs):
        self.k = k
        super(NegativeLogLikelihood, self).__init__(**kwargs)

    def build(self, input_shape):
        self.M = input_shape[-1]
        super(NegativeLogLikelihood, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):
        P = tf.reshape(inputs, (-1, self.k * self.M))
        loglikelihood = - tf.math.reduce_mean(P, axis=-1) 

        return loglikelihood 

    def get_config(self):
        base_config = super(NegativeLogLikelihood, self).get_config()
        base_config['k'] = self.k
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class ReshapeBatch(tf.keras.layers.Layer):
    """Reshape batch dimension"""
    def __init__(self, k, **kwargs):
        self.k = k
        super(ReshapeBatch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.M = input_shape[-1]
        super(ReshapeBatch, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):
        #X = tf.transpose(tf.reshape(inputs, (-1, self.k, self.M)), perm = (0,2,1))  # shape (B,M,k)
        X = tf.reshape(inputs, (-1, self.k * self.M))  # shape (B, M * k)
        return X

    def get_config(self):
        base_config = super(ReshapeBatch, self).get_config()
        base_config['k'] = self.k
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class FrameLogLikelihood(tf.keras.layers.Layer):
    """Mean loglikelihood of each frame on each side of the boundary"""
    def __init__(self, s, k, **kwargs):
        self.k = k
        self.s = s

        # get tuple length
        if s in [4,20]:
            tuple_length = 1
        else:
            max_tuple_length = 10
            min_tuple_length = 2
            nuc_s = [4 ** i for i in range(min_tuple_length, max_tuple_length)]
            amino_s = [20 ** i for i in range(min_tuple_length, max_tuple_length)]
            if s in nuc_s:
                tuple_length = nuc_s.index(s) + min_tuple_length
            elif s in amino_s:
                tuple_length = amino_s.index(s) + min_tuple_length
            else:
                raise ValueError(f"Currently only dna (4) and amino acid (20) alphabets are supported by this model. This means, that\
                your input alphabet size s (s={s}) must be 4**t (dna) or 20**t (amino acids) for tuple length t with {max_tuple_length} >= t >= 1.")

        # tuple length needs to be < k 
        if tuple_length >= k:
            l = k + tuple_length - 1  # length of original sequence
            raise ValueError(f"The tuple length t needs to be at most as long as half the sequence length l before encoding (s <= l/2) i.e.\
                less than the length of the encoded sequence k (t < k). Current values are t={tuple_length}, k={k}, l={l}, with t >= k.")

        # prepare for seq_ids
        n = np.int32(1 + (k - tuple_length - 1)/2)  # number of tuples per side that do not overlap the other side
        self.left_ids = np.resize([0,1,2], n)
        self.right_ids = np.resize([3,4,5], n)
        # negative ids for tuples that overlap to the other side, they get ignored by unsorted_segment_mean
        self.overlap_ids = np.resize([-1], np.int32(k-2*n)) if k > 2*n  else np.array([], dtype=np.int32) 

        super(FrameLogLikelihood, self).__init__(**kwargs)

    def build(self, input_shape):
        self.M = input_shape[-1]
        super(FrameLogLikelihood, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):

        B = tf.shape(inputs)[0]  # size of batch dim
        b = tf.cast(tf.math.round(B/self.k), tf.int32)  # number of msas i.e. batch size, rounding bc of precision errors

        # seq_ids for unsorted_segment_mean
        seq_ids = np.array([], dtype=np.int32)
        for i in tf.range(b):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(seq_ids, tf.TensorShape([None]))])  # tell tf that seq_id changes shape to unknown
            seq_ids = tf.concat([seq_ids, i * 6 + self.left_ids, self.overlap_ids, i * 6 + self.right_ids], 0)
         
        framelikelihood = tf.math.unsorted_segment_mean(inputs, seq_ids, 6 * b) 
        #framelikelihood = tf.transpose(tf.reshape(framelikelihood, (-1, 6, self.M)), perm = (0,2,1))  # shape (B,M,6)
        framelikelihood = tf.reshape(framelikelihood, (b, 6 * self.M))  # shape (B,6*M)
        
        return framelikelihood 

    def get_config(self):
        base_config = super(FrameLogLikelihood, self).get_config()
        base_config['k'] = self.k
        base_config['s'] = self.s
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncodeGaps(tf.keras.layers.Layer): 
    """Encode sequences with binary alphabet: position has gap or no gap"""
    def __init__(self, **kwargs):
        super(EncodeGaps, self).__init__(**kwargs)

    def build(self, input_shape):
        self.s = input_shape[-1]
        super(EncodeGaps, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):
        sequence_sum = tf.math.reduce_sum(inputs, axis=-1)  # count ones in alphabet dim, all ones means gap
        indices = tf.cast(sequence_sum == self.s, tf.int32)  # convert to binary alphabet: 1 means gap, 0 mean no gap
        gap_sequences = tf.one_hot(indices, 2)  # one-hot encode binary alphabet

        return gap_sequences

    def get_config(self):
        base_config = super(EncodeGaps, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
        base_config['new_size'] = self.new_size
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
