import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from functools import partial

from utilities import database_reader
from tf_tcmc.tcmc.tcmc import TCMCProbability

def create_model(forest, 
                 alphabet_size,
                 new_alphabet_size = 0,
                 tcmc_models = 8,
                 num_positions = 1,
                 sparse_rates = False,
                 depth_as_feature = False,
                 frames_as_feature = False,
                 gaps_as_feature = False,
                 conv_filters = 32,
                 conv_kernelsize = 8,
                 conv2_filters = 16,
                 conv2_kernelsize = 4,
                 maxpool_size = 6,
                 dense_dimension = 16,
                 name = "clamsa_ebony",
                 num_classes = 2):

    if new_alphabet_size > 0 and gaps_as_feature:
        sys.exit(f"Error: With gaps as feature new_alphabet_size needs to be 0, instead it is {new_alphabet_size}.")
    
    if conv_kernelsize > tcmc_models:
        sys.stderr.write(f"Warning: conv_kernelsize ({conv_kernelsize}) is larger than tcmc_models ({tcmc_models})! This means the convolution kernel is larger than the channels dimension of the input ot the convolution layer.\n")

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
    reshape_layer = ReshapeBatch(num_positions, name = "reshape") # reshape to (batch_size, tcmc_models, num_positions)
    conv_layer = tf.keras.layers.Conv1D(conv_filters, conv_kernelsize, padding = 'same', kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name = "convolution", dtype=tf.float64)  #padding = "same",
    maxpool_layer = tf.keras.layers.MaxPool1D(maxpool_size, name = "max_pool")
    dropout_layer = tf.keras.layers.Dropout(0.2, name = "dropout")
    conv2_layer = tf.keras.layers.Conv1D(conv2_filters, conv2_kernelsize, padding = 'same', kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name = "convolution2", dtype=tf.float64)
    flatten_layer = tf.keras.layers.Flatten(name = "flatten")

    concat_layer = tf.keras.layers.Concatenate(name = "sequence_and_features")

    if depth_as_feature:
        depth_encoding_layer = Depth(num_positions, name = "depth_sequences") 
        depth_dense_layer = tf.keras.layers.Dense(16, kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name="depth_dense", dtype=tf.float64)

    if gaps_as_feature:
        gap_encoding_layer = EncodeGaps(name = "encoded_gaps")
        gap_tcmc_layer = TCMCProbability((tcmc_models,), forest, num_positions = num_positions, name="P_gap_columns")
        gap_conv_layer = tf.keras.layers.Conv1D(conv_filters, conv_kernelsize, padding = 'same', kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name = "gap_convolution", dtype=tf.float64)
        gap_conv2_layer = tf.keras.layers.Conv1D(conv2_filters, conv2_kernelsize, padding = 'same', kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name = "gap_convolution2", dtype=tf.float64)   

    if frames_as_feature:
        frame_mean_layer = FrameLogLikelihood(alphabet_size, num_positions, name = "frame_mean_log")  # shape (B, 6*M)
        frame_dense_layer = tf.keras.layers.Dense(16, kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name="frame_dense", dtype=tf.float64)
    
    dense_layer = tf.keras.layers.Dense(dense_dimension, kernel_initializer = "TruncatedNormal", activation = "leaky_relu", name="dense", dtype=tf.float64)

    guesses_layer = tf.keras.layers.Dense(num_classes, kernel_initializer = "glorot_normal", activation = "softmax", name = "guesses", dtype=tf.float64)

    # assemble the computational graph
    Encoded_sequences = encoding_layer(sequences) if new_alphabet_size > 0 else sequences
    P = tcmc_layer(Encoded_sequences, clade_ids)
    log_P = log_layer(P)
    X = log_P
    X = reshape_layer(X)

    X = conv_layer(X)
    X = maxpool_layer(X)
    X = dropout_layer(X)
    X = conv2_layer(X)
    X = flatten_layer(X)
    concat = [X]

    if gaps_as_feature:
        encoded_gaps = gap_encoding_layer(sequences)
        P_gaps = gap_tcmc_layer(encoded_gaps, clade_ids)
        log_P_gaps = log_layer(P_gaps)
        
        reshape_gaps = reshape_layer(log_P_gaps)
        gap_conv = gap_conv_layer(reshape_gaps)
        gap_conv = maxpool_layer(gap_conv)
        gap_conv = dropout_layer(gap_conv)
        gap_conv = gap_conv2_layer(gap_conv)
        gap_conv = flatten_layer(gap_conv)

        concat += [gap_conv]
        
    if frames_as_feature:
        frame_mean = frame_mean_layer(log_P)
        frame_dense = frame_dense_layer(frame_mean)
        concat += [frame_dense]

    if depth_as_feature:
        depth = depth_encoding_layer(sequences)
        depth_dense = depth_dense_layer(depth)
        concat += [depth_dense]

    if len(concat) > 1:
        X = concat_layer(concat)

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
    def call(self, inputs):
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
        #self.M = input_shape[-1]
        super(ReshapeBatch, self).build(input_shape)

    def call(self, inputs):
        #X = tf.transpose(tf.reshape(inputs, (-1, self.k, self.M)), perm = (0,2,1))  # shape (B,M,k)
        #X = tf.reshape(inputs, (-1, self.k * self.M))  # shape (B, M * k)
        M = tf.shape(inputs)[-1]
        X = tf.reshape(inputs, (-1, self.k, M))
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
    def call(self, inputs):

        B = tf.shape(inputs)[0]  # size of batch dim
        b = tf.cast(tf.math.round(B/self.k), tf.int32)  # number of msas i.e. batch size, rounding bc of precision errors

        # seq_ids for unsorted_segment_mean
        #seq_ids = np.array([], dtype=np.int32)
        #for i in tf.range(b):
        #    tf.autograph.experimental.set_loop_options(shape_invariants=[(seq_ids, tf.TensorShape([None]))])  # tell tf that seq_id changes shape to unknown
        #    seq_ids = tf.concat([seq_ids, i * 6 + self.left_ids, self.overlap_ids, i * 6 + self.right_ids], 0)
         
        left_len = tf.shape(self.left_ids)[0]
        overlap_len = tf.shape(self.overlap_ids)[0]
        right_len = tf.shape(self.right_ids)[0]

        # allocate memory for seq_ids
        total_len = b * (left_len + overlap_len + right_len)
        seq_ids = tf.zeros([total_len], dtype=tf.int32)

        # fill seq_ids
        for i in tf.range(b):
            start = i * (left_len + overlap_len + right_len)
            seq_ids = tf.tensor_scatter_nd_add(seq_ids, tf.reshape(tf.range(start, start + left_len), (-1,1)), i * 6 + self.left_ids)
            seq_ids = tf.tensor_scatter_nd_add(seq_ids, tf.reshape(tf.range(start + left_len, start + left_len + overlap_len), (-1,1)), self.overlap_ids)
            seq_ids = tf.tensor_scatter_nd_add(seq_ids, tf.reshape(tf.range(start + left_len + overlap_len, start + left_len + overlap_len + right_len), (-1,1)), i * 6 + self.right_ids)
        # get values in TensorArray as Tensor
        #seq_ids = seq_ids.concat()
         
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
    def call(self, inputs):
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


class Depth(tf.keras.layers.Layer): 
    """Encode msa with binary alphabet: 1 for a species present, and gap for species absent"""
    def __init__(self, k, **kwargs):
        self.k = k
        super(Depth, self).__init__(**kwargs)

    def build(self, input_shape):
        # shape1 = (B, N, s)
        self.s = input_shape[-1]
        self.N = input_shape[-2]
        super(Depth, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        sequences = tf.cast(inputs, tf.float64)
        #clade_ids = inputs[1]
        batched_sequences = tf.reshape(sequences, (-1, self.k, self.N, self.s))  # shape (b, k, N, s)
        #depth_ids = tf.reshape(clade_ids, (-1,self.k))[:,0]

        # count ones in alphabet dim and sum over columns of msas, all ones means species not aligned
        sequence_sum = tf.math.reduce_sum(batched_sequences, axis=[1,-1]) # shape (b,N)
        # convert to binary alphabet: 1 means species aligned, 0 means not aligned
        indices = tf.cast(tf.math.not_equal(sequence_sum, tf.constant(self.s * self.k, dtype = tf.float64)), tf.float64) 

        # count present species by summing over N
        depth = tf.reduce_sum(indices, axis = -1, keepdims = True)
        #ones = tf.ones_like(indices, dtype = tf.float64)  # ones the same shape of indices
        # stack to create one-hot encoding, last dim has alphabet size 2
        # this transforms species aligned to character 1 and species not aligned to gap
        #depth_sequences = tf.stack([indices, ones], axis = -1)  
        
        # output shape (b,1)
        # tf.math.log(depth)
        return tf.math.log(depth)

    def get_config(self):
        base_config = super(Depth, self).get_config()
        base_config['k'] = self.k
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
