import sys
sys.path.append("..")
import tensorflow as tf
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
                 name="clamsa_class",
                 num_classes=2):

    num_leaves = database_reader.num_leaves(forest)
    N = max(num_leaves)
    s = alphabet_size

    # define the inputs
    sequences = tf.keras.Input(shape=(N,s), name = "sequences", dtype=tf.float64)
    clade_ids = tf.keras.Input(shape=(), name = "clade_ids", dtype=tf.int32)
    sequence_lengths = tf.keras.Input(shape = (1,), name = "sequence_lengths", dtype = tf.int64) # keras inputs doesn't allow shape [None]

    # define the layers
    encoding_layer = Encode(new_alphabet_size, name='encoded_sequences', dtype=tf.float64) if new_alphabet_size > 0 else None
    tcmc_layer = TCMCProbability((tcmc_models,), forest, num_positions = num_positions, sparse_rates = sparse_rates, name="P_sequence_columns")
    mean_log_layer = NegativeLogLikelihood(name='mean_loglik', dtype=tf.float64)
    guesses_layer = tf.keras.layers.Dense(num_classes, kernel_initializer = "TruncatedNormal", activation = "softmax", name = "guesses", dtype=tf.float64)
    
    # assemble the computational graph
    Encoded_sequences = encoding_layer(sequences) if new_alphabet_size > 0 else sequences
    P = tcmc_layer(Encoded_sequences, clade_ids)
    LL = mean_log_layer([P, sequence_lengths])
    guesses = guesses_layer(LL)
    
    model = tf.keras.Model(inputs = [sequences, clade_ids, sequence_lengths], outputs = [guesses, LL], name = name)

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
    """"""
    def __init__(self, **kwargs):
        super(NegativeLogLikelihood, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NegativeLogLikelihood, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):
        P = inputs[0]
        sl = inputs[1]
        sls = tf.reshape(sl, shape = [-1])

        seq_ids = segment_ids(sls)
        log_P = tf.math.log(P)
        loglikelihood = - tf.math.segment_mean(log_P, seq_ids) 
        
        return loglikelihood 

    def get_config(self):
        base_config = super(NegativeLogLikelihood, self).get_config()
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
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
