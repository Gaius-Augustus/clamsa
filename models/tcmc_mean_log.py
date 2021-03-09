import sys
sys.path.append("..")
import tensorflow as tf
from functools import partial



from utilities import database_reader
from utilities import visualization
from tf_tcmc.tcmc.tcmc import TCMCProbability
from tf_tcmc.tcmc.tensor_utils import segment_ids



def create_model(forest, 
                 alphabet_size,
                 new_alphabet_size = 0,
                 tcmc_models=8,
                 sequence_length_as_feature=False,
                 dense1_dimension=16,
                 dense2_dimension=16,
                 name="clamsa_mean_log",
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
    tcmc_layer = TCMCProbability((tcmc_models,), forest, name="P_sequence_columns")
    mean_log_layer = SequenceLogLikelihood(name='mean_log_P', dtype=tf.float64)
 
    sl_concat_layer = None
    dense1_layer = None
    dense2_layer = None

    if sequence_length_as_feature:
        sl_concat_layer = tf.keras.layers.Concatenate(name="mean_log_P_and_log_sequence_length")

    if dense1_dimension > 0:
        dense1_layer = tf.keras.layers.Dense(dense1_dimension, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="dense1", dtype=tf.float64)

    if dense2_dimension > 0:
        dense2_layer = tf.keras.layers.Dense(dense2_dimension, kernel_initializer = "TruncatedNormal", activation = "sigmoid", name="dense2", dtype=tf.float64)

    guesses_layer = tf.keras.layers.Dense(num_classes, kernel_initializer = "TruncatedNormal", activation = "softmax", name = "guesses", dtype=tf.float64)

    # assemble the computational graph
    if new_alphabet_size > 0:
        sequences2 = encoding_layer(sequences)
    else:
        sequences2 = sequences
    P = tcmc_layer(sequences2, clade_ids)
    mean_log_P = mean_log_layer([P, sequence_lengths])
    X = mean_log_P

    if sequence_length_as_feature:
        log_sl_layer = tf.keras.layers.Lambda(tf.math.log, name="log_sequence_length", dtype=tf.float64)
        log_sl = log_sl_layer(tf.cast(sequence_lengths, dtype = tf.float64))
        X = sl_concat_layer([X, log_sl])

    if dense1_layer != None:
        X = dense1_layer(X)

    if dense2_layer != None:
        X = dense2_layer(X)

    guesses = guesses_layer(X)

    model = tf.keras.Model(inputs = [sequences, clade_ids, sequence_lengths], outputs = guesses, name = name)

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

class SequenceLogLikelihood(tf.keras.layers.Layer):
    """From variable-length sequence to single number: mean log likelihood"""
    def __init__(self, **kwargs):
        super(SequenceLogLikelihood, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SequenceLogLikelihood, self).build(input_shape)

    @tf.function
    def call(self, inputs, training = None):
        P = inputs[0]
        sl = inputs[1]
        sls = tf.reshape(sl, shape = [-1]) # hence use [None,1] and reshape it

        seq_ids = segment_ids(sls)
        log_P = tf.math.log(P)
        loglikelihood = - tf.math.segment_mean(log_P, seq_ids) # lengths tf.dtypes.cast(sl, tf.float64)

 
        return loglikelihood

    def get_config(self):
        base_config = super(SequenceLogLikelihood, self).get_config()
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
