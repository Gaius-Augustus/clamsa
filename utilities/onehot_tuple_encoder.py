import json
import itertools
import numpy as np

class OnehotTupleEncoderSingleton(object):
    def __init__(self):
        self._encoders = {}
        self._decoders = {}

    def _encoder(self, alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet=True):
        
        # create a dict. with the parameters
        encoder_config = {
            "alphabet": alphabet,
            "tuple_length": tuple_length,
            "bucket_alphabet": bucket_alphabet,
            "use_bucket_alphabet": use_bucket_alphabet
        }
        
        # dump into a string to make hashable
        encoder_key = json.dumps(encoder_config)
        
        # if the encocer has already been calculated we can retrieve it
        if encoder_key in self._encoders:
            return self._encoders[encoder_key]
        
        
        # calculate the words of length `tuple_length` in the given alphabet
        # these will automatically be lexikographically sorted
        tuples = itertools.product(*[alphabet for i in range(tuple_length)])
        tuples = [''.join(c) for c in tuples]
        C = {c:i for (i,c) in enumerate(tuples)}

        # calculate the 'buckets', i.e. all characters like 'n' in the fasta notation
        # which could be any of the characters 'acgt'
        all_tuples = tuples
        B = {c: c for c in alphabet}

        if use_bucket_alphabet:
            bucket_keys = ''.join(bucket_alphabet.keys())
            all_tuples = itertools.product(*[alphabet+bucket_keys for i in range(tuple_length)])
            all_tuples = [''.join(c) for c in all_tuples]
            B.update(bucket_alphabet)


        # use these buckets to calculate the dict `E`.
        # the keys are elements of `all_tuples` and their values are
        # their onehot encoding
        E = {}

        for gen_tuple in all_tuples:

            possible_tuples = itertools.product(*[B[c] for c in gen_tuple])
            possible_tuples = [''.join(c) for c in possible_tuples]

            encoding = np.zeros(len(tuples), dtype=np.int32)
            encoding[[C[c] for c in possible_tuples]] = 1
            E[gen_tuple] = encoding
            
        # save the encoder for later reuse
        self._encoders[encoder_key] = E
            
        return E


    def _decoder(self, alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet=True):
        
        # create a dict. with the parameters
        decoder_config = {
            "alphabet": alphabet,
            "tuple_length": tuple_length,
            "bucket_alphabet": bucket_alphabet,
            "use_bucket_alphabet": use_bucket_alphabet
        }
        
        # dump into a string to make hashable
        decoder_key = json.dumps(decoder_config)

        if decoder_key in self._decoders:
            return self._decoders[decoder_key]

        E = self._encoder(alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet)

        D = {tuple(v): k for k,v in E.items()}

        self._decoders[decoder_key] = D

        return D

    
    def encode(self,
               sequences, 
               alphabet = "acgt",
               tuple_length = 3,
               tuples_overlap = False, 
               bucket_alphabet = {
                    'r': 'ag',
                    'y': 'ct',
                    'k': 'gt',
                    'm': 'ac',
                    's': 'cg',
                    'w': 'at',
                    'b': 'cgt',
                    'd': 'agt',
                    'h': 'act',
                    'v': 'acg',
                    'n': 'acgt'
                }, 
               use_bucket_alphabet = True):
        
        E = self._encoder(alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet)
        coder = lambda c: E.get(c, np.ones(len(alphabet) ** tuple_length))
    
        coded_sequences = np.array([[coder(s[i:i+tuple_length]) for i in range(0,len(s)-tuple_length+1)] for s in sequences]) if tuples_overlap \
            else np.array([[coder(s[i:i+tuple_length]) for i in range(0,len(s), tuple_length)] for s in sequences])

        return coded_sequences

    def decode(self,
            coded_sequences,
            alphabet = 'acgt',
            tuple_length = 3,
            tuples_overlap = False,
            bucket_alphabet = {
                    'r': 'ag',
                    'y': 'ct',
                    'k': 'gt',
                    'm': 'ac',
                    's': 'cg',
                    'w': 'at',
                    'b': 'cgt',
                    'd': 'agt',
                    'h': 'act',
                    'v': 'acg',
                    'n': 'acgt'
                }, 
                use_bucket_alphabet = True,
                missing_entry_character = '-'):

        D = self._decoder(alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet)

        D[(1,)*(len(alphabet)**tuple_length)] = missing_entry_character * tuple_length

        num_sequences = coded_sequences.shape[0]
        coded_lengths = coded_sequences.shape[1]

        decoded_sequences = [''.join([D[tuple(coded_sequences[s,c,:])] for c in range(0, coded_lengths, tuple_length)]) for s in range(num_sequences)] if tuples_overlap \
            else [''.join([D[tuple(coded_sequences[s,c,:])] for c in range(coded_lengths)]) for s in range(num_sequences)]

        return decoded_sequences
        

    def decode_tfrecord_entry(
            self,
            coded_sequences,
            alphabet = 'acgt',
            tuple_length = 3,
            tuples_overlap = False,
            bucket_alphabet = {
                    'r': 'ag',
                    'y': 'ct',
                    'k': 'gt',
                    'm': 'ac',
                    's': 'cg',
                    'w': 'at',
                    'b': 'cgt',
                    'd': 'agt',
                    'h': 'act',
                    'v': 'acg',
                    'n': 'acgt'
                }, 
                use_bucket_alphabet = True,
                missing_entry_character = '-'):

        D = self._decoder(alphabet, tuple_length, bucket_alphabet, use_bucket_alphabet)
        D[(1,)*(len(alphabet)**tuple_length)] = missing_entry_character * tuple_length

        num_sequences = coded_sequences.shape[0]
        coded_lengths = coded_sequences.shape[1]

        decoded_sequences = np.array([[D[tuple(coded_sequences[s,c,:])] for c in range(0, coded_lengths, tuple_length)] for s in range(num_sequences)]) if tuples_overlap \
            else np.array([[D[tuple(coded_sequences[s,c,:])] for c in range(coded_lengths)] for s in range(num_sequences)])

        return decoded_sequences


# Singleton Object
# In this way we can reuse the calculated encoders and do not
# need to recalculate them in each new `encode` call
OnehotTupleEncoder = OnehotTupleEncoderSingleton()
