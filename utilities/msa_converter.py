#!/usr/bin/env python3
import gzip
from operator import is_
import regex as re
import random
import numpy as np
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
import sys
import math
import numbers
from contextlib import ExitStack
import time
import os
from tqdm import tqdm
import newick
from collections import Counter
import itertools
import json
import pathlib
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from . import onehot_tuple_encoder as ote

stop_codons = {"taa", "tag", "tga"}

class MSA(object):
    def __init__(self, label = None, chromosome_id = None, start_index = None, end_index = None,
                 is_on_plus_strand = False, frame = 0, spec_ids = [], offsets = [], sequences = [], use_amino_acids = False, 
                 tuple_length = 1, tuples_overlap = False, fname = None, use_codons = False, is_codon_aligned = False,
                 removeFinalStopColumn = True):
        self.label = label # label, class, e.g. y=1 for coding, y=0 for non-coding
        self.chromosome_id = chromosome_id
        self.start_index = start_index # chromosomal position
        self.end_index = end_index
        self.is_on_plus_strand = is_on_plus_strand
        self.frame = frame
        self.spec_ids = spec_ids
        self.offsets = offsets
        self.sequences = sequences
        self._updated_sequences = True
        self._coded_sequences = None
        self.in_frame_stops = []
        self.use_amino_acids = use_amino_acids
        self.tuple_length = tuple_length
        self.tuples_overlap = tuples_overlap
        self.fname = fname
        self.use_codons = use_codons
        self.is_codon_aligned = is_codon_aligned
        self.removeFinalStopColumn = removeFinalStopColumn

    @property
    def coded_sequences(self, alphabet = "acgt"):
        if self.use_amino_acids:
            alphabet = "ARNDCEQGHILKMFPSTWYV"
            
        # lazy loading
        if not self._updated_sequences:
            return self._coded_sequences

        # whether the sequences and coding alphabet shall be flipped
        inv_coef = -1 if not self.is_on_plus_strand else 1

        # translated alphabet as indices of (inversed) alphabet
        #translated_alphabet = dict(zip(alphabet, range(len(alphabet))[::inv_coef] ))

        # view the sequences as a numpy array
        ca = [S[::inv_coef] for S in self.sequences]

        # translate the list of sequences and convert it to a numpy matrix
        # non-alphabet characters are replaced by -1
        tuple_length = self.tuple_length if self.tuples_overlap else 1
        self._coded_sequences = ote.OnehotTupleEncoder.encode(ca, alphabet = alphabet, tuple_length = tuple_length, tuples_overlap = self.tuples_overlap, 
                                                              use_bucket_alphabet=False)

        # update lazy loading
        self._updated_sequences = False

        return self._coded_sequences

    @property
    def codon_aligned_sequences(self, alphabet="acgt", gap_symbols='-'):
        if self.use_amino_acids:
            alphabet = "ARNDCEQGHILKMFPSTWYV"
        # size of a codon in characters
        c = 3 if self.use_codons else self.tuple_length

        # the list of sequences that should be codon aligned
        sequences = self.sequences

        # reverse complement if not on plus strand
        if not self.is_on_plus_strand:
            rev_alphabet = alphabet[::-1]
            tbl = str.maketrans(alphabet, rev_alphabet)
            sequences = [s[::-1].translate(tbl) for s in sequences]

        if not self.is_codon_aligned:
            cali, self.in_frame_stops = tuple_alignment(sequences, gap_symbols, frame = self.frame, tuple_length = c, removeFinalStopColumn = self.removeFinalStopColumn)
        else:
            cali = sequences
        return cali

    @property
    def coded_codon_aligned_sequences(self, alphabet="acgt", gap_symbols = '-'):
        if self.use_amino_acids:
            alphabet = "ARNDCEQGHILKMFPSTWYV"
        ca = self.codon_aligned_sequences
        c = 3 if self.use_codons else self.tuple_length
        return ote.OnehotTupleEncoder.encode(ca, alphabet = alphabet, tuple_length = c, use_bucket_alphabet = False)

    @property
    def sequences(self):
        return self._sequences

    def alilen(self):
        # TODO: is somewhere checked that all rows have the same length?
        if (self._coded_sequences is not None):
            length = len(self._coded_sequences[0])
        else:
            length = len(self._sequences[0])
            if self.use_codons:
                length = int(length / 3) # may differ from the number of codon columns
            else:
                length = int(length / self.tuple_length) 
        return length

    def alidepth(self):
        return len(self._sequences)
    
    def get_omegas(self, seperator):
        small_omegas = []
        large_omegas = []
        for omega in self.label:
            if omega > seperator:
                large_omegas.append(omega)
            else:
                small_omegas.append(omega)
        return small_omegas, large_omegas
        
    def delete_rows(self, which):
        assert len(which) == len(self.sequences), "Row number mismatch. Alignment {} is expected to have {} rows".format(self, len(which))
        for i in reversed(range(len(which))):
            if which[i]:
                del self.sequences[i]
                del self.spec_ids[i]
                if self.offsets:
                    del self.offsets[i]
        self._updated_sequences = True

    @sequences.setter
    def sequences(self, value):
        self._sequences = value
        self._updated_sequences = True

    def __str__(self):
        return f"{{\n\tlabell: {self.label},\n\tchromosome_id: {self.chromosome_id},\n\tstart_index: {self.start_index},\n\tend_index: {self.end_index},\n\tis_on_plus_strand: {self.is_on_plus_strand},\n\tframe: {self.frame},\n\tspec_ids: {self.spec_ids},\n\toffsets: {self.offsets},\n\tsequences: {self.sequences},\n\tcoded_sequences: {self.coded_sequences},\n\tcodon_aligned_sequences: {self.codon_aligned_sequences}\n}}"




def tuple_alignment(sequences, gap_symbols='-', frame = 0, tuple_length = 3,
                    removeFinalStopColumn = True):
    """
    Align a list of string sequences to tuples of a fixed length with respect to a set of gap symbols.

    Args:
        sequences (List[str]) The list of sequences
        gap_symbols (str): A string containing all symbols to be treated as gap-symbols
        frame (int): Ignore the first `(tuple_length - frame) % tuple_length` symbols found
        tuple_length (int): Length of the tuples to be gathered

    Returns:
        List[str]: The list of tuple strings of the wanted length. The first gap-symbols `gap_symbols[0]` is used to align these (see the example).
    Example:
        * Two codons are considered aligned, when all 3 of their bases are aligned with each other.
        * Note that not all bases of an ExonCandidate need be aligned.
        * example input (all ECs agree in phase at both boundaries)
        *        
        *                 a|c - - t t|g a t|g t c|g a t|a a 
        *                 a|c - - c t|a a - - - c|a n c|a g
        *                 g|c g - t|t g a|- g t c|g a c|a a
        *                 a|c g t|t t g|a t - t|c g a|c - a
        *                 a|c g - t|t g a|t g t|t g a|- a a
        *                   ^                       ^
        * firstCodonBase    |                       | lastCodonBase (for last species)
        * example output: (stop codons are excluded for single and terminal exons)
        *                 - - -|c t t|- - -|g t c|- - -|g a t
        *                 - - -|c c t|- - -|- - -|- - -|a n c
        *                 c g t|- - -|t g a|g t c|- - -|g a c
        *                 - - -|- - -|- - -|- - -|c g a|- - -
        *                 c g t|- - -|t g a|- - -|t g a|- - -
        *
        Reproduce via 
               S = ['ac--ttgatgtcgataa',
                    'ac--ctaa---cancag',
                    'acg-ttga-gtcgacaa',
                    'acgtttgat-tcgac-a',
                    'acg-ttgatgttga-aa']
               print(tuple_alignment(S, frame=2))
    """
    # shorten notation
    S = sequences

    # number of entries missing until the completion of the framing tuple
    frame_comp = frame # this is the GGF frame definition, was before: frame_comp = (tuple_length - frame) % tuple_length

    # pattern to support frames, i.e. skipping the first `frame_comp` tuple entries at line start
    frame_pattern = '(?:(?:^' + f'[^{gap_symbols}]'.join([f'[{gap_symbols}]*' for i in range(frame_comp+1)]) + f')|[{gap_symbols}]*)\K'

    # generate pattern that recognizes tuples of the given length and that ignores gap symbols
    tuple_pattern = f'[{gap_symbols}]*'.join([f'([^{gap_symbols}])' for i in range(tuple_length)])
    tuple_re = re.compile(frame_pattern + tuple_pattern)

    # for each sequence find the tuples of indices 
    T = [set(tuple(m.span(i+1)[0] for i in range(tuple_length)) for m in tuple_re.finditer(s)) for s in S]

    # flatten T to a list and count how often each multiindex is encountered
    occ = Counter( list(itertools.chain.from_iterable([list(t) for t in T])) )

    # find those multiindices that are in more than one sequence and sort them lexicographically
    I = sorted([i for i in occ if occ[i] > 1])

    # trivial case: there is nothing to align
    if len(I) == 0:
        return ['' for s in S], False

    # calculate a matrix with `len(S)` rows and `len(I)` columns.
    #   if the j-th multindex is present in sequence `i` the entry `(i,j)` of the matrix will be 
    #   a substring of length `tuple_length` corresponding to the characters at the positions 
    #   specified by the `j`-th multiindex
    #
    #   otherwise the prime gap_symbol (`gap_symbol[0]`) will be used as a filler
    missing_entry = gap_symbols[0] * tuple_length
    entry_func = lambda i,j: ''.join([S[i][a] for a in I[j]]) if I[j] in T[i] else missing_entry
    ta_matrix = np.vectorize(entry_func)(np.arange(len(S))[:,None],np.arange(len(I)))

    # remove last column if it contains a stop codon (happens for single and terminal exons)
    if removeFinalStopColumn:
        stops_in_lastcol = set(ta_matrix[:,-1]) & stop_codons
        if stops_in_lastcol:
            ta_matrix = ta_matrix[:, 0:-1]

    # check which rows contain an in-frame stop codon elsewhere
    stops = []
    for row in ta_matrix:
        stops.append(bool(set(row) & stop_codons))

    # join the rows to get a list of tuple alignment sequences
    ta = [''.join(row) for row in ta_matrix]
    return ta, stops



def leaf_order(path, use_alternatives=False):
    """
        Find the leaf names in a Newick file and return them in the order specified by the file
        
        Args:
            path (str): Path to the Newick file
            use_alternatives (bool): TODO
            
        Returns:
            List[str]: Leaf names encountered in the file
    """
    
    # regex that finds leave nodes in a newick string
    # these are precisely those nodes which do not have children
    # i.e. to the left of the node is either a '(' or ',' character
    # or the beginning of the line
    leave_regex = re.compile('(?:^|[(,])([\w.-]*)[:](?:(?:[0-9]*[.])?[0-9]+)')
    
    with open(path, 'r') as fp:
        
        nwk_string = fp.readline()
        
        matches = leave_regex.findall(nwk_string)
        
        alt_path = path + ".alt"
        
        if use_alternatives and os.path.isfile(alt_path):
            with open(alt_path) as alt_file:
                alt = json.load(alt_file)
                
                # check for valid input 
                assert isinstance(alt, dict), f"Alternatives in {alt_path} is no dictionary!"
                for i in alt:
                    assert isinstance(alt[i], list), f"Alternative for {i} in {alt_path} is not a list!"
                    for entry in alt[i]:
                        assert isinstance(entry, str), f"Alternative {alt[i][j]} for {i} in {alt_path} is not a string!"
                
                matches = [set([matches[i]] + alt[matches[i]]) for i in range(len(matches))]

        return matches
    
def import_fasta_training_file(paths, undersample_neg_by_factor = 1., 
                               reference_clades = None, margin_width = 0, fixed_sequence_length = None,
                               tuple_length = 1, tuples_overlap = False, use_amino_acids = False, 
                               use_codons = False, sitewise = False):
    """ Imports the training files in fasta format.
    Args:
        paths (List[str]): Location of the file(s) 
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor,
                                           set to 1.0 to use all negative examples
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
        fixed_sequence_length (int): Sequences with a different length will be discarded, calculated after margin_width was removed from sequence.
        tuple_length (int): Length of an entry of the alphabet. e.g. 3 if you use codons or 1 if you use nucleotides as alphabet
        tuples_overlap (bool): True if you want the tuples to overlap tuple_length-1 characters, else they are consecutive
        use_amino_acids (bool): True if you want to use amino acids instead of nucleotides as alphabet.
        use_codons (bool): msas will be interpreted as a codon alignment

    Example for input fasta file:
        *
        * >species_name_1|...|1
        * acaatcggt
        * >species_name_2
        * acaat---t
        *
        * The last entry in the header determine the label of the alignment. Here: label = 1.
        * For sitewise training data the last entry is the list of labels for the codon columns.

    Returns:
        List[MSA]: Training examples read from the file(s).
        List[List[str]]: Unique species configurations either encountered or given by reference.

    """

    def species_name(fasta_hdr):
        """
         extract the species name from the ">" line in fasta, e.g.
        Homo_sapiens.chr17:63832|0,0,1,1,0 -> Homo_sapiens
        Bos_taurus -> Bos_taurus
        """
        mafpatstr = r"(\w+)(?:\..*)?\|?"
        m = re.match(mafpatstr, fasta_hdr)
        if not m:
           print ("FASTA header format not recognized. Perhaps special characters in",
                  fasta_hdr)
        return m.group(1)

    training_data = []

    # If clades are specified, the leaf species will be imported.
    species = [leaf_order(c) for c in reference_clades] if reference_clades != None else []

    # total number of bytes to be read
    total_bytes = sum([os.path.getsize(x) for x in paths])

    # Status bar for the reading process
    pbar = tqdm(total = total_bytes, desc = "Parsing FASTA file(s)", unit = 'b', unit_scale = True)

    for path in paths:
        fasta = gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')
        bytes_read = fasta.tell() # byte reading progress (here 0)
        # split a FASTA file with potentially multiple MSAs at the empty lines
        while True:
            if not fasta:
                break
            nextMSAsection = ""
            while True:
                line = fasta.readline()
                if line == "\n" or not line:
                    break
                nextMSAsection += line
            if nextMSAsection == "":
                break
            #print ("nextMSAsection=\n", nextMSAsection)
            msa_io = io.StringIO(nextMSAsection)

            #entries = [rec for rec in SeqIO.parse(fasta, "fasta")]
            entries = [rec for rec in SeqIO.parse(msa_io, "fasta")]
            # parse the species names
            spec_in_file = [species_name(e.id) for e in entries]
            # parse the label
            if sitewise:
                # assume that the first record is from the references and contains the labels
                label = entries[0].id.split('|')[-1]
                label = list(map(float, label.split(",")))
            else:
                try:
                    label = int(entries[0].id.split('|')[-1])
                except ValueError:
                    print ("Wrong format in FASTA header. Cannot extract integer class label from ",{entries[0].id}, file=sys.stderr)
            # compare them with the given references
            ref_ids = [[(r,i) for r in range(len(species)) for i in range(len(species[r])) if s in species[r][i]] for s in spec_in_file]

            # check if these are contained in exactly one reference clade
            n_refs = [len(x) for x in ref_ids]

            if 0 == min(n_refs) or max(n_refs) > 1:
                continue

            ref_ids = [x[0] for x in ref_ids]

            if len(set(r for (r,i) in ref_ids)) > 1:
                continue

            # read the sequences and trim them if wanted
            if not use_amino_acids:
                sequences = [str(rec.seq).lower() for rec in entries]
            else:
                sequences = [str(rec.seq) for rec in entries]
            sequences = sequences[margin_width:-margin_width] if margin_width > 0 else sequences

            # decide whether the upcoming entry should be skipped
            skip_entry = (model == 0 and random.random() > 1. / undersample_neg_by_factor) or (fixed_sequence_length and fixed_sequence_length != len(sequences[0]))
            if skip_entry:
                fasta.close()
                continue

            msa = MSA(
                label = label,
                chromosome_id = None, 
                start_index = None,
                end_index = None,
                is_on_plus_strand = True,
                frame = 0,
                spec_ids = ref_ids,
                offsets = [],
                sequences = sequences,
                use_amino_acids = use_amino_acids,
                tuple_length = tuple_length,
                tuples_overlap = tuples_overlap,
                use_codons = use_codons
            )        
            training_data.append(msa)

            pbar.update(fasta.tell() - bytes_read)
            bytes_read = fasta.tell()
        fasta.close()

    return training_data, species

def import_augustus_training_file(paths, undersample_neg_by_factor = 1., alphabet=['a', 'c', 'g', 't'], reference_clades = None, 
                                  margin_width = 0, fixed_sequence_length = None, tuple_length = 1, tuples_overlap = False, use_codons = False):
    """ Imports the training files generated by augustus. This method is tied to the specific
     implementation of GeneMSA::getAllOEMsas and GeneMSA::getMsa.
     
    Args:
        paths (List[str]): Location of the file(s) generated by augustus (typically denoted *.out or *.out.gz)
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor,
                                           set to 1.0 to use all negative examples
        alphabet (List[str]): Alphabet in which the sequences are written
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
        fixed_sequence_length (int): Sequences with a different length will be discarded, calculated after margin_width was removed from sequence.
        tuple_length (int): Length of an entry of the alphabet. e.g. 3 if you use codons or 1 if you use nucleotides as alphabet
        tuples_overlap (bool): True if you want the tuples to overlap tuple_length-1 characters, else they are consecutive
        use_codons (bool): msas will be interpreted as a codon alignment

    Returns:
        List[MSA]: Training examples read from the file(s).
        int: Number of unique species encountered in the file(s).
    
    """
    
    
    # entries already read
    training_data = []
    
    # counts number of unique species encountered the file
    num_species = 0

    # The species encountered in the file(s).
    # If clades are specified the leave species will be imported.
    species = [leaf_order(c) for c in reference_clades] if reference_clades != None else []
    print ("species=", species, "reference_clades=", reference_clades)
    # total number of bytes to be read
    total_bytes = sum([os.path.getsize(x) for x in paths])

    # Status bar for the reading process
    pbar = tqdm(total = total_bytes, desc = "Parsing AUGUSTUS file(s)", unit = 'b', unit_scale = True)

    for p in range(len(paths)):
        path = paths[p]

        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')) as f:

            # species encountered in the header of the file
            encountered_species = {}

            # if a reference clade is specified we need to translate the indices
            # else just use the order specified in the file
            spec_reorder = {}
            
            # Regex Pattern recognizing lines generated by GeneMSA::getMsa
            # old format "^[0-9]+\\t[0-9]+\\t\\t[acgt\-]+"
            slice_pattern = re.compile("^[0-9]+\\t[acgt\-]+$")
            
            # whether the current sequence should be skipped due to 
            # a undersample roll
            skip_entry = False

            line = f.readline()
            bytes_read = f.tell()

            # if the header is completely read, i.e. all species are read
            header_read = False
            
            while line:
                # parse the lines generated by GeneMSA::getAllOEMsas
                if line.startswith("y="):
                    if not header_read:
                        if reference_clades != None:
                            e = set(encountered_species.values())
                            C = [set(c) for c in species]
                            esubC = [e <= c for c in C]
                            num_parents = sum(esubC)

                            if num_parents == 0:
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}'
                                                f'" are not fully included in any of the given clades:' + str(C))
                            if num_parents > 1:
                                parent_clades = [reference_clades[i] for i in range(len(C)) if e <= C[i]]
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}" are included in multiple clades, namely in {parent_clades}.')
                            # index of the clade to use
                            i = esubC.index(True)
                            ref = species[i]
                            spec_reorder = {j:(i,ref.index(encountered_species[j])) for j in encountered_species}
                        else:
                            spec_reorder = {j:(p,j) for j in encountered_species}
                            species.append(list(encountered_species.values()))

                        header_read = True
                    
                    # decide whether the upcoming entry should be skipped
                    skip_entry = line[2]=='0' and random.random() > 1. / undersample_neg_by_factor
                    
                    if skip_entry:
                        continue
                    
                    oe_data = line.split("\t")

                    try:
                        msa = MSA(
                            label = int(oe_data[0][2]),
                            chromosome_id = oe_data[2], 
                            start_index = int(oe_data[3]),
                            end_index = int(oe_data[4]),
                            is_on_plus_strand = (oe_data[5] == '+'),
                            frame = int(oe_data[6][0]),
                            spec_ids = [],
                            offsets = [],
                            sequences = [],
                            tuple_length = tuple_length, 
                            tuples_overlap = tuples_overlap, 
                            use_codons = use_codons
                        )
                    except ValueError:
                        sys.exit("Parsing error in line\n" + line + "\nsplit into:" + str(oe_data))
                        
                    training_data.append(msa)
                
                # parse the lines generated by GeneMSA::getMsa
                elif slice_pattern.match(line) and not skip_entry:
                    slice_data = line.split("\t")
                    entry = training_data[-1]
                    sidx = int(slice_data[0])
                    if sidx not in spec_reorder:
                        sys.exit(f"Error: species index {sidx} out of bounds in line\n{line}")
                    padded_sequence = slice_data[1][:-1]
                    sequence = padded_sequence[margin_width:-margin_width] if margin_width > 0 else padded_sequence
                    if fixed_sequence_length and fixed_sequence_length != len(sequence):  
                        # if sequence is too short, skip entry and delete msa from training_data
                        del training_data[-1]
                        skip_entry = True
                        continue
                    entry.spec_ids.append(spec_reorder[sidx])
                    # entry.offsets.append(int(slice_data[1]))
                    entry.sequences.append(sequence)

                    
                # retrieve the number of species
                elif line.startswith("species ") and not skip_entry:
                    spec_line = line[len('species '):].split('\t')
                    specid = int(spec_line[0])
                    spec_name = spec_line[1].strip()
                    encountered_species[specid] = spec_name

                line = f.readline()
                pbar.update(f.tell() - bytes_read)
                bytes_read = f.tell()

    return training_data, species


def get_fasta_seqs(fasta_path : str, use_amino_acids, margin_width = 0):
    """ Returns the sequences in a fasta file.
    @param fasta_path: Path to the fasta file with one MSA
    @return List[str]: Sequences (= rows of MSA) in the fasta file
            List[str]: Species names for each row
    """
    with gzip.open(fasta_path, 'rt') if fasta_path.endswith('.gz') else open(fasta_path, 'r') as fasta_file:
        entries = [rec for rec in SeqIO.parse(fasta_file, "fasta")]

    # extract the sequences and trim them if wanted
    sequences = []
    for rec in entries:
        rowstr = str(rec.seq)
        if not use_amino_acids:
            rowstr = rowstr.lower() # to a,c,g,t alphabet
        if margin_width > 0:
            rowstr = rowstr[margin_width:-margin_width]
        sequences.append(rowstr)

    # parse the species names
    spec_in_file = [e.id.split('|')[0] for e in entries]

    # the first entry of the fasta file has the header informations
    header_fields = entries[0].id.split("|")
    # allow noninformative fasta headers as well
    frame = 0
    if len(header_fields) > 2:
        try:
            frame =  int(header_fields[2][-1])
        except ValueError:
            pass # leave frame at 0 by default
    plus_strand = True if len(header_fields) < 5 or header_fields[4] != 'revcomp' else False

    return [{"seqs" : sequences, "species" : spec_in_file,
             "frame" : frame, "plus_strand" : plus_strand,
             "fasta_path" : fasta_path}]


def get_Bio_seqs(msa : MultipleSeqAlignment):
    """ Returns codon alignments for all frames uninterrupted in the reference
    at least 6 codon alignments (3 frames x 2 strands), provided the input MSA is long enough 
    @param msa: one MSA object parsed by Biopython, e.g. from a MAF file
    @return List[str]: Sequences (= rows of MSA) in the fasta file
            List[str]: Species names for each row
    """
    
    refseqrec = msa[0] # the first sequence is the reference
    dot = refseqrec.id.find('.')
    if dot < 0:
        raise AssertionError("MAF record does not contain a dot - to separate species and seq name")

    refChr = refseqrec.id[dot+1:]
    refchrStart = refseqrec.annotations["start"] + 1 # 0-based in MAF, here 1-based for GFF comp.
    refchrLen = refseqrec.annotations["size"]
    refrow = str(refseqrec.seq)
    rownchars = len(refrow.replace('-', ''))
    if rownchars < refchrLen:
        print (f"MAF format error: character number in alignment row ({rownchars}) smaller than annotated ({refchrLen})")
        sys.exit(1)

    for seqrec in msa: # remove the .chr from the species name
        seqrec.id = seqrec.id.split('.', 1)[0]
        seqrec.seq = seqrec.seq.lower() # to a,c,g,t alphabet

    alilen = len(refrow)
    fragMSAs = []
 
    """
    distinguish 3 types of coordinates:
    chrPos in reference genome, alipos in MSA, i in reference row (no gaps)
    chrPos = refchrStart + i
    0  1  2  3  4  5  6  7  8  9 10 11 12 13  alipos - position in MSA coordinates
         |        |             |             codon boundaries - here frame=1
    -  C  A  T  G  -  -  T  G  G  A  T  G     reference row refrow (str)
    0  1  2  3  4        5  6  7  8  9 10     i
    """
    def right_move(alipos, by=3):
        """ right-move 'by' positions from alipos, return the new position as well as the number of gaps encountered. alipos is assumed to point to a non-gap and will point to a non-gap after.
        """
        gapsWithin = gapsAfter = 0
        while by > 0 and alipos < alilen:
            alipos += 1
            by -= 1
            while alipos < alilen and refrow[alipos] == '-':
                alipos += 1
                if by > 0:
                    gapsWithin += 1
                else:
                    gapsAfter += 1

        return alipos, gapsWithin, gapsAfter

    def appendMSA(outMSA, frame, strand, chrAliStart):
        """ append the MSA to the list of fragment MSAs"""
        if outMSA is not None and outMSA.get_alignment_length() > 0:
            outMSA.annotations["local_frame"] = frame # frame in the fragment, currently not used
            outMSA.annotations["strand"] = strand
            outMSA.annotations["startPos"] = chrAliStart
            fragMSAs.append(outMSA) #(sequences, spec_in_file, frame, plus_strand)

    for strand in (1, -1):
        for frame in range(3):
            # print (f"frame={frame} strand={strand}")
            alipos = 0
            # read over gaps at the beginning of the reference row
            while refrow[alipos] == '-' and alipos < alilen:
                alipos += 1
            alipos, gapsWithin, gapsAfter = right_move(0, frame)
            i = frame
            outMSA = None
            chrAliStart = refchrStart + i
            outalilen = 0
            while alipos < alilen:
                nextalipos, gapsWithin, gapsAfter = right_move(alipos)
                # throw away codons that are interrupted by gaps
                if gapsWithin == 0 and nextalipos >= alipos + 3:
                    # cut out the codon from msa
                    msa1codon = msa[:,alipos:alipos+3]
                    if strand == -1:
                        " reverse complement the single codon alignment "
                        for seqrec in msa1codon:
                            seqrec.seq = seqrec.seq.reverse_complement()
                    if outMSA is None:
                        outMSA = msa1codon
                        chrAliStart = refchrStart + i
                    else:
                        outMSA += msa1codon
                    outalilen += 3
                    # print (f"codon column chrAliStart={chrAliStart} i={i} alipos={alipos}", msa1codon)
                else: # codon will not be scored, fragment ends
                    appendMSA(outMSA, frame, strand, chrAliStart)
                    outMSA = None

                alipos = nextalipos
                i += 3 # next codon in the same frame
            appendMSA(outMSA, frame, strand, chrAliStart)
    
    # construct return list
    msalst = []
    for j, msa in enumerate(fragMSAs):
        sequences = []
        spec_in_file = []
        for seqrec in msa:
            sequences.append(str(seqrec.seq))
            spec_in_file.append(seqrec.id)
        numSites = msa.get_alignment_length()
        if numSites % 3 != 0:
            print (f"Error: number of sites ({numSites}) not divisible by 3:", msa)
        numSites = int(numSites / 3) # codon columns
        plus_strand = (msa.annotations["strand"] == 1)
        chrPos = msa.annotations["startPos"]
        msalst.append({"seqs" : sequences, "species" : spec_in_file,
             "frame" : 0, # given to clamsa: start codon right at start of MSA
             "plus_strand" : plus_strand,
             "seqname" : refChr, "chrPos" : chrPos, "numSites" : numSites})
        # print (f"\nfragment {j}:", msa,
        #       f"\nf={msa.annotations['local_frame']} strand={msa.annotations['strand']}", chrPos, # len(sequences[0]))

    return msalst

def get_window_seqs(msa : MultipleSeqAlignment, sl: int):
    """ 
    Get MSAs of length sl that mimic a sliding window across the input MSA and its reverse complement
    @param msa: one MSA object parsed by Biopython, e.g. from a MAF file
            sl: length of window msa
    @return List[str]: Sequences (= rows of MSA) in the fasta file
            List[str]: Species names for each row
    """
    
    refseqrec = msa[0] # the first sequence is the reference
    dot = refseqrec.id.find('.')
    if dot < 0:
        raise AssertionError("MAF record does not contain a dot - to separate species and seq name")

    refChr = refseqrec.id[dot+1:]
    refchrLen = refseqrec.annotations["size"]
    if refchrLen < sl:
        return []
    #frame = refseqrec.annotations["frame"]
    refrow = str(refseqrec.seq)
    rownchars = len(refrow.replace('-', ''))
    if rownchars < refchrLen:
        print (f"MAF format error: character number in alignment row ({rownchars}) smaller than annotated ({refchrLen})")
        sys.exit(1)

    spec_in_file = []
    seqname = []
    strand = []
    chrStart = []
    for seqrec in msa: 
        spec_id = seqrec.id.split('.', 1)  
        spec_in_file.append(spec_id[0])  # species
        seqname.append(spec_id[1:][0])  # sequence  name
        strand.append(seqrec.annotations["strand"])  # strand
        chrStart.append(seqrec.annotations["start"]+1)  # start pos of sequence, +1 bc 0-based in MAF, here 1-based for GFF comp.
    
    orig_strand = [s == 1 for s in strand]  # plus strand or not
    reverse_strand = [not s for s in orig_strand]  # plus strand or not for reverse complement

    ebony_pos = int(sl/2) # pos of boundary in window
    alilen = len(refrow)  # alignment length
    msalst = []
    for i in range(alilen - sl + 1):
        window = msa[:,i:i+sl]
        chrPos = [(start+ebony_pos-1+i, start+ebony_pos+i) for start in chrStart] # tuple encases boundary
        for i in (1, -1):
            plus_strand = orig_strand
            if i == -1:
                plus_strand = reverse_strand
                for rec in window:
                    rec.seq = rec.seq.reverse_complement()

            msalst.append({"seqs" : [str(rec.seq) for rec in window], "species" : spec_in_file,
                "frame" : 0, # not used
                "plus_strand" : plus_strand,
                "seqname" : seqname, "chrPos" : chrPos})

    return msalst

# Part of this logic also resides inside `import_fasta` and `import_phylocsf`
#
# This function is currently just written for the particular fasta header format
# we generate for phylocsf.
def parse_text_MSA(text_MSA, clades, use_codons=True, margin_width=0, num_positions = None, 
                   trans_dict=None, remove_stop_rows=False, use_amino_acids = False,
                   tuple_length = 1, tuples_overlap = False, frame_align_codons = True,
                   sliding_window = False):
    """
       trans_dict   dictionary for translating names used in FASTA headers to taxon ids from the trees (clades)
    """
    tuple_length = 3 if use_codons else tuple_length

    trans_dict = {} if trans_dict is None else trans_dict
    if use_amino_acids:
        species = [leaf_order(c) for c in clades] if clades != None else []
    else:
        species = [leaf_order(c,use_alternatives=True) for c in clades] if clades != None else []

    if isinstance(text_MSA, str): 
        # extract a single MSA from a FASTA formatted file
        seq_msas = get_fasta_seqs(text_MSA, use_amino_acids, margin_width)
    elif isinstance(text_MSA, MultipleSeqAlignment):
         # extract multiple MSAs from a MultipleSeqAlignment object
        if sliding_window:
            sl = num_positions + tuple_length - 1
            seq_msas = get_window_seqs(text_MSA, sl)
        else:
            seq_msas = get_Bio_seqs(text_MSA)
    else:
        raise TypeError("parse_fasta_file: text_MSA must be either a string or a MultipleSeqAlignment")
    
    if not seq_msas: # empty seq list
        return [(-1, 0, None, {})]

    tensor_msas = [] # list of MSAs to be returned
    num_mismatch = 0 # number of MSAs that do not match the reference in length
    for seq_msa in seq_msas:
        sequences = seq_msa["seqs"]
        spec_in_file = seq_msa["species"]
        frame = seq_msa["frame"]
        plus_strand = seq_msa["plus_strand"]
        auxdata = None
        if "seqname" in seq_msa: # for BioPython MSAs from MAF
            auxdata = {"seqname" : seq_msa["seqname"],
                       "chrPos" : seq_msa["chrPos"],
                       "plus_strand" : seq_msa["plus_strand"]}
            if "numSites" in seq_msa: # sitewise prediction
                auxdata["numSites"] = seq_msa["numSites"]
            else: # sliding window prediction
                auxdata["species"] = spec_in_file 
        if "fasta_path" in seq_msa: # for FASTA files
            auxdata = {"fasta_path" : seq_msa["fasta_path"]}

        clade_id, sequence_length, S = None, None, None

        # translate species name from file to taxon ids
        translator = lambda s : trans_dict[s] if s in trans_dict else s
        msa_taxon_ids = list(map(translator, spec_in_file))
    
        # compare them with the given references
        ref_ids = [[(r,i) for r in range(len(species)) for i in range(len(species[r])) if s == species[r][i] ] for s in msa_taxon_ids]

        # check if these are contained in exactly one reference clade
        n_refs = [len(x) for x in ref_ids]

        if 0 == min(n_refs) or max(n_refs) > 1:
            continue

        ref_ids = [x[0] for x in ref_ids]
        if len(set(r for (r,i) in ref_ids)) > 1:
            # taxon is in multiple trees of the forest
            continue

        msa = MSA(
            label = None,
            chromosome_id = None, 
            start_index = None,
            end_index = None,
            is_on_plus_strand = plus_strand,
            frame = frame,
            spec_ids = ref_ids,
            offsets = [],
            sequences = sequences,
            use_amino_acids = use_amino_acids,
            tuple_length = tuple_length,
            tuples_overlap = tuples_overlap,
            use_codons = use_codons
        )
        # Use the correct onehot encoded sequences
        coded_sequences = msa.coded_codon_aligned_sequences if msa.use_codons or (msa.tuple_length > 1 and not msa.tuples_overlap) \
                                                            else msa.coded_sequences

        # remove all rows with an in-frame stop codon (except last col)
        stops = msa.in_frame_stops
        # print (msa, stops)
        remove_stop_rows = False
        if stops and remove_stop_rows :
            msa.delete_rows(stops)
            coded_sequences = coded_sequences[np.invert(stops)]
        # print ("after stop deletion:", msa, "\ncoded_sequences=", coded_sequences)

        if len(msa.sequences) < 2:
            continue
        
        sequence_length = len(coded_sequences[0])
        if sequence_length == 0:
            continue
        
        elif num_positions and num_positions != sequence_length:
            if (num_positions - sequence_length) % 2 != 0:
                # the sequence will be cut and padded symmetrically on both sides to get num_positions 
                # this cant happen if num_positions - sequence_length is an odd number
                continue
            if num_positions > sequence_length:
                # if sequence is too short, pad with 1 on both sides to get required length
                pad_width = int((num_positions - sequence_length)/2)  # width to pad on each side
                coded_sequences = np.pad(coded_sequences, ((0,0), (pad_width,pad_width), (0,0)), constant_values=1)
            else:
                # if sequence is too long, remove margin from sequence to get required length 
                marg_width = int((sequence_length - num_positions)/2)  # width to remove on each side
                coded_sequences = coded_sequences[:, marg_width:-marg_width, :]
            sequence_length = coded_sequences.shape[1]


        #print ("codon MSA")
        #for cs in msa.codon_aligned_sequences:
        #    print (cs)
        if "numSites" in auxdata and sequence_length != auxdata["numSites"]:
            """ This can happen as the codon construction in tuple_alignment is complicated.
            Either MSA can be longer than the other and this would mess up the rest of the batch. 
            TODO: write a function tuple_alignment_ref with as many columns as the reference. 
            """
            num_mismatch += 1
            continue

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
        tensor_msas.append((clade_id, sequence_length, S, auxdata))

    if num_mismatch > 0 and len(tensor_msas) > 0:
        print (f"Warning: {num_mismatch} ({100*num_mismatch/(num_mismatch+len(tensor_msas)):.2f}%) MSAs did not match the reference in length")
    
    return tensor_msas




def import_phylocsf_training_file(paths, undersample_neg_by_factor = 1., reference_clades = None, margin_width = 0, fixed_sequence_length = None, 
                                  use_codons = False):
    """ Imports archives of training files generated by PhyloCSF in zip format
     
    Args:
        paths (List[str]): Location of the file(s) generated by PhyloCSF
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor, set to 1.0 to use all negative examples
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
        fixed_sequence_length (int): Sequences with a different length will be discarded, calculated after margin_width was removed from sequence.
        use_codons (bool): msas will be interpreted as a codon alignment

    Returns:
        List[MSA]: Training examples read from the file(s).
        List[List[str]]: Unique species configurations either encountered or given by reference.
    
    """

    # TODO: support either the non-clades case or make reference_clades a required parameter
    species = [leaf_order(c,use_alternatives=True) for c in reference_clades] if reference_clades != None else []

    training_data = []

    with ExitStack() as stack:

        #file_indices = [(split_name, model, ) for ]
        phylo_files = [zipfile.ZipFile(path, 'r') for path in paths]
        # read the filenames of all fasta files inside these archives
        fastas = [[p.getinfo(n) for n in p.namelist() if (n.endswith('.mfa') or n.endswith('.fa'))] for p in phylo_files]

        # total number of bytes to be read
        total_bytes = sum([sum([f.compress_size for f in fastas[i]]) for i in range(len(phylo_files))])

        # Status bar for the reading process
        pbar = tqdm(total=total_bytes, desc="Parsing PhyloCSF file(s)", unit='b', unit_scale=True)

        for i, phylo_file in enumerate(phylo_files):
            for j, fasta in enumerate(fastas[i]):
                label = 0 if 'control' in fasta.filename else 1
                
                # decide whether the upcoming entry should be skipped
                skip_entry = label==0 and random.random() > 1. / undersample_neg_by_factor

                if skip_entry:
                    continue
                
                with io.TextIOWrapper(phylo_file.open(fasta), encoding="utf-8") as fafile:

                    entries = list(SeqIO.parse(fafile, "fasta"))
                    # parse the species names
                    spec_in_file = [e.id.split('|')[0] for e in entries]
                    
                    # compare them with the given references
                    ref_ids = [[(r,i) for r in range(len(species))  for i in range(len(species[r])) if s in species[r][i] ] for s in spec_in_file]

                    # check if these are contained in exactly one reference clade
                    n_refs = [len(x) for x in ref_ids]

                    if 0 == min(n_refs) or max(n_refs) > 1:
                        continue

                    ref_ids = [x[0] for x in ref_ids]

                    if len(set(r for (r,i) in ref_ids)) > 1:
                        continue

                    # the first entry of the fasta file has the header informations
                    header_fields = entries[0].id.split("|")
                    locus  = header_fields[3].split(":")
                    seqname = locus[0]
                    posrange = locus[1].replace(",", "").split("-")
                    start = int(posrange[0])
                    end = int(posrange[1])
                    
                    # read the sequences and trim them if wanted
                    sequences = [str(rec.seq).lower() for rec in entries]
                    if margin_width > 0:
                        sequences = [row[margin_width:-margin_width] for row in sequences]

                    if fixed_sequence_length and fixed_sequence_length != len(sequences[0]):
                        # if sequence is too short, skip entry
                        continue

                    msa = MSA(
                            label = label,
                            chromosome_id = seqname, 
                            start_index = start,
                            end_index = end,
                            is_on_plus_strand = True if len(header_fields) < 5 or header_fields[4] != 'revcomp' else False,
                            frame = int(header_fields[2][-1]),
                            spec_ids = ref_ids,
                            offsets = [],
                            sequences = sequences,
                            fname = fasta.filename,
                            use_codons = use_codons
                    )
                    training_data.append(msa)
                    pbar.update(fasta.compress_size)

    return training_data, species


def plot_lenhist(msas, id = "unfiltered"):
    """ plot the length distribution of classes (labels) y=0, and y=1 to a pdf
    Returns:
        mlen  : array of alignment lengths
        labels: array of labels (= classes / model indices) 
    """
    num_alis = len(msas)
    mlen = np.zeros(num_alis, dtype = np.int32)
    labels = np.zeros(num_alis, dtype = np.int32)
    for i, msa in enumerate(msas):
        mlen[i] = msa.alilen()
        labels[i] = msa.label
    
    fig, ax = plt.subplots(1, 2, figsize = (24, 8))
    colors = ["red", "green"]
    for label in [0, 1]:
        these_len = mlen[labels == label]
        ax[label].hist(these_len, bins = 200, range = [0, 2000], density = True, color = colors[label])
        ax[label].set_title("length distribution " + id + " \ny=" + str(label) + 
                           " n=" + str(np.sum(labels == label)) + " mean=" + str(np.round(np.mean(these_len), 1)))
    
    fig.savefig("lendist-oe-" + id + ".pdf", format = 'pdf')
    return [mlen, labels]

def plot_depth_length_scatter(msas, id = "unfiltered", ylim=300):
    """ plot a scatter plot of depth and lengths of classes (labels) y=0, and y=1 to a pdf
    Returns:
        mlen  : array of alignment lengths
        labels: array of labels (= classes / model indices)
    """
    num_alis = len(msas)
    mlen = np.zeros(num_alis, dtype = np.int32)
    mdep = np.zeros(num_alis, dtype = np.int32) # alignment depths = number of rows
    labels = np.zeros(num_alis, dtype = np.int32)
    colors = ["red", "green"]
    for i, msa in enumerate(msas):
        mlen[i] = msa.alilen()
        mdep[i] = msa.alidepth()
        labels[i] = msa.label
        #vprint (f"len={mlen[i]} depth={mdep[i]} label={labels[i]}")


    fig,ax = plt.subplots(1, 2, figsize = (16, 8))
    for label in [0, 1]:
        these_len = mlen[labels == label]
        these_dep = mdep[labels == label]
        scatter = sns.stripplot(these_dep, these_len,
                                jitter=0.4, size=4, alpha=.5,
                                color=colors[label], ax=ax[label])
        scatter.set_title(f"depths and lengths of class {label} (n={len(these_len)})")
        scatter.set_xlabel("MSA depth")
        scatter.set_ylabel("MSA length (codons)")
        scatter.set_ylim(1, ylim)
    plt.savefig("deplenscatter-oe-" + id + ".pdf", format = 'pdf', bbox_inches='tight')
    return [mdep, mlen, labels]

def subsample_lengths(msas, max_sequence_length = 14999, min_sequence_length = 1, relax = 1):
    """ Subsample the [short] negatives so that
        the length distribution is very similar to that of the positives.
        Negative examples (label=0) of a length that is overrepresented compared to the
        frequency of that length in positive examples (label=1) are removed at random.
        Also, filter out 'alignments' with fewer than 2 sequences.
    Args:
        msas: an input list of MSAs
        max_sequence_length: upper bound on number of codons
        min_sequence_length: lower bound on number of codons
        relax: >=1, factor for subsampling probability, if > 1, the 
               subsampling deliveres more data but the negative length
               distribution fits not as closely.
    Returns:
        filtered_msas: a subset of the input
    """
    ### compute and plot lengths and labels
    msas_in_range = []
    num_dropped_shallow = 0
    for msa in msas:
        if len(msa.sequences) < 2:
            num_dropped_shallow += 1
            continue
        length = msa.alilen()
        if msa.use_codons:
            length = int(length / 3)
        else:
            length = int(length / msa.tuple_length)
        if (length >= min_sequence_length and length <= max_sequence_length):
            msas_in_range.append(msa)
    if (num_dropped_shallow > 0):
        print(f"{num_dropped_shallow} MSAs were dropped as they had fewer than 2 rows.")
    mlen, labels = plot_lenhist(msas_in_range)
    assert (len(mlen) == len(labels) and len(mlen) == len(msas_in_range)), "length inconsistency"
    
    ### compute probabilities for subsampling
    max_subsample = 2001 # Don't apply subsampling for longer alignments, there typically are too few.
    distr = np.zeros([2, max_subsample], dtype = float)

    for i, slen in enumerate(mlen):
        if (slen < max_subsample):
            label = labels[i]
            distr[label, slen] += 1.0

    ratio = distr[1] / np.maximum(distr[0], 1.0) # overrepresentation ratio, for each length
    # as the sample has random variation, the ratios are smoothed before using them 
    ratio_smooth = np.zeros_like(ratio)

    def radius(slen):
        """ The offsets to the averaging interval, equal offsets leads to systematic overestimation
            Up to a length of 100 there is no smoothing. Beyond that, it is increasing.
        """
        return [int(.04 * max(slen - 100, 0)), int(.1 * max(slen - 100, 0))]

    for slen in range(1, max_subsample):
        r1, r2 = radius(slen)
        a = max(slen - r1, 0)
        b = min(slen + r2, max_subsample - 1)
        ratio_smooth[slen] = np.mean(ratio[a : b + 1])

    ratio_smooth /= np.max(ratio_smooth)
    ratio_smooth = np.minimum(ratio_smooth * relax, 1.0)

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.plot(ratio_smooth, "b-")
    ax.set_title("length distribution subsampling probabilities")
    fig.savefig("subsampling-probs.pdf", format = 'pdf')

    filtered_msas = []
    for i, msa in enumerate(msas_in_range):
        slen = msa.alilen()
        if msa.label != 0 or slen >= max_subsample or random.random() < ratio_smooth[slen]:
             filtered_msas.append(msa)

    plot_lenhist(filtered_msas, id = "subsampled")
    print ("Subsampling based on lengths has reduced the number of alignments from",
           len(msas), "to", len(filtered_msas))
    return filtered_msas

def subsample_depths_lengths(msas, max_sequence_length = 14999, min_sequence_length = 1,
                             relax = 1, pos_over_neg_mod = 1.0):
    """ Subsample the [short and shallow] negatives and [long and deep] positives so that
        the depth and length distributions are similar between the two classes.
        TODO: Negative examples (label=0) of a length that is overrepresented compared to the
        frequency of that length in positive examples (label=1) are removed at random.
        Also, filter out 'alignments' with fewer than 2 sequences.
    Args:
        msas: an input list of MSAs
        max_sequence_length: upper bound on number of codons
        min_sequence_length: lower bound on number of codons
        relax: >=1, factor for subsampling probability, if > 1, the 
               subsampling deliveres more data but the negative length
               distribution fits not as closely.
    Returns:
        filtered_msas: a subset of the input
    """
    ### compute and plot lengths and labels
    msas_in_range = []
    num_dropped_shallow = 0
    max_depth = 0
    for msa in msas:
        if len(msa.sequences) < 2:
            num_dropped_shallow += 1
            continue
        length = msa.alilen()
        depth = msa.alidepth()
        if depth > max_depth:
            max_depth = depth
        if msa.use_codons:
            length = int(length / 3)
        else:
            length = int(length / msa.tuple_length)
        if (length >= min_sequence_length and length <= max_sequence_length):
            msas_in_range.append(msa)
    if (num_dropped_shallow > 0):
        print(f"{num_dropped_shallow} MSAs were dropped as they had fewer than 2 rows.")
    mdep, mlen, labels = plot_depth_length_scatter(msas_in_range)
    assert (len(mlen) == len(labels) and len(mdep) == len(labels) and len(mlen) == len(msas_in_range)), "length inconsistency"
    
    ### compute probabilities for subsampling
    max_subsample = 2001 # Don't apply subsampling for longer alignments, there typically are too few.
    distr = np.zeros([2, max_depth+1, max_subsample], dtype = float)

    for i in range(len(mlen)):
        if (mlen[i] < max_subsample):
            distr[labels[i], mdep[i], mlen[i]] += 1.0
    # normalize the two joint distributions to sum up to 1
    sums = np.sum(distr, axis=(1,2))
    sumse = np.expand_dims(sums, axis=(1,2))
    distr = distr / sumse
    # overrepresentation ratio, for each length
    epsilon = 1e-2 / len(mdep)
    ratio = (epsilon + distr[1]) / ( epsilon + distr[0]) / pos_over_neg_mod

    # as the sample has random variation, the ratios are smoothed before using them
    ratio_smooth = np.ones_like(ratio)

    def radius(slen):
        """ The offsets to the averaging interval, equal offsets leads to systematic overestimation
            Up to a length of 100 there is no smoothing. Beyond that, it is increasing.
        """
        # return [int(.04 * max(slen - 100, 0)), int(.1 * max(slen - 100, 0))]
        return [int(0 + .15 * slen), int(0 + .2 * slen)]
    
    for depth in range(2, max_depth+1):
        for slen in range(1, max_subsample):
            r1, r2 = radius(slen)
            a = max(slen - r1, 0)
            b = min(slen + r2, max_subsample - 1)
            da = max(depth-1, 2)
            db = min(depth+1, max_depth)
            # average over neighboring depths and nearby lengths
            ratio_smooth[depth, slen] = np.mean(ratio[da:db, a : b + 1])

    if False: # debug printing
        for d in range(2, max_depth+1):
            for ell in range(max_subsample):
                print (f"d={d} ell={ell} ratio_smooth={ratio_smooth[d,ell]} ", end="")
                for j in [0,1]:
                    print (f"label={j} distr={distr[j,d,ell]} ", end="")
                print()

    # ratio_smooth /= np.max(ratio_smooth)
    # ratio_smooth = np.minimum(ratio_smooth * relax, 1.0)

    #fig, ax = plt.subplots(figsize = (6, 6))
    #ax.plot(ratio_smooth, "b-")
    #ax.set_title("length distribution subsampling probabilities")
    #fig.savefig("subsampling-probs.pdf", format = 'pdf')
    #print ("ratio_smooth\n", ratio_smooth)
    
    filtered_msas = []
    filtered_class_nums = [0,0]
    for i, msa in enumerate(msas_in_range):
        keep = True
        if mlen[i] < max_subsample: # very long MSAs are all kept
            r = ratio_smooth[mdep[i], mlen[i]]
            if r < 1.0 and labels[i] == 0: # is negative and negatives are overrepresented
                keep = False
                if random.random() < relax * r:
                    keep = True

            if r > 1.0 and labels[i] == 1: # is positive and positives are overrepresented
                keep = False
                if random.random() < relax / r:
                    keep = True
        if keep:
            filtered_msas.append(msa)
            filtered_class_nums[labels[i]] += 1
    print ("Subsampling based on depths and lengths has reduced the number of alignments from",
           len(msas), "(", sums , ") to", len(filtered_msas), " thereof ", filtered_class_nums, " [neg,pos].")

    if filtered_class_nums[0] > 0 and filtered_class_nums[1] > 0:
        plot_depth_length_scatter(filtered_msas, id = "subsampled")
    return filtered_msas


def subsample_labels(msas, ratio):
    """ Subsample excess examples to that the ratio of negative to positive examples is 'ratio'
    Args:
        msas: an input list of MSAs
        ratio: fraction negatives/positives in the output
    Returns:
        filtered_msas: a subset of the input
    """
    print ("subsampling with ratio", ratio)
    
    if (ratio <= 0.0):
        print ("Warning: ratio_neg_to_pos must be positive. Skipping the subsampling based on label.")
        return msas
    
    num_neg = num_pos = 0
    pos_msas = []
    neg_msas = []
    filtered_msas = []
    for msa in msas:
        if msa.label == 0:
            num_neg += 1
            neg_msas.append(msa)
        elif msa.label == 1:
            num_pos += 1
            pos_msas.append(msa)

    if (num_neg > num_pos * ratio): # too many negatives
        reduced_size = int(num_pos * ratio + 0.5)
        print ("Reducing number of negatives from", num_neg, "to", reduced_size, ". Left number of positives at", num_pos)
        filtered_msas.extend(random.sample(neg_msas, reduced_size))
        filtered_msas.extend(pos_msas)
    else: # too many positives
        reduced_size = int(num_neg / ratio + 0.5)
        print ("Warning: --ratio_neg_to_pos removed positive examples. Maybe you want to omit the parameter to save positves.")
        print ("Reducing number of positives from", num_pos, "to", reduced_size)
        filtered_msas.extend(random.sample(pos_msas, reduced_size))
        filtered_msas.extend(neg_msas)

    random.shuffle(filtered_msas)
    return filtered_msas


def subsample_omegas(msas, ratio, separator):
    """ Subsample excess examples so that the ratio of large to small omegas is at least 'ratio'
    Args:
        msas: an input list of MSAs
        ratio: fraction small/large omegas in the output
        separator: float value to separate the small and large omegas
    Returns:
        filtered_msas: a subset of the input
    """
    print("\n subsampling with ratio: ", ratio, "omegas >= ", separator)
    filtered_msas = []
    n_big = 0
    n_small = 0
    random.shuffle(msas)
    
    omega_values = []
    for msa in msas:
        small, big = msa.get_omegas(separator)
        if (n_big+len(big)) / (n_small+len(small)) >= ratio:
            filtered_msas.append(msa)
            omega_values.append(msa.label)
            n_big += len(big)
            n_small += len(small)
    
    #plot omega values
    # omega_values = list(itertools.chain.from_iterable(omega_values))
    # counts, edges, bars = plt.hist(omega_values, density = False, bins = 40, log = True)
    # plt.bar_label(bars)
    # plt.ylabel("log(counts)")
    # plt.xlabel("omega")
    # plt.show()
            
    print(len(filtered_msas), "of total" ,len(msas), "MSAs sampled.\n Actual ratio: ",n_big/n_small)
    random.shuffle(filtered_msas)
    return filtered_msas


def export_nexus(msas, species, nex_fname, n):
    """ A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.
    Args:
        msas: an input list of MSAs
        nex_fname: output file name
        n: maximal sample size
    """
    positiveMSAs = []
    num_in_frame_stops = num_positives = 0
    for msa in msas:
        if (msa.label == 1):
            num_positives += 1
            msa.codon_aligned_sequences
            if msa.in_frame_stops: # in at least one row
                num_in_frame_stops += 1
            else:
                positiveMSAs.append(msa)
    num_pos = len(positiveMSAs)
    if n > num_pos:
        print ("Warning: Requested NEXUS sample size larger than the number of positive alignments (",
              num_pos, "). Taking all of them as sample.")
        n = num_pos
    if num_in_frame_stops > 0:
        print("Found", num_in_frame_stops, "in-frame stop codons in ", num_positives, "positive alignments. Omitting them.")

    sampledMSAs = random.sample(positiveMSAs, n)
    num_col = 0
    snameset = set()
    for msa in sampledMSAs:
        ca = msa.codon_aligned_sequences
        num_col += len(ca[0])
        # num_col += msa.alilen()
        for (c, l) in msa.spec_ids:
            snameset.add(l)
    sidxs = list(snameset) # all species names occurring in any chosen MSA
    
    # write as .nex file
    clade_specieslist = species[0]
    max_len_speciesname = max(len(name) for name in clade_specieslist)
    nexF = open(nex_fname, "w")
    nexF.write("#NEXUS\n")
    
    # data block (MSA)
    nexF.write("begin data;\n")
    nexF.write("dimensions ntax=" + str(len(sidxs))
               + " nchar=" + str(num_col) + ";\n")
    nexF.write("format datatype=dna interleave=yes gap=-;\nmatrix\n")
    for msa in sampledMSAs:
        msanames = [l for (c, l) in msa.spec_ids]
        
        ca = msa.codon_aligned_sequences
        codonalilen = len(ca[0])
        for sidx in sidxs: 
            nexF.write('{1:{0}}'.format(max_len_speciesname + 2, clade_specieslist[sidx]))
            # write alignment row
            try:
                i = msanames.index(sidx)
                nexF.write(ca[i])
            except ValueError: # row does not exist in this MSA, pad with gaps
                nexF.write("-" * codonalilen)
            nexF.write("\n")
        nexF.write("\n")
    nexF.write(";\nend;\n")
    
    # MrBayes command block
    nexF.write("begin mrbayes;\n")
    nexF.write("set autoclose=yes nowarn=yes;\n")
    #nexF.write("execute " + nex_fname + ";\n")
    nexF.write("lset nst=6 Nucmodel=Codon omegavar=M3 rates=gamma;\n") # for codon models: Nucmodel=Codon omegavar=M3
    nexF.write("mcmc nruns=1 ngen=100000;\n")
    nexF.write("sumt relburnin=yes burninfrac=0.2;\n")
    nexF.write("end;\n")

    nexF.close()
    
# TODO: Delete this function when debug is done. It is now directly implemented in the persistence function
def write_msa(msa, species, tfwriter, use_codons=True, verbose=False):
    """Write a coded MSA (either as sequence of nucleotides or codons) as an entry into a  Tensorflow-Records file.
    
    Args:
        msa (MSA): Sequence that is to be persisted
        use_codons (bool): Whether one should write onehot encoded sequences of codons which are codon-aligned 
                           or a onehot encoded sequences of nucleotides
        tfwriter (TFRecordWriter): Target to which the example shall be written.
        verbose (bool): Whether debug messages shall be written
    """
    

    # Use the correct onehot encoded sequences
    coded_sequences = msa.coded_codon_aligned_sequences if use_codons else msa.coded_sequences
    
    # Infer the length of the sequences
    sequence_length = len(coded_sequences[1])  

    # cardinality of the alphabet that has been onehot-encoded
    s = coded_sequences.shape[-1]
    
    # get the id of the used clade and leaves inside this clade
    clade_id = msa.spec_ids[0][0]
    num_species = len(species[clade_id])
    leaf_ids = [l for (c,l) in msa.spec_ids]
    
    
    # embed the coded sequences into a full MSA for the whole leaf-set of the given clade
    S = np.ones((num_species, sequence_length, s), dtype = np.int32)
    S[leaf_ids,...] = coded_sequences
    
    # make the shape conform with the usual way datasets are structured,
    # namely the columns of the MSA are the examples and should therefore
    # be the first axis
    S = np.transpose(S, (1,0,2))
    


    # use label (`0` or `1`), the id of the clade and length of the sequences
    # as context features
    msa_context = tf.train.Features(feature = {
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [msa.label])),
        'clade_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [clade_id])),
        'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
    })

    ## save `S` as a one element byte-sequence in feature_lists
    sequence_feature = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [S.tostring()]))]
    msa_feature_lists = tf.train.FeatureLists(feature_list = {
        'sequence_onehot': tf.train.FeatureList(feature = sequence_feature)
    })

    
    # create the SequenceExample
    msa_sequence_example = tf.train.SequenceExample(
        context = msa_context,
        feature_lists = msa_feature_lists
    )

    # write the serialized example to the TFWriter
    msa_serialized = msa_sequence_example.SerializeToString()
    tfwriter.write(msa_serialized)

    
    
def preprocess_export(dataset, species, splits = None, split_models = None,
                      verbose = False):
    
    # Prepare for iteration
    splits = splits if splits != None else {None: 1.0}
    split_models = split_models if split_models != None else [None]

    # Check whether splits are valid
    split_values_are_numbers = all([ isinstance(x, numbers.Number) for x in list(splits.values()) ])

    if not split_values_are_numbers:
        raise ValueError("The values of the dict `splits` must be numbers. But it is given by {splits}")

    # Convert relative numbers to absolute numbers
    random.shuffle(dataset)
    n_data = len(dataset)
    requested_sizes = np.array(list(splits.values()))
    negs = np.nonzero(requested_sizes < 0) # maximally fill the negative sizes, e.g. -1
    to_total = lambda x: int(max(x, 0) * n_data) if isinstance(x, float) else max(x, 0)
    
    split_totals = np.array([to_total(x) for x in requested_sizes])
    n_wanted = sum(split_totals)
    if len(negs[0]) > 0 and n_wanted < n_data:
        # divide the remaining examples between the ones with requested negative size
        each_gets = int((n_data - n_wanted) / len(negs[0]))
        if verbose:
            print("Subsets (splits) with requested negative size each get ", each_gets, "alignents.")
        split_totals[negs] = each_gets
        n_wanted = sum(split_totals) # = n_data
    
    
    # rescale accordindly
    if n_wanted > n_data:
        split_totals = split_totals * (n_data / n_wanted)
    print("Split totals:", split_totals)
    # The upper bound indices used to decide where to write the `i`-th entry
    split_bins = np.cumsum(split_totals)
    if n_wanted > n_data:
        n_wanted = int(split_bins[-1])
    # print ("split_bins=", split_bins , "\nsplits=", splits, "\nsplit_models=", split_models, "\nn_wanted=", n_wanted)
    return splits, split_models, split_bins, n_wanted


def persist_as_tfrecord(dataset, out_dir, basename, species,
                        splits=None, split_models=None, split_bins=None, 
                        n_wanted=None, use_compression=True, sitewise = False,
                        num_positions = None, verbose=False, no_codon_alignment=False):
    # Importing Tensorflow takes a while. Therefore to not slow down the rest 
    # of the script it is only imported once used.
    print ("Writing to tfrecords...")
    import tensorflow as tf

    options = tf.io.TFRecordOptions(compression_type = 'GZIP') if use_compression else None


    # Generate target file name based on chosen split and label
    
    target_path = lambda split_name, label: \
        os.path.join(out_dir, # this appends a slash if the user did not
                     basename
                     + ('-' + split_name if split_name != None else '')
                     + ('-m' + str(label) if label != None else '')
                     + '.tfrecord'
                     + ('.gz' if use_compression else ''))
    
    with ExitStack() as stack:
    
        #file_indices = [(split_name, model, ) for ]
        tfwriters = [[
                stack.enter_context(tf.io.TFRecordWriter(target_path(sn,m), options = options))
            for m in split_models] for sn in splits]

        n_written = np.zeros([len(splits), len(split_models)])
        
        for i in tqdm(range(n_wanted), desc="Writing TensorFlow record", unit=" MSA"):
            msa = dataset[0]
            del dataset[0]
            label = msa.label

            # retrieve the wanted tfwriter for this MSA
            # split_idx is the index of the split, m is the class label (model)
            split_idx = np.digitize(i, split_bins)
            m = split_models.index(label) if label in split_models else 0 
            tfwriter = tfwriters[split_idx][m]

            # Write a coded MSA (either as sequence of nucleotides or codons) as an entry into a  Tensorflow-Records file.
            # in order to do so we need to setup the proper format for `tf.train.SequenceExample`
            
            msa.is_codon_aligned = no_codon_alignment # assumed to already be in codons
            # Use the correct onehot encoded sequences
            coded_sequences = msa.coded_codon_aligned_sequences if msa.use_codons or (msa.tuple_length > 1 and not msa.tuples_overlap) \
                                                                else msa.coded_sequences

            # Infer the length of the sequences
            sequence_length = coded_sequences.shape[1]

            if sequence_length == 0:
                continue

            elif num_positions and num_positions != sequence_length:
                if (num_positions - sequence_length) % 2 != 0:
                    # the sequence will be cut and padded symmetrically on both sides to get num_positions
                    # this cant happen if num_positions - sequence_length is an odd number
                    continue
                if num_positions > sequence_length:
                    # if sequence is too short, pad with 1 on both sides to get required length
                    pad_width = int((num_positions - sequence_length)/2)  # width to pad on each side
                    coded_sequences = np.pad(coded_sequences, ((0,0), (pad_width,pad_width), (0,0)), constant_values=1)
                else:
                    # if sequence is too long, remove margin from sequence to get required length
                    marg_width = int((sequence_length - num_positions)/2)  # width to remove on each side
                    coded_sequences = coded_sequences[:, marg_width:-marg_width, :]
                sequence_length = coded_sequences.shape[1]
   
            # cardinality of the alphabet that has been onehot-encoded
            s = coded_sequences.shape[-1]

            # get the id of the used clade and leaves inside this clade
            clade_id = msa.spec_ids[0][0]
            num_species = len(species[clade_id])
            leaf_ids = [l for (c,l) in msa.spec_ids]


            # embed the coded sequences into a full MSA for the whole leaf-set of the given clade
            S = np.ones((num_species, sequence_length, s), dtype = np.int32)
            S[leaf_ids,...] = coded_sequences

            # make the shape conform with the usual way datasets are structured,
            # namely the columns of the MSA are the examples and should therefore
            # be the first axis
            S = np.transpose(S, (1,0,2))


            if sitewise:
                if len(label) != sequence_length:
                    print(f"The length of the list of labels ({len(label)})",
                          f"differs from the sequence length ({sequence_length}).",
                          file=sys.stderr)
                    continue
                # use the id of the clade and length of the sequences as context features
                msa_context = tf.train.Features(feature = {
                    'clade_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [clade_id])),
                    'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
                    })

                ## save `S` as a one element byte-sequence and label in feature_lists
                label_feature = [tf.train.Feature(float_list = tf.train.FloatList(value = [single_value])) for single_value in msa.label]
                sequence_feature = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [S.tostring()]))]
                msa_feature_lists = tf.train.FeatureLists(feature_list = {
                    'sequence_onehot': tf.train.FeatureList(feature = sequence_feature),
                    'label': tf.train.FeatureList(feature = label_feature)
                    })


                # create the SequenceExample
                msa_sequence_example = tf.train.SequenceExample(
                        context = msa_context,
                        feature_lists = msa_feature_lists
                        )

                # write the serialized example to the TFWriter
                msa_serialized = msa_sequence_example.SerializeToString()
                tfwriter.write(msa_serialized)
                n_written[split_idx][m] += 1
            else:
                # use label (`0` or `1`), the id of the clade and length of the sequences
                # as context features
                msa_context = tf.train.Features(feature = {
                    # use "model" here for backwards compability
                    'model': tf.train.Feature(int64_list = tf.train.Int64List(value = [msa.label])),
                    'clade_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [clade_id])),
                    'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
                    })
    
                ## save `S` as a one element byte-sequence in feature_lists
                sequence_feature = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [S.tostring()]))]
                msa_feature_lists = tf.train.FeatureLists(feature_list = {
                    'sequence_onehot': tf.train.FeatureList(feature = sequence_feature)
                    })
    
    
                # create the SequenceExample
                msa_sequence_example = tf.train.SequenceExample(
                        context = msa_context,
                        feature_lists = msa_feature_lists
                        )
    
                # write the serialized example to the TFWriter
                msa_serialized = msa_sequence_example.SerializeToString()
                tfwriter.write(msa_serialized)
                n_written[split_idx][m] += 1

    
    print ("number of tf records written [rows: split bin s, column: model/label m]:\n", n_written)


def get_end_offset(start_offset, seqlen):
    """ 
       get the largest position where a complete codon could start,
       end_offset - start_offset is a multiple of 3 so incomplete codons are truncated
       TODO: this can fail if the alignment boundary were to contains gaps
    """
    end_offset = seqlen - 3 # -3 so, a full codon could end right at end_offset
    end_offset = start_offset + math.floor((end_offset - start_offset) / 3) * 3
    return end_offset

    
def write_phylocsf(dataset, out_dir, basename, species,
                   splits = None, split_models = None, split_bins = None, 
                   n_wanted = None, refid = None, orig_fnames = False):
    """
       Each MSA is written into a single text file in a FASTA format required by PhyloCSF
       and accepted by clamsa predict. In particular,
       - the putative codon MSA is one the forward strand and
       - the output phase is 0, i.e. the alignment would start with a complete codon if y=1
       - the refid (index of species in its clade) is listed as first sequence if it is present.
    """
    print ("Writing to PhyloCSF flat files...")
    classnames = ["controls", "exons"]
    subdir_size = 500
    phyloDEBUG = False
    margin_width = 0
    splitnames = list(splits.keys()) # e.g. train, val1, val2, test
    class_counts = [0, 0] # how many examples of class y=0 and y=1 have been seen yet
    n_written = np.zeros([len(splits), len(split_models)])
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(n_wanted), desc = "Writing PhyloCSF dataset", unit = " MSA"):
        s = np.digitize(i, split_bins)
        split_dir = os.path.join(out_dir, basename, splitnames[s])
        
        msa = dataset[i]
        # get the id of the used clade and leaves inside this clade
        clade_id = msa.spec_ids[0][0]
        phyloCSFspecies = species[clade_id] # correct?  
    
        frame = msa.frame
        y = msa.label
        assert y == 0 or y == 1, "PhyloCSF output expects binary class"
        
        class_dir = os.path.join(split_dir, classnames[y])
        subdir = os.path.join(class_dir, "{:03d}".format(int(class_counts[y] / subdir_size)))

        if orig_fnames and msa.fname:
            fname = msa.fname
        else:
            fname = os.path.join(subdir, "{:03d}.fa".format(class_counts[y]))

        # create all necessary parent directories like with mkdir -p
        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok = True)

        fa = open(fname, "w+")
        
        class_counts[y] += 1
        # indices to species ids sorted so reference is first
        # not required by PhyloCSF

        if refid:
            sids = sorted(range(len(msa.spec_ids)), key = lambda k: ((msa.spec_ids[k])[1] != refid))
        else:
            sids = range(len(msa.spec_ids))

        newrows = [] # manipulated alignment rows
        # to complement a sequence
        alphabet = "acgt"
        tbl = str.maketrans(alphabet, alphabet[::-1])

        """
               Shift start to codon boundary. Example (f=2):
               c-gatgttg           atgttg
               -tgatgttg  ======>  atgttg
               c---t-ttg           ---ttg
        """
        maxrowlen = -1
        if msa.use_codons:
            rows = msa.codon_aligned_sequences
            frame = 0
            on_plus_strand = True
        else:
            rows = msa.sequences
            on_plus_strand = msa.is_on_plus_strand

        # in a first pass, delete the first f non-gap chars from each seq
        for j, k in enumerate(sids):
            oldseq = rows[sids[j]]
            if not on_plus_strand: # on minus strand
                oldseq = oldseq[::-1].translate(tbl) # reverse and complement
            i = c = 0
            while c < frame and i < len(oldseq):
                if oldseq[i] != '-':
                    c += 1
                i += 1

            newseq = oldseq[i:]
            maxrowlen = max(maxrowlen, len(newseq))
            newrows.append(newseq)
        # in the second pass, prepend gaps so each alignment row has the same length
        # output MSA

        for j, k in enumerate(sids):
            (_, k) = msa.spec_ids[k]
            fa.write(">" + phyloCSFspecies[k])
            if j == 0:
                fa.write("|y=" + str(y) + "|phase=0|") # frame is corrected to 0, strand to +
                if msa.chromosome_id is not None \
                   and msa.start_index is not None \
                   and msa.end_index is not None:
                    fa.write(msa.chromosome_id + ":" + str(msa.start_index) + "-" + str(msa.end_index))
                    fa.write("||originally:f=" + str(frame) + ",strand=" + ("+" if msa.is_on_plus_strand else "-"))

            fa.write("\n")
            seq = newrows[j]
            fa.write("-" * (maxrowlen - len(seq)) + seq + "\n") # padd with gaps if MSA frayed after frame correction

        n_written[s][y] += 1
        fa.close()
    print ("number of PhyloCSF records written [rows: split bin s, column: model/label m]:\n", n_written)

def maf_generator(mafpath : str):
    """ Generator for MAF files. 
        It generates multiple sequence alignments
    """
    try:
        from Bio import AlignIO # to read MAF files
        from Bio.Align import MultipleSeqAlignment
    except ImportError:
        print("Please install Biopython to use the MAF format.")
        sys.exit(1)
    
    opener = open # for regular text files
    if '.gz' in pathlib.Path(mafpath).suffixes:
        opener = gzip.open
    
    try:
        with opener(mafpath, "rt") as msas_file:
            msa = None
            for msa in AlignIO.parse(msas_file, "maf"):
                yield msa
            if msa is None:
                print("Error: no MSA found in file %s" % mafpath)
                sys.exit(1)
                        
    except OSError:
        print ("Error: cannot open file %s" % mafpath)
        sys.exit(1)
