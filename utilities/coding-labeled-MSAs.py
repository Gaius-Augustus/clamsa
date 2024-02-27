# Compile MSAs for ClaMSA sitewise that are labeled with whether a codon MSA site is coding or not
from Bio import AlignIO # to read MAF files
from Bio.Align import MultipleSeqAlignment
from intervaltree import Interval, IntervalTree
import argparse
import pathlib
import gzip
import sys
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, Lock
from itertools import repeat, islice


def read_gff(gff_file, strand = "both"):
    """
    Read a GFF file into a hash of interval trees.
    Keys are the chromosome/seq names, values are interval trees.
    The data in the tree is a pair (strand, frame).
    """
    forest = {}

    # conditionally open a file or a gzip file
    opener = open # for regular text files
    if '.gz' in pathlib.Path(gff_file).suffixes:
        opener = gzip.open

    with opener(gff_file, "rt") as gff:
        for line in gff:
            if line[0] == '#':
                continue
            f = line.strip().split('\t')
            if f[2] == 'CDS':
                seqid = f[0]
                start = int(f[3])
                end = int(f[4]) + 1 # exclusive in intervaltree
                if f[6] == '+':
                    _strand = 1
                elif f[6] == '-':
                    _strand = -1
                else:
                    print ("Error: strand must be + or -\n", line)
                    _strand = None
                frame = int(f[7])
                if end <= start: # happens, e.g. in NCBI Refseq GFFs
                    continue
                if (seqid not in forest):
                    forest[seqid] = IntervalTree()
                if strand == "both" or strand == _strand:
                    forest[seqid].addi(start, end, (_strand, frame))

    return forest


def process_MSA(msa, refspecies, sample_freq_nongenic, sample_freq_ovlp_oppositestrand, minlen):
    """
    Process a single MSA. Use the global forest variable to determine which codon sites are coding.
    Output 6 new site-labeled MSAs in FASTA format, one for each frame and strand.
    TODO: remove gap-only columns from msa
    """
    refseqrec = None
    for seqrec in msa:
        species, chr = seqrec.id.split('.', 1)
        if species == refspecies:
            refseqrec = seqrec
            break
    if refseqrec is None:
        print ("Warning: reference species %s not found in the MSA" % refspecies)
        return
    # print ("ref=", refseqrec)
    refchrStart = refseqrec.annotations["start"] + 1 # 0-based in MAF, here 1-based for GFF comp.
    refchrLen = refseqrec.annotations["size"]
    refchrEnd = refchrStart + refchrLen # exclusive

    refrow = str(seqrec.seq)
    if len(refrow) < refchrLen:
        print ("Error: alignment row length %s smaller than annotated (%s)" % (len(refrow), refchrLen))
        sys.exit(1)

    if chr not in forest:
        print ("Warning: reference chromosome %s not found in the annotation" % chr)
        return
    tree = forest[chr]
    tree_oppo = forest_oppo[chr]

    ovlpCDS = sorted(tree[refchrStart:refchrEnd]) # all CDS overlapping the reference sequence
    ovlpCDS_oppo = sorted(tree_oppo[refchrStart:refchrEnd]) # all overlapping minus strand CDS

    sixMSAs = []
    pos_sites = sites = 0
    if ovlpCDS or \
        (ovlpCDS_oppo and np.random.random() < sample_freq_ovlp_oppositestrand) or \
        np.random.random() < sample_freq_nongenic:
        sixMSAs, pos_sites, sites = six_frame_loop(msa, refrow, refchrStart, ovlpCDS, minlen)

    return sixMSAs, pos_sites, sites

def six_frame_loop(msa:MultipleSeqAlignment, refrow:str,
                   refchrStart:int, ovlpCDS:list[Interval], minlen=7):
    """
    Loop over the 6 frames of the reference sequence and find the classes of codon sites
    """
    alilen = len(refrow)
    sixMSAs = []
    pos_sites = sites = 0
    """
    distinguish 3 types of coordinates:
    chrPos in genome, alipos in MSA, i in reference row (no gaps)
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

    for frame in range(3):
        for strand in (1, -1):
            if strand == -1:
                continue # reverse strand not implemented
            # print (f"frame={frame} strand={strand}")
            alipos = 0
            # read over gaps at the beginning of the reference row
            while refrow[alipos] == '-' and alipos < alilen:
                alipos += 1
            alipos, gapsWithin, gapsAfter = right_move(0, frame)
            i = frame
            outMSA = None
            outalilen = 0
            classes = []
            while alipos < alilen:
                nextalipos, gapsWithin, gapsAfter = right_move(alipos)
                # throw away codons that are interrupted by gaps
                if gapsWithin == 0 and nextalipos >= alipos + 3:
                    # cut out the codon from msa
                    msa1codon = msa[:,alipos:alipos+3]
                    chrPos = refchrStart + i
                    codingclass = coding_class(chrPos, strand, ovlpCDS) 
                    classes.append(codingclass)
                    if debug:
                        append_char(msa1codon, ' ')
                    if outMSA is None:
                        outMSA = msa1codon
                    else:
                        outMSA += msa1codon
                    outalilen += 3
                    # print (f"codon column chrPos={chrPos} i={i} alipos={alipos}", msa1codon)
                   
                alipos = nextalipos
                i += 3
            if outMSA is None:
                continue
            #assert outalilen == outMSA.get_alignment_length(),\
            #    f"alignment length mismatch {outalilen} != {outMSA.get_alignment_length()}"
            if outalilen >= minlen * 3:
                # add info to header
                # 1. genome region
                outMSA[0].id += ":" + str(refchrStart)
                # 2. list of class labels
                sepstr = "," if not debug else ",  "
                outMSA[0].id += "|" + sepstr.join(str(c) for c in classes)
                sixMSAs.append(outMSA)
                pos_sites += sum(classes)
                sites += len(classes)
                
    return sixMSAs, pos_sites, sites 


def append_char(msa, char):
    """
    Append a character to all sequences in the MSA
    """
    for seqrec in msa:
        seqrec.seq += char

def coding_class(chrPos:int, cstrand:int, ovlpCDS:list[Interval]):
    """
    Determine the coding class of a codon site
    Return 1 if a codon starts at chrPos in any of the CDS intervals, 0 otherwise.
    """
    for cds in ovlpCDS:
        (strand, frame) = cds.data
        isCoding = ( cstrand == 1 and strand == 1 and # currently only works for + strand
                    chrPos >= cds.begin and chrPos + 2 < cds.end # triplet is within the CDS range
                    and (chrPos - cds.begin - frame) % 3 == 0 ) # right frame
        if isCoding:
            # print (f"isCoding={isCoding} chrPos={chrPos} in {cds.begin} {cds.end} {strand} {frame}")
            return 1
    return 0

if __name__ == '__main__':

    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description = 'Compile MSAs for ClamSA sitewise that are labeled with whether a codon MSA site is coding or not', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--inmsa', required=True, type=str, help='MSA input file in MAF format. Can be gzipped.')
    parser.add_argument('--outmsa', required=True, type=str, help='MSA output file in FASTA format. Will be gzipped if ends in .gz')
    parser.add_argument('--refspecies', required=True, type=str,
                        help='Reference species. Must be the same as the one used in the GFF file')
    parser.add_argument('--anno', required=True, type=str, help='genome annotation file in GFF')
    parser.add_argument('--minlen', type=int, help='minimum MSA length in codons', default=7)
    parser.add_argument('--sample_freq_nongenic', type=float, help='Alignments not overlapping a gene on the forward strand are sampled with this probability.', default=0.02)
    parser.add_argument('--sample_freq_ovlp_oppositestrand', type=float, help='Alignments overlapping a gene on the reverse strand are sampled with this probability.', default=0.5)

    args = parser.parse_args()

    if args.sample_freq_nongenic < 0 or args.sample_freq_nongenic > 1:
        print ("Error: sample_freq_nongenic must be between 0 and 1")
        sys.exit(1)

    if args.sample_freq_ovlp_oppositestrand < 0 or args.sample_freq_ovlp_oppositestrand > 1:
        print ("Error: sample_freq_ovlp_oppositestrand must be between 0 and 1")
        sys.exit(1)

    num_MSA = pos_sites = sites = 0
    debug = False

    # read the annotation file
    forest = read_gff(args.anno, strand = +1)

    # to allow the enrichment of false positives on the opposite strand
    # collect coding regions on the reverse strand
    forest_oppo = read_gff(args.anno, strand = -1)

    # stream through the MSA file

    # conditionally open a regular text or a gzip file for input
    opener = open # for regular text files
    if '.gz' in pathlib.Path(args.inmsa).suffixes:
        opener = gzip.open

    out_opener = open
    if '.gz' in pathlib.Path(args.outmsa).suffixes:
        out_opener = gzip.open

    outf = out_opener(args.outmsa, "wt")

    try:
        with opener(args.inmsa, "rt") as msas_file:
            lock = Lock()
            with lock:
                with Pool(8) as pool:
                    jobitemsit = zip(tqdm(AlignIO.parse(msas_file, "maf"), unit="MSAs"),
                                          repeat(args.refspecies),
                                          repeat(args.sample_freq_nongenic),
                                          repeat(args.sample_freq_ovlp_oppositestrand),
                                          repeat(args.minlen))
                    print (f"\nProcessing MSAs in parallel with {pool._processes} processes")
                    while jobitemsit:
                        batch = list(islice(jobitemsit, 10000))
                        if not batch:
                            break
                        r = pool.starmap(process_MSA, batch)
                        for msas, p, s in r:
                            pos_sites += p
                            sites += s
                            for msa in msas:
                                num_MSA += 1
                                AlignIO.write(msa, outf, "fasta")
                                # append an empty line to separate MSAs
                                outf.write("\n")

    except OSError:
        print ("Error: cannot open file %s" % args.inmsa)
        sys.exit(1)

    outf.close()

    ratio = pos_sites/sites if sites else 0
    print (f"{num_MSA} MSAs were output with a total of {pos_sites} ",
           f"positive sites ({ratio*100:.2f}%).")
