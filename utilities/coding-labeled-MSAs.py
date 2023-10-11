# Compile MSAs for ClamSA sitewise that are labeled with whether a codon MSA site is coding or not
from Bio import AlignIO # to read MAF files
from intervaltree import Interval, IntervalTree
import argparse
import pathlib
import gzip

# parse command-line arguments
parser = argparse.ArgumentParser(
    description = 'Compile MSAs for ClamSA sitewise that are labeled with whether a codon MSA site is coding or not')


parser.add_argument('--msa', required=True, type=str, help='MSA file in MAF format')
parser.add_argument('--refspecies', required=True, type=str,
                    help='Reference species. Must be the same as the one used in the GFF file')
parser.add_argument('--anno', required=True, type=str, help='genome annotation file in GFF')

args = parser.parse_args()

def read_gff(gff_file):
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
                end = int(f[4])
                strand = f[6]
                frame = int(f[7])
                if end <= start: # happens, e.g. in NCBI Refseq GFFs
                    continue
                if (seqid not in forest):
                    forest[seqid] = IntervalTree()
                forest[seqid].addi(start, end, (strand, frame))

    return forest

forest = read_gff(args.anno)
# print(forest)

def process_MSA(msa):
    """
    Process a single MSA. Use the global forest variable to determine which codon sites are coding.
    Output 6 new site-labelled MSAs in FASTA format, one for each frame and strand.
    """
    refseqrec = None
    for seqrec in msa:
        species, chr = seqrec.id.split('.', 1)
        if species == args.refspecies:
            refseqrec = seqrec
            break
    if refseqrec is None:
        print ("Warning: reference species %s not found in the MSA" % args.refspecies)
        return
    # print ("ref=", refseqrec)
    refchrstart = refseqrec.annotations["start"]
    refchrlen = refseqrec.annotations["size"]
    refchrEnd = refchrstart + refchrlen
    if chr not in forest:
        print ("Warning: reference chromosome %s not found in the annotation" % chr)
        return
    tree = forest[chr]
    ovlpCDS = sorted(tree[refchrstart:refchrEnd])
    if ovlpCDS:
        print (f"{refchrstart}-{refchrEnd} overlapping {len(ovlpCDS)}")
        print (ovlpCDS)
    #print("%s starts at %s on the %s strand of a sequence %s in length, and runs for %s bp"
    #    % ( seqrec.id, seqrec.annotations["start"], seqrec.annotations["strand"], seqrec.annotations["srcSize"], seqrec.annotations["size"]))


# stream through the MSA file

# conditionally open a regular text or a gzip file
opener = open # for regular text files
if '.gz' in pathlib.Path(args.msa).suffixes:
    opener = gzip.open

with opener(args.msa, "rt") as msas_file:
    for msa in AlignIO.parse(msas_file, "maf"):
        process_MSA(msa)
