from more_itertools import last
import numpy as np
from contextlib import ExitStack
import sys
import math

files = {}
def file(out_file_stem, species, chrStart, strand):
    genomeFrame = chrStart % 3
    strandstr = "plus" if strand == 1 else "minus"
    fname = f"{out_file_stem}_{species}_{genomeFrame}-{strandstr}.wig"
    # print (chrStart, fname)
    if fname in files:
        return files[fname]
    else:
        files[fname] = open(fname, 'w')
        return files[fname]


def output_wig_record_ref(wigfile, seqname, chrPos, numSites, y, logits=False):
    """
    Write a single record of the alignment reference to a wiggle file.
    """
    wigfile.write(f"fixedStep chrom={seqname} start={chrPos} step=3\n")
    if logits:
        # apply reverse sigmoid for better visualization at extremes
        epsilon = 1e-4
        y = np.log((y + epsilon)/(1 - y + epsilon))
    wigstr = ""
    for i in range(numSites):
        wigstr += f"{y[i]:.4f}\n"
    wigfile.write(wigstr)



def output_wig_record_all(out_file_stem, species, chr, chrStart,
                          rowseq, numSites, y, strand, logits=False):
    """
    Write a single record to a wiggle file, in the case where we output for all species.
    Example:
     rowseq:   acg|c-t|cc-c|acc|tct|cgg
    Warning: The fixedSteps chunks are not necessarily in order
    """
    # print (f"output_wig_record_all: {species}, {chr}, {chrStart}, {rowseq}, {numSites} {y} {strand}")
    if logits:
        # apply reverse sigmoid for better visualization at extremes
        epsilon = 1e-4
        y = np.log((y + epsilon)/(1 - y + epsilon))
    fixedStepStart = None
    lastChrStart = None
    i = 0
    wigstr = ""
    for i in range(numSites):
        alipos = 3*i
        # print ("processing codon", rowseq[alipos:alipos+3])
        gaps = rowseq[alipos:alipos+3].count('-')
        if gaps == 0:
            if lastChrStart is None or chrStart != lastChrStart + 3:
                if fixedStepStart is not None:
                    #print(wigstr)
                    wigfile.write(wigstr)
                    wigstr = ""
                fixedStepStart = chrStart
                wigstr += f"fixedStep chrom={chr} start={chrStart} step=3\n"
                wigfile = file(out_file_stem, species, chrStart, strand)
            lastChrStart = chrStart
            chrStart += 3
            wigstr += f"{y[i]:.4f}\n"
        else:
            #print ("gap")
            # a gap interrupts the fixedStep
            lastChrStart = chrStart
            chrStart += 3 - gaps

    if wigstr != "":
        #print (wigstr)
        wigfile.write(wigstr)


def write_preds_to_wig(preds, aux, out_file_stem, output_all_species=False, logits=False):
    if output_all_species:
        write_preds_to_wig_all(preds, aux, out_file_stem, logits=logits)
    else:
        write_preds_to_wig_ref(preds, aux, out_file_stem, logits=logits)


def write_preds_to_wig_all(preds, aux, out_file_stem, logits=False):
    """
    Write predictions to 6 wiggle files:
    out_file_stem_{0,1,2}-{plus,minus}.wig 
    """
    isCoding = preds[:,1] # 2 classes: 0 = non-coding, 1 = coding
    # print ("aux:", aux)

    i=0
    cumSites = 0
    for msa in aux:
        # print (f"fragment {i}", msa)
        y = isCoding[cumSites:cumSites + msa['numSites']]
        for i, seq in enumerate(msa['seqs']):
            fragLen = msa['fragLens'][i]
            chrStart = msa['chrStarts'][i]
            aliStrand = msa['aliStrands'][i] # 1 = plus, -1 = minus
            chrSize = msa['chrSizes'][i]
            offset = msa['offsets'][i] # for chrStart and fraglen
            chrStart += offset
            fragLen -= offset
            if aliStrand == -1:
                chrStart = chrSize - chrStart - fragLen
            strand = msa['codon_strand'] * aliStrand
            # reverse codon on minus strand is on positive strand in genome

            output_wig_record_all(out_file_stem,
                                  msa['species'][i],
                                  msa['chrs'][i],
                                  chrStart,
                                  seq,
                                  msa['numSites'],
                                  y,
                                  strand,
                                  logits=logits)

        cumSites += msa['numSites']
        i += 1

    print (f"cumulative sites: {cumSites}")
    if cumSites != len(isCoding):
        print (f"ERROR: cumulative sites = {cumSites} != {len(isCoding)} = len(isCoding)")
        sys.exit(1)
    # close all files
    for f in files:
        files[f].close()

def write_preds_to_wig_ref(preds, aux, out_file_stem, logits=False):
    """
    Append predictions to 6 wiggle files per species
    out_file_stem_species_{0,1,2}-{plus,minus}.wig
    """
    isCoding = preds[:,1] # 2 classes: 0 = non-coding, 1 = coding
    # print (len(isCoding), isCoding)
    # print ("aux:", aux)

    with ExitStack() as stack:
        wigfiles = [stack.enter_context(open(out_file_stem + str(f) + ".wig", 'w')) 
                    for f in range(6)]

        i=0
        cumSites = 0
        for msa in aux:
            # print (f"fragment {i}", msa)
            y = isCoding[cumSites:cumSites + msa['numSites']]
            # wiggle starts coordinates at 1, not 0
            s = 1 if msa['codon_strand'] == 1 else 0 # s=1 if plus strand, 0 if minus strand
            genomeFrame = msa['chrPos'] % 3 + (1 - s)*3
            # print (msa['seqname'], msa['chrPos'], msa['numSites'], y, "genomeFrame:", genomeFrame)
            output_wig_record_ref(wigfiles[genomeFrame], msa['seqname'],
                              msa['chrPos'], msa['numSites'], y, logits=logits)

            cumSites += msa['numSites']
            i += 1

        print (f"cumulative sites: {cumSites}")
        if cumSites != len(isCoding):
            print (f"ERROR: cumulative sites = {cumSites} != {len(isCoding)} = len(isCoding)")
            sys.exit(1)
