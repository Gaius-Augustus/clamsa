import numpy as np
from contextlib import ExitStack
import sys
import math

def output_wig_record(wigfile, seqname, chrPos, numSites, y, logits=False):
    """
    Write a single record to a wiggle file.
    """
    wigfile.write(f"fixedStep chrom={seqname} start={chrPos} step=3\n")
    for i in range(numSites):
        z = y[i]
        if logits:
            # apply reverse sigmoid for better visualization at extremes
            epsilon = 1e-4
            z = math.log((z + epsilon)/(1 - z + epsilon))
        wigfile.write(f"{z:.4f}\n")

def write_preds_to_wig(preds, aux, out_file_stem, logits=False):
    """
    Write predictions to 6 wiggle files:
    out_file_stem_{0,1,2}-{plus,minus}.wig 
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
            genomeFrame = msa['chrPos'] % 3 + (1-msa['plus_strand'])*3
            # print (msa['seqname'], msa['chrPos'], msa['numSites'], y, "genomeFrame:", genomeFrame)
            output_wig_record(wigfiles[genomeFrame], msa['seqname'],
                              msa['chrPos'], msa['numSites'], y, logits=logits)

            cumSites += msa['numSites']
            i += 1

        print (f"cumulative sites: {cumSites}")
        if cumSites != len(isCoding):
            print (f"ERROR: cumulative sites = {cumSites} != {len(isCoding)} = len(isCoding)")
            sys.exit(1)
        
