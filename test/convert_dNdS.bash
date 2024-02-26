#A small example for the convert function of dNdS training data.
#10 example alignments are in the folder: ../examples/dNdS_alignments/

python ../clamsa.py convert fasta ../examples/dNdS_alignments/*.fa \
--basename amniota \
--splits '{"train-m0": 0.7, "test-m0": 0.2, "val-m0": 0.1}' \
--tf_out_dir ../data/train_dNdS \
--clades ../examples/amniota.nwk \
--use_codons \
--sitewise \
| tee dNdS_convert.log
