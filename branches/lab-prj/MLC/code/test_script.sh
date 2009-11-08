#!/bin/sh
# BCH Encoder/Decoder

export MM=13
export KK=4096
export TT=4
export PP=8
export EE=16

./data_generator -n $KK > data_in.txt
./bch_encoder -m $MM -k $KK -t $TT -p $PP < data_in.txt > data_codeword.txt
./error -e $EE < data_codeword.txt > data_error.txt
./bch_decoder -m $MM -k $KK -t $TT -p $PP < data_error.txt > data_out.txt
