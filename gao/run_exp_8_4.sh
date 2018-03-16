#!/bin/bash
source activate my_torch

python cnn_main.py --imdb-path='./imdb_DBF.pkl' \
                   --filter-sizes='3_3_3_3' \
                   --channels='256_256_256_256' \
                   --dilation='1_1_1_1' \
                   --ndim=50 \
                   --epochs=40 \
                   --start-epoch=0 \
                   --batch-size=128 \
                   --lr=0.01 \
                   --lr-step=20 \
                   --momentum=0.9 \
                   --print-freq=100 \
                   --droprate=0 \
                   --resume='' \
                   --save="results_8_4"
