#!/bin/sh

inputdir="run/gnn_evaluation/test"
outputdir="run/trkx_from_gnn"
maxevts=100

python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut 0.0 \
    --epsilon 0.25 \
    --min-samples 2