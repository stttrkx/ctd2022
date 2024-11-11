#!/bin/bash

# This script runs 'trkx_from_gnn.py'

# params
epsilon=0.15
maxevts=30000
edge_score_cut=0.5


# input
if test "$1" != ""; then
  epsilon=$1
fi

if test "$2" != ""; then
  maxevts=$2
fi

if test "$3" != ""; then
  edge_score_cut=$3
fi

# Stage [dnn, gnn, agnn,...]
ann=gnn

# Data Directories
inputdir="/home/nikin105/mlProject/data/machineLearning/XiAntiXi/training/baseline/trainingResults/test"  # input from GNN stage as in test/pred
outputdir="/home/nikin105/mlProject/data/machineLearning/XiAntiXi/track_reconstruction/baseline/track_candidates" # output of trkx_from_gnn.sh i.e. TrackCands
mkdir -p $outputdir

# original: trkx_from_gnn_v1
# uproot  : trkx_from_gnn_uproot
# cleaned : trkx_from_gnn

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "scores" \
    --edge-score-cut $edge_score_cut \
    --epsilon $epsilon \
    --min-samples 2
