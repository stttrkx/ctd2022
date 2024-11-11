#!/bin/bash

# This script runs 'trkx_reco_eval.py', the trkx_from_gnn.py must
# run before this script. Better way is to use 'reco_to_eval.sh'.

# Max Events
maxevts=30000

if test "$1" != ""; then
  maxevts=$1
fi

# Matching Fractions
fraction=0.5


# Stage [dnn, gnn, agnn,...]
ann=gnn

# Data Directories
raw_inputdir="/home/nikin105/mlProject/data/machineLearning/XiAntiXi/training/baseline/trainingResults/test"  # output of GNN stage as in test/pred
rec_inputdir="/home/nikin105/mlProject/data/machineLearning/XiAntiXi/track_reconstruction/baseline/track_candidates"  # output of trkx_from_gnn.sh
outputdir="/home/nikin105/mlProject/data/machineLearning/XiAntiXi/track_reconstruction/baseline/evaluated_tracks"    # output of eval_reco_trkx.sh
outfile=$outputdir"/$fraction"                   # name prefix of output files
mkdir -p $outputdir


# Don't move above outfile, name will be messed up.
if (( $(echo "$fraction == 0.5" | bc -l) )); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi


# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --csv-path $raw_inputdir \
    --reco-track-path $rec_inputdir \
    --outname $outfile \
    --max-evts $maxevts \
    --num-workers 8 \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 5 \
    --frac-reco-matched $fraction \
    --frac-truth-matched $fraction


# Last 4 Params:

# Reconstructable: i.e. min_true_hits

# ATLAS: 7       PANDA: 6 (Only STT Functor >= 6 STT hits)

# Reconstructed: i.e. min_reco_hits

# ATLAS: 4, 5    PANDA: 5, 6

# Matching Fractions

# ATLAS: 0.5     PANDA: 0.5, 0.75, 0.95, etc.

