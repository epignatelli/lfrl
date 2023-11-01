#!/bin/bash -l
#$ -l h_rt=72:10:0
#$ -l mem=512G
#$ -l tmpfs=15G
#$ -N cluster_request_test
#$ -wd /home/uceeepi/Scratch/workspace

# Your work should be done in $TMPDIR
set REPO_DIR=$HOME/repos/lfrl

conda env create -f $REPO_DIR/environment.yml
conda activate lfrl
python $REPO_DIR/ucl_cluster_test/cluster_test.py
