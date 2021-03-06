#!/bin/bash

DATA_DIR="/projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/"
OUT_DIR="/projects/katefgroup/datasets/cater/npzs/"

DATA_FORMAT="multiview"

SEQLEN=6
# INCR=1

MOD="ac" # generate multiview data

# OUT_DIR="/projects/katefgroup/datasets/carla/processed/npzs/surveil_multiview_${MOD}_s${SEQLEN}_i${INCR}"
# mkdir -p ${OUT_DIR}
# echo "writing to $OUT_DIR"

# OMP_NUM_THREADS=1
# export OMP_NUM_THREADS

python dump_npzs.py \
       --mod ${MOD} \
       --seq_len ${SEQLEN} \
       --data_format ${DATA_FORMAT} \
       --dump_base_dir ${OUT_DIR} \
       --output_dir ${DATA_DIR}