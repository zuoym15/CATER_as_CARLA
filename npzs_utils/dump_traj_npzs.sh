#!/bin/bash

DATA_DIR="/projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/"
# DATA_DIR="/projects/katefgroup/datasets/cater/raw/ab_s300_c2_m10_rc_2/"
# DATA_DIR="/projects/katefgroup/datasets/cater/raw/ab_s300_c2_m10_rc"
OUT_DIR="/projects/katefgroup/datasets/cater/npzs/"

DATA_FORMAT="traj"

SEQLEN=2
NUM_CAM=6
depth_downsample_factor=1.0
# INCR=1

MOD="ab" # generate traj data
MOD="ac" # downsample pointcloud to 10k per scene
MOD="ad" # seqlen 300, and save shapelist info
MOD="ae" # save action labels
MOD="ag" # multiview data
MOD="af" # multiview data part 2
MOD="ah" # do not downsample pts
MOD="ai" # again genertae data on aa raw, to get comparable results to our cvpr submission

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
       --output_dir ${DATA_DIR} \
       --total_num_cameras ${NUM_CAM} \
       --depth_downsample_factor ${depth_downsample_factor}