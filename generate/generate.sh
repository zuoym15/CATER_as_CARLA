#!/bin/bash

NUM_VIDEOS=20
NUM_FRAMES_PER_VIDEO=300 #Minimum 50

NUM_CAMERAS=2
MAX_MOTIONS=10

MOD="ab"
OUTPUT_DIR="../output/"
START_IDX=0

CUDA_VISIBLE_DEVICES="1" blender --background --python render_videos.py -- --num_images ${NUM_VIDEOS} \
                                                                            --suppress_blender_logs \
                                                                            --save_blendfiles 1 \
                                                                            --debug \
                                                                            --num_frames ${NUM_FRAMES_PER_VIDEO} \
                                                                            --num_cameras ${NUM_CAMERAS} \
                                                                            --max_motions ${MAX_MOTIONS} \
                                                                            --mod ${MOD} \
                                                                            --random_camera \
                                                                            --output_dir ${OUTPUT_DIR} \
                                                                            --start_idx ${START_IDX}