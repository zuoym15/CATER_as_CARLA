CUDA_VISIBLE_DEVICES="0" blender --background --python render_videos.py -- --num_images 1 --suppress_blender_logs --save_blendfiles 1 --max_motions=2 --debug --num_frames 50 --use_additional_camera