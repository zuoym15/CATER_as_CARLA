import numpy as np
import utils_3d

import os

def find_all_files(dir, extension='.json'):
    print(dir)
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(extension):
            # file_list.append(os.path.join(dir, file))
            file_list.append(file.split('.')[0]) # discard extension, should return "CLEVR_new_000000"

    return file_list

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


mod = 'aa' # dump traj

MODE = 'traj' # one view
# MODE = 'multiview' # one frame

NUM_CAMERAS = 6
CAMERA_NAMES = ['Camera_'+str(i) for i in range(1, NUM_CAMERAS+1)] 
TOTAL_NUM_FRAME = 100
MAX_OBJECTS = 10

dump_base_dir = '../npzs'
dump_folder_dir = os.path.join(dump_base_dir, MODE + '_' + mod + '_' + 's' + str(TOTAL_NUM_FRAME))

mkdir_p(dump_base_dir)
mkdir_p(dump_folder_dir)

output_dir = '../output' # render results
output_image_dir = os.path.join(output_dir, 'images')
output_scene_dir = os.path.join(output_dir, 'scenes')
output_blend_dir = os.path.join(output_dir, 'blend')
output_camera_info_dir = os.path.join(output_dir, 'camera_info')
output_object_info_dir = os.path.join(output_dir, 'object_info')

# we want to save the following things:
# pix_T_cams: S x 4 x 4
# rgb_camXs: S x H x W x 3
# xyz_camXs: S x N x 3
# world_T_camXs: S x 4 x 4
# lrt_traj_world: S x N x 19, 19 is lens + RT, N is MAX_number of objects
# scorelist: S x N, 1 if the lrt_traj_world is valid (i.e. has an object), else 0
# world_T_camR: S x 4 x 4, we define camR as concentric as world, but with y axis pointing upward (definition in pytorch_disco)
# X: right
# Y: down
# Z: forward

# find all files
video_list = find_all_files(output_scene_dir)
for video_name in video_list:
    camera_info_file = os.path.join(output_camera_info_dir, video_name+'.json')
    bbox_info_file = os.path.join(output_object_info_dir, video_name+'.json')

    if MODE == 'traj':
        # S is the sequence dim
        for camera_name in CAMERA_NAMES: # save a different file for each view
            # initialize
            pix_T_camXs_list = []
            rgb_camXs_list = []
            xyz_camXs_list = []
            world_T_camXs_list = []
            lrt_traj_world_list = []
            scorelist_list = []
            world_T_camR_list = []

            # load frame irrelevant info
            pix_T_camXs, camXs_T_world = utils_3d.load_camera_info(camera_info_file, camera_name)
            world_T_camXs = np.linalg.inv(camXs_T_world)

            # loop over the frames
            for frame_id in range(TOTAL_NUM_FRAME + 1): # 0 to TOTAL_NUM_FRAME
                # load image
                frame_name = str(frame_id).zfill(4)+camera_name[-2:] # e.g. 0000_L
                depth_img_name = os.path.join(output_image_dir, video_name, 'Depth%s.exr' % frame_name)
                rgb_img_name = os.path.join(output_image_dir, video_name, 'RGB%s.jpg' % frame_name)

                rgb_camXs = utils_3d.load_image(rgb_img_name, normalize=False) # keep value in [0, 255]
                depth = utils_3d.load_depth(depth_img_name)
                xyz_camXs_raw = utils_3d.depth2pointcloud(depth, pix_T_camXs) # 4 x N
                xyz_camXs = np.transpose(xyz_camXs_raw)[:, 0:3]

                # load bbox info
                bbox_info = utils_3d.load_bbox_info(bbox_info_file, frame_id)
                num_objects, lrtlist = utils_3d.preprocess_bbox_info(bbox_info)

                lrt_traj_world = np.zeros((MAX_OBJECTS, 19))
                scorelist = np.zeros(MAX_OBJECTS)
                lrt_traj_world[:num_objects, :] = lrtlist
                scorelist[:num_objects] = 1.0

                # define camR here
                world_T_camR = np.array(
                    [[0, 0, 1, 0],
                     [-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]], dtype=float)

                pix_T_camXs_list.append(pix_T_camXs)
                rgb_camXs_list.append(rgb_camXs)
                xyz_camXs_list.append(xyz_camXs)
                world_T_camXs_list.append(world_T_camXs)
                lrt_traj_world_list.append(lrt_traj_world)
                scorelist_list.append(scorelist)
                world_T_camR_list.append(world_T_camR)

            pix_T_camXs_list = np.array(pix_T_camXs_list, dtype=np.float32)
            rgb_camXs_list = np.array(rgb_camXs_list, dtype=np.float32)
            xyz_camXs_list = np.array(xyz_camXs_list, dtype=np.float32)
            world_T_camXs_list = np.array(world_T_camXs_list, dtype=np.float32)
            lrt_traj_world_list = np.array(lrt_traj_world_list, dtype=np.float32)
            scorelist_list = np.array(scorelist_list, dtype=np.float32)
            world_T_camR_list = np.array(world_T_camR_list, dtype=np.float32)

            dict_to_save = {
                'pix_T_camXs': pix_T_camXs_list,
                'rgb_camXs': rgb_camXs_list,
                'xyz_camXs': xyz_camXs_list,
                'world_T_camXs': world_T_camXs_list,
                'lrt_traj_world': lrt_traj_world_list,
                'scorelist': scorelist_list,
                'world_T_camR': world_T_camR_list
            }

            dump_file_name = utils_3d.generate_seq_dump_file_name(TOTAL_NUM_FRAME, video_name, camera_name, start_frame=0)
            dump_file_name = os.path.join(dump_folder_dir, dump_file_name)

            np.savez(dump_file_name, **dict_to_save)

    elif MODE == 'multiview':
        pass
    else:
        assert False, 'unknown mode'