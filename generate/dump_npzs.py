import numpy as np
import utils_3d

import os

import argparse
import itertools

def find_all_files(dir, extension='.json'):
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(extension):
            # file_list.append(os.path.join(dir, file))
            file_list.append(file.split('.')[0]) # discard extension, should return "CLEVR_new_000000"

    return file_list

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_intervals(range_max, S):
    res = []
    for s in range(S):
        res.append(range(range_max)[s::S])

    return list(zip(*res))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mod', default='aa', help="npzs version control. e.g. 'aa'")
parser.add_argument(
    '--seq_len', default=8, type=int,
    help="Sequence length for the npzs. when writing multiview data this is the number of cameras")
parser.add_argument(
    '--data_format', default='traj', help="can be traj or multiview")
parser.add_argument(
    '--dump_base_dir', default='../npzs', help="where to store the npzs")
parser.add_argument(
    '--output_dir', default='../output', help="where the raw data are stored")




args = parser.parse_args()


mod = args.mod # dump traj
data_format = args.data_format


NUM_CAMERAS = 6
# NUM_CAMERAS = 2
CAMERA_NAMES = ['Camera_'+str(i) for i in range(1, NUM_CAMERAS+1)] 
TOTAL_NUM_FRAME = 300
# TOTAL_NUM_FRAME = 50
MAX_OBJECTS = 10

dump_base_dir = args.dump_base_dir
dump_folder_dir = os.path.join(dump_base_dir, data_format + '_' + mod + '_' + 's' + str(args.seq_len))

mkdir_p(dump_base_dir)
mkdir_p(dump_folder_dir)

output_dir = args.output_dir # render results
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
for video_id in range(len(video_list)):
    video_name = video_list[video_id]
    camera_info_file = os.path.join(output_camera_info_dir, video_name+'.json')
    bbox_info_file = os.path.join(output_object_info_dir, video_name+'.json')

    print('processing video {}/{}'.format(video_id+1, len(video_list)))

    if data_format == 'traj':
        # S is the sequence dim
        for camera_name in CAMERA_NAMES: # save a different file for each view
            # load frame irrelevant info
            pix_T_camXs, camXs_T_world = utils_3d.load_camera_info(camera_info_file, camera_name)
            world_T_camXs = np.linalg.inv(camXs_T_world)

            timestep_intervals = get_intervals(TOTAL_NUM_FRAME, args.seq_len)

            for interval in timestep_intervals:
                # initialize
                pix_T_camXs_list = []
                rgb_camXs_list = []
                xyz_camXs_list = []
                world_T_camXs_list = []
                lrt_traj_world_list = []
                scorelist_list = []
                world_T_camR_list = []

                # loop over the frames
                for frame_id in interval: # 0 to S
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

                dump_file_name = utils_3d.generate_seq_dump_file_name(args.seq_len, video_name, [camera_name, ], start_frame=interval[0])
                dump_file_name = os.path.join(dump_folder_dir, dump_file_name)

                np.savez(dump_file_name, **dict_to_save)

    elif data_format == 'multiview':
        assert args.seq_len == NUM_CAMERAS # shall not write different combos. we can do random selection in dataloader
        for frame_id in range(TOTAL_NUM_FRAME):
            # load some camera irrlevant information
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

            camera_combos = itertools.combinations(CAMERA_NAMES, args.seq_len)
            for camera_combo in camera_combos:
                pix_T_camXs_list = []
                rgb_camXs_list = []
                xyz_camXs_list = []
                world_T_camXs_list = []
                lrt_traj_world_list = []
                scorelist_list = []
                world_T_camR_list = []

                for camera_name in camera_combo:
                    pix_T_camXs, camXs_T_world = utils_3d.load_camera_info(camera_info_file, camera_name)
                    world_T_camXs = np.linalg.inv(camXs_T_world)

                    frame_name = str(frame_id).zfill(4)+camera_name[-2:] # e.g. 0000_L
                    depth_img_name = os.path.join(output_image_dir, video_name, 'Depth%s.exr' % frame_name)
                    rgb_img_name = os.path.join(output_image_dir, video_name, 'RGB%s.jpg' % frame_name)

                    rgb_camXs = utils_3d.load_image(rgb_img_name, normalize=False) # keep value in [0, 255]
                    depth = utils_3d.load_depth(depth_img_name)
                    xyz_camXs_raw = utils_3d.depth2pointcloud(depth, pix_T_camXs) # 4 x N
                    xyz_camXs = np.transpose(xyz_camXs_raw)[:, 0:3]

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

                dump_file_name = utils_3d.generate_seq_dump_file_name(1, video_name, camera_combo, start_frame=frame_id)
                dump_file_name = os.path.join(dump_folder_dir, dump_file_name)

                np.savez(dump_file_name, **dict_to_save)




    else:
        assert False, 'unknown mode'