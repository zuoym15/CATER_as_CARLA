import numpy as np
import utils_3d

import os
import json

import argparse
import itertools

def find_all_files(dir, extension='.json'):
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(extension):
            # file_list.append(os.path.join(dir, file))
            file_list.append(file.split('.')[0]) # discard extension, should return "CLEVR_new_000000"

    file_list.sort()

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
    '--mod', default='aa')
parser.add_argument(
    '--seq_len', default=8, type=int,
    help="Sequence length for the npzs. when writing multiview data this is the number of cameras")
parser.add_argument(
    '--data_format', default='traj', help="can be traj or multiview")
parser.add_argument(
    '--dump_base_dir', default='../npzs', help="where to store the npzs")
parser.add_argument(
    '--output_dir', default='../output', help="where the raw data are stored")
parser.add_argument(
    '--depth_downsample_factor', type=float, default=0.5, help="set this number smaller than 1.0 to downsample the pointcloud")
parser.add_argument(
    '--camera_to_use', nargs='+')
parser.add_argument(
    '--number_of_videos_to_use', type=int, default=-1, help="if negative, use all videos in output_dir")
parser.add_argument(
    '--pointcloud_size', type=int, default=999999999, help="number of points per scene. downsample to make npzs smaller")
parser.add_argument(
    '--total_num_cameras', type=int, default=6, help="total number of cameras in dataset")


args = parser.parse_args()

data_format = args.data_format
mod = args.mod

np.random.seed(0)

NUM_CAMERAS = args.total_num_cameras
# NUM_CAMERAS = 2
if args.camera_to_use is None: # use all cams as default
    if NUM_CAMERAS == 1:
        CAMERA_NAMES = ['Camera']
    else: 
        CAMERA_NAMES = ['Camera_'+str(i) for i in range(1, NUM_CAMERAS+1)] 
else:
    CAMERA_NAMES = args.camera_to_use
# TOTAL_NUM_FRAME = 300
TOTAL_NUM_FRAME = 50
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
if args.number_of_videos_to_use > 0:
    video_list = video_list[:args.number_of_videos_to_use]

for video_id in range(len(video_list)):
    video_name = video_list[video_id]
    camera_info_file = os.path.join(output_camera_info_dir, video_name+'.json')
    bbox_info_file = os.path.join(output_object_info_dir, video_name+'.json')
    with open(bbox_info_file) as json_file:
        bbox_info_meta = json.load(json_file)
    with open(camera_info_file) as json_file:
        camera_info_meta = json.load(json_file)
    

    print('processing video {}/{}, video_name: {}'.format(video_id+1, len(video_list), video_name))

    if data_format == 'traj':
        # S is the sequence dim
        for camera_name in CAMERA_NAMES: # save a different file for each view
            print('processing view: {}'.format(camera_name))
            # load frame irrelevant info
            timestep_intervals = get_intervals(TOTAL_NUM_FRAME, args.seq_len)

            for interval in timestep_intervals:
                print('processing interval: ', interval)
                dump_file_name = utils_3d.generate_seq_dump_file_name(args.seq_len, video_name, [camera_name, ], start_frame=interval[0])
                dump_file_name = os.path.join(dump_folder_dir, dump_file_name)

                if os.path.isfile(dump_file_name):
                    print('file already exist. skip this')
                    continue
                # initialize
                pix_T_camXs_list = []
                rgb_camXs_list = []
                xyz_camXs_list = []
                world_T_camXs_list = []
                lrt_traj_world_list = []
                scorelist_list = []
                world_T_camR_list = []
                shapelist_list = []

                # loop over the frames
                for frame_id in interval: # 0 to S
                    # load camera info
                    pix_T_camXs, camXs_T_world = utils_3d.load_camera_info(camera_info_meta, camera_name, frame_id)
                    world_T_camXs = np.linalg.inv(camXs_T_world)
                    # load image
                    frame_name = str(frame_id).zfill(4)+camera_name[-2:] if len(camera_name)==8 else str(frame_id).zfill(4)# e.g. 0000_L or 0000
                    depth_img_name = os.path.join(output_image_dir, video_name, 'Depth%s.exr' % frame_name)
                    rgb_img_name = os.path.join(output_image_dir, video_name, 'RGB%s.jpg' % frame_name)

                    rgb_camXs = utils_3d.load_image(rgb_img_name, normalize=False) # keep value in [0, 255]
                    depth = utils_3d.load_depth(depth_img_name)
                    xyz_camXs_raw = utils_3d.depth2pointcloud(depth, pix_T_camXs, downsample_factor=args.depth_downsample_factor) # 4 x N
                    xyz_camXs = np.transpose(xyz_camXs_raw)[:, 0:3] # N x 3

                    if xyz_camXs.shape[0] > args.pointcloud_size:
                        random_id = np.random.choice(range(xyz_camXs.shape[0]), size=args.pointcloud_size, replace=False)
                        xyz_camXs = xyz_camXs[random_id, :]

                    # load bbox info
                    bbox_info = utils_3d.load_bbox_info(bbox_info_meta, frame_id)
                    num_objects, lrtlist, shapes = utils_3d.preprocess_bbox_info(bbox_info)

                    lrt_traj_world = np.zeros((MAX_OBJECTS, 19))
                    # initialize as a unit cube at center
                    lrt_traj_world[:, 0:3] = 1.0 # set length=1
                    lrt_traj_world[:, 3:] = (np.eye(4).reshape(-1)[None, :]).repeat(MAX_OBJECTS, axis=0) # set RT to indentity transformation
                    lrt_traj_world[:num_objects, :] = lrtlist

                    scorelist = np.zeros(MAX_OBJECTS)
                    scorelist[:num_objects] = 1.0

                    shapelist = np.zeros(MAX_OBJECTS)
                    shapelist.fill(-1)
                    shapelist[:num_objects] = shapes

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
                    shapelist_list.append(shapelist)

                pix_T_camXs_list = np.array(pix_T_camXs_list, dtype=np.float32)
                rgb_camXs_list = np.array(rgb_camXs_list, dtype=np.uint8)
                xyz_camXs_list = np.array(xyz_camXs_list, dtype=np.float32)
                world_T_camXs_list = np.array(world_T_camXs_list, dtype=np.float32)
                lrt_traj_world_list = np.array(lrt_traj_world_list, dtype=np.float32)
                scorelist_list = np.array(scorelist_list, dtype=np.float32)
                world_T_camR_list = np.array(world_T_camR_list, dtype=np.float32)
                shapelist_list = np.array(shapelist_list, dtype=np.int64)

                dict_to_save = {
                    'pix_T_camXs': pix_T_camXs_list,
                    'rgb_camXs': rgb_camXs_list,
                    'xyz_camXs': xyz_camXs_list,
                    'world_T_camXs': world_T_camXs_list,
                    'lrt_traj_world': lrt_traj_world_list,
                    'scorelist': scorelist_list,
                    'world_T_camR': world_T_camR_list,
                    'shapelist': shapelist_list,
                }

                np.savez(dump_file_name, **dict_to_save)

    elif data_format == 'multiview':
        assert args.seq_len == NUM_CAMERAS # shall not write different combos. we can do random selection in dataloader
        for frame_id in range(TOTAL_NUM_FRAME):
            print('processing frame: {}'.format(frame_id))
            # load some camera irrlevant information
            bbox_info = utils_3d.load_bbox_info(bbox_info_meta, frame_id)
            num_objects, lrtlist, shapes = utils_3d.preprocess_bbox_info(bbox_info)

            lrt_traj_world = np.zeros((MAX_OBJECTS, 19))
            lrt_traj_world[:, 0:3] = 1.0 # set length=1
            lrt_traj_world[:, 3:] = (np.eye(4).reshape(-1)[None, :]).repeat(MAX_OBJECTS, axis=0) # set RT to indentity transformation
            lrt_traj_world[:num_objects, :] = lrtlist
            
            scorelist = np.zeros(MAX_OBJECTS)
            scorelist[:num_objects] = 1.0

            shapelist = np.zeros(MAX_OBJECTS)
            shapelist.fill(-1)
            shapelist[:num_objects] = shapes

            # define camR here
            world_T_camR = np.array(
                [[0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]], dtype=float)

            camera_combos = itertools.combinations(CAMERA_NAMES, args.seq_len)
            for camera_combo in camera_combos:
                print('processing view: ', camera_combo)

                dump_file_name = utils_3d.generate_seq_dump_file_name(1, video_name, camera_combo, start_frame=frame_id)
                dump_file_name = os.path.join(dump_folder_dir, dump_file_name)

                if os.path.isfile(dump_file_name):
                    print('file already exist. skip this')
                    continue

                pix_T_camXs_list = []
                rgb_camXs_list = []
                xyz_camXs_list = []
                world_T_camXs_list = []
                lrt_traj_world_list = []
                scorelist_list = []
                world_T_camR_list = []
                shapelist_list = []

                for camera_name in camera_combo:
                    pix_T_camXs, camXs_T_world = utils_3d.load_camera_info(camera_info_meta, camera_name, frame_id)
                    world_T_camXs = np.linalg.inv(camXs_T_world)

                    frame_name = str(frame_id).zfill(4)+camera_name[-2:] if len(camera_name)==8 else str(frame_id).zfill(4)# e.g. 0000_L or 0000
                    depth_img_name = os.path.join(output_image_dir, video_name, 'Depth%s.exr' % frame_name)
                    rgb_img_name = os.path.join(output_image_dir, video_name, 'RGB%s.jpg' % frame_name)

                    rgb_camXs = utils_3d.load_image(rgb_img_name, normalize=False) # keep value in [0, 255]
                    depth = utils_3d.load_depth(depth_img_name)
                    xyz_camXs_raw = utils_3d.depth2pointcloud(depth, pix_T_camXs, downsample_factor=args.depth_downsample_factor) # 4 x N
                    xyz_camXs = np.transpose(xyz_camXs_raw)[:, 0:3]

                    if xyz_camXs.shape[0] > args.pointcloud_size:
                        random_id = np.random.choice(range(xyz_camXs.shape[0]), size=args.pointcloud_size, replace=False)
                        xyz_camXs = xyz_camXs[random_id, :]

                    pix_T_camXs_list.append(pix_T_camXs)
                    rgb_camXs_list.append(rgb_camXs)
                    xyz_camXs_list.append(xyz_camXs)
                    world_T_camXs_list.append(world_T_camXs)
                    lrt_traj_world_list.append(lrt_traj_world)
                    scorelist_list.append(scorelist)
                    world_T_camR_list.append(world_T_camR)
                    shapelist_list.append(shapelist)

                pix_T_camXs_list = np.array(pix_T_camXs_list, dtype=np.float32)
                rgb_camXs_list = np.array(rgb_camXs_list, dtype=np.uint8)
                xyz_camXs_list = np.array(xyz_camXs_list, dtype=np.float32)
                world_T_camXs_list = np.array(world_T_camXs_list, dtype=np.float32)
                lrt_traj_world_list = np.array(lrt_traj_world_list, dtype=np.float32)
                scorelist_list = np.array(scorelist_list, dtype=np.float32)
                world_T_camR_list = np.array(world_T_camR_list, dtype=np.float32)
                shapelist_list = np.array(shapelist_list, dtype=np.int64)

                dict_to_save = {
                    'pix_T_camXs': pix_T_camXs_list,
                    'rgb_camXs': rgb_camXs_list,
                    'xyz_camXs': xyz_camXs_list,
                    'world_T_camXs': world_T_camXs_list,
                    'lrt_traj_world': lrt_traj_world_list,
                    'scorelist': scorelist_list,
                    'world_T_camR': world_T_camR_list,
                    'shapelist': shapelist_list
                }

                np.savez(dump_file_name, **dict_to_save)




    else:
        assert False, 'unknown mode'