import numpy as np
import utils_3d
import utils_py

import os
import json

import argparse
import itertools

ACTION_CLASSES = {
        '_slide': 1,
        '_pick_place': 2,
        '_rotate': 3,
        '_contain': 4,
        '_be_contained': 5,
    }

SHAPE_CLASSES = {
    'Sphere':0,
    'Spl':1,
    'Cylinder':2,
    'Cube':3,
    'Cone':4,
}

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
    # if range_max == S:
    #     return([list(range(range_max)), ])
    # else:
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

# set here
args.output_dir = '/projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/'
args.depth_downsample_factor = 1.0
# args.number_of_videos_to_use = 10 # try on a small dataset
# args.number_of_videos_to_use = 1 # debug

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

CAMERA_NAMES = ['Camera_1'] # only use one cam

# TOTAL_NUM_FRAME = 100 # debug
TOTAL_NUM_FRAME = 300
# TOTAL_NUM_FRAME = 50
MAX_OBJECTS = 10

# dump_base_dir = args.dump_base_dir
# dump_folder_dir = os.path.join(dump_base_dir, data_format + '_' + mod + '_' + 's' + str(args.seq_len))

# mkdir_p(dump_base_dir)
# mkdir_p(dump_folder_dir)

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

# mod = 'train'
mod = 'val'

if mod == 'train':
    video_list = video_list[0:30] # create train set
elif mod == 'val':
    video_list = video_list[30:40] # create eval set
    # video_list = video_list[30:31] # create eval set

out = {'images': [], 'annotations': [], 'videos': [],
        'categories': [{'id': 1, 'name': 'pedestrian'}]}
image_cnt = 0
video_cnt = 0
ann_cnt = 0

for video_id in range(len(video_list)):
    video_name = video_list[video_id]
    camera_info_file = os.path.join(output_camera_info_dir, video_name+'.json')
    bbox_info_file = os.path.join(output_object_info_dir, video_name+'.json')
    scene_info_file = os.path.join(output_scene_dir, video_name+'.json')
    with open(bbox_info_file) as json_file:
        bbox_info_meta = json.load(json_file)
    with open(camera_info_file) as json_file:
        camera_info_meta = json.load(json_file)
    

    print('processing video {}/{}, video_name: {}'.format(video_id+1, len(video_list), video_name))

    # get object list
    object_name_list = []
    bbox_info = utils_3d.load_bbox_info(bbox_info_meta, 0)
    for object_name, object_info in bbox_info.items():
        object_name_list.append(object_name)

    action_label_list = []
    action_label_dict = utils_3d.load_action_label(scene_info_file, object_name_list, ACTION_CLASSES, TOTAL_NUM_FRAME)
    for object_name in object_name_list:
        action_label = action_label_dict[object_name]
        action_label_list.append(action_label)

    action_label_list = np.array(action_label_list) # N x TOTAL_NUM_FRAME

    # S is the sequence dim
    assert(len(CAMERA_NAMES) == 1) # only use one cam
    for camera_name in CAMERA_NAMES: # save a different file for each view
        print('processing view: {}'.format(camera_name))

        video_cnt += 1  # video sequence number.
        out['videos'].append({'id': video_cnt, 'file_name': video_name})
        # load frame irrelevant info
        # loop over the frames
        # for frame_id in interval: # 0 to S
        for frame_id in range(TOTAL_NUM_FRAME): # 0 to S
            if frame_id % 10 == 0:
                print('processing frame: {}'.format(frame_id))
            # define camR here
            world_T_camR = np.array(
                [[0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]], dtype=float)
            camR_T_world = np.linalg.inv(world_T_camR)

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

            height, width = rgb_camXs.shape[:2]
            image_info = {'file_name': '%s/RGB%s.jpg' % (video_name, frame_name),  # image name.
                            'id': image_cnt + frame_id + 1,  # image number in the entire training set.
                            'frame_id': frame_id + 1,  # image number in the video sequence, starting from 1.
                            'prev_image_id': image_cnt + frame_id if frame_id > 0 else -1,  # image number in the entire training set.
                            'next_image_id': image_cnt + frame_id + 2 if frame_id < TOTAL_NUM_FRAME - 1 else -1,
                            'video_id': video_cnt,
                            'height': height, 'width': width}
            out['images'].append(image_info)
            

            # load bbox info
            bbox_info = utils_3d.load_bbox_info(bbox_info_meta, frame_id)
            num_objects, lrtlist, _ = utils_3d.preprocess_bbox_info(bbox_info, SHAPE_CLASSES) # this is world!!

            lrtlist_camXs = utils_py.apply_4x4_to_lrtlist(camXs_T_world.reshape(1,4,4).repeat(num_objects, axis=0), lrtlist) # N x 19
            lrtlist_camRs = utils_py.apply_4x4_to_lrtlist(camR_T_world.reshape(1,4,4).repeat(num_objects, axis=0), lrtlist)

            lenlist, camRs_T_objs_list = utils_py.split_lrtlist(lrtlist_camRs)
            object_scales = np.sum(lenlist, axis=1) # N
            _, obj_centers = utils_py.split_rts(camRs_T_objs_list) # N x 3
            obj_centers = np.squeeze(obj_centers, axis=2)
            obj_bottom_centers = np.copy(obj_centers)
            obj_bottom_centers[:, 1] += lenlist[:, 1] / 2.0 

            is_covered = np.zeros(num_objects, dtype=np.bool)
            rel_matrix = np.zeros((num_objects, num_objects))
            for i in range(num_objects):
                for j in range(num_objects):
                    rel_matrix[i, j] = np.linalg.norm(obj_bottom_centers[i] - obj_bottom_centers[j])
                    if rel_matrix[i, j] < 0.25: # close enough
                        if object_scales[i] < object_scales[j]: # smaller obj
                            is_covered[i] = True
            
            # lrtlist is num_objects x 19
            # convert it into 2d bbox
            for n in range(num_objects):
                # discard if occluded/contained
                # if action_label_list[n, frame_id] == 5:
                #     continue
                if is_covered[n]:
                    continue

                xyz0 = xyz_camXs # N x 3
                lrt0 = lrtlist_camXs[n, :] # 19 

                
                pix_T_cam = pix_T_camXs # 4 x 4

                xyz_camR0 = utils_py.apply_4x4(np.matmul(camR_T_world, world_T_camXs), xyz0)

                mask_inbound = utils_py.get_pts_inbound_lrt(xyz0, lrt0, mult_pad=1.1)

                # filter ground
                mask_inbound = np.logical_and(mask_inbound, xyz_camR0[:,1] < -0.05)

                if np.sum(mask_inbound): # has at least 1 valid pixel
                    xyz0_inb = xyz0[mask_inbound, :]
                    # projecy back into 2d

                    xy0_inb = utils_py.apply_pix_T_cam(pix_T_cam, xyz0_inb) # N x 2
                    # take min and max (maybe convert to int?)
                    xmin, xmax, ymin, ymax = np.min(xy0_inb[:,0]), np.max(xy0_inb[:,0]), np.min(xy0_inb[:,1]), np.max(xy0_inb[:,1])

                    # TODO: check if covered

                    category_id = 1
                    ann_cnt += 1
                    ann = {'id': ann_cnt,
                        'category_id': category_id,
                        'image_id': image_cnt + frame_id + 1,
                        'track_id': n + 1,
                        'bbox': [xmin, ymin, xmax-xmin, ymax-ymin], # format is <bb_left>, <bb_top>, <bb_width>, <bb_height>
                        'conf': 1.0,
                        'iscrowd': 0,
                        'area': float((xmax-xmin) * (ymax-ymin))}
                    out['annotations'].append(ann)

        image_cnt += TOTAL_NUM_FRAME

out_path = os.path.join(args.output_dir, 'annotations', '%s.json' % mod)
json.dump(out, open(out_path, 'w'))