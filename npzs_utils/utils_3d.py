import numpy as np
import imageio
import os
import cv2
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def load_image(img_dir, normalize=True, downsample_factor=1.0):
    img = imageio.imread(img_dir)
    if downsample_factor != 1.0:
        img = cv2.resize(img, None, fx=downsample_factor, fy=downsample_factor)
    if normalize:
        img = img / 255.0 # normalize to [0,1]
    return img

def load_depth(img_dir):
    return cv2.imread(img_dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

def meshgrid2D(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def split_intrinsics(K):
    # K is 3 x 3 or 4 x 4
    fx = K[0,0]
    fy = K[1,1]
    x0 = K[0,2]
    y0 = K[1,2]
    return fx, fy, x0, y0

def depth2pointcloud(depth, pix_T_cam, downsample_factor=1.0):
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)

    if downsample_factor != 1.0: # reshape intrinsics and depth
        fx = fx * downsample_factor
        fy = fy * downsample_factor
        x0 = x0 * downsample_factor
        y0 = y0 * downsample_factor
        depth = cv2.resize(depth, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_NEAREST)

    H, W = depth.shape
    y, x = meshgrid2D(H, W)

    # unproject
    x = (depth/fx)*(x-x0)
    y = (depth/fy)*(y-y0)

    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(depth, (-1,))

    # https://blender.stackexchange.com/questions/130970/cycles-generates-distorted-depth
    # the depth is actually distance. so do normalization here

    normalize_factor = z / np.sqrt(x**2+y**2+z**2)
    x = x * normalize_factor
    y = y * normalize_factor
    z = z * normalize_factor

    one_s = np.ones_like(x)

    xyz = np.stack((x, y, z, one_s), axis=1) # N x 3

    return np.transpose(xyz) # 3 x N

def do_lim(xs, ys, zs, rgb=None, x_lim=None, y_lim=None, z_lim=None):
    if x_lim is not None:
        x_good = np.logical_and(xs < x_lim[1], xs > x_lim[0])
    else:
        x_good = np.ones_like(xs, dtype=bool)
    if y_lim is not None:
        y_good = np.logical_and(ys < y_lim[1], ys > y_lim[0])
    else:
        y_good = np.ones_like(ys, dtype=bool)
    if z_lim is not None:
        z_good = np.logical_and(zs < z_lim[1], zs > z_lim[0])
    else:
        z_good = np.ones_like(zs, dtype=bool)
    all_good = np.logical_and(np.logical_and(x_good, y_good), z_good)
    xs = xs[all_good]
    ys = ys[all_good]
    zs = zs[all_good]
    if rgb is not None:
        rgb = rgb[all_good]
    return xs, ys, zs, rgb



def show_pointcloud(name, xyz, color=None, x_lim=None, y_lim=None, z_lim=None, top_view=True, bbox_info=None):
    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])
    
    fig = plt.figure(name)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if top_view:
        ax.view_init(elev=90, azim=0)

    xs = xyz[0,:]
    ys = xyz[1,:]
    zs = xyz[2,:]

    if color is not None:
        color = color.reshape(-1, 3)

    xs, ys, zs, color = do_lim(xs, ys, zs, color, x_lim, y_lim, z_lim)

    if color is not None:
        rgba = np.concatenate([color, 0.5*np.ones([color.shape[0], 1])], axis=1)
        ax.scatter(xs, ys, zs, s=1, c=rgba)
    else:
        ax.scatter(xs, ys, zs, s=1)

    if bbox_info is not None: # draw bbox
        edge_combos = get_edge_combos()
        for object_name, object_info in bbox_info.items():
            lenlist = object_info['lenlist']
            world_T_obj = object_info['world_T_obj']
            bbox_corners = get_corners_world(lenlist, world_T_obj) # bbox corners in world coord, 4 x 8

            for edge_combo in edge_combos:
                ax.plot(bbox_corners[0, edge_combo], bbox_corners[1, edge_combo], bbox_corners[2, edge_combo], c='orange')

    set_axes_equal(ax)

def preprocess_bbox_info(bbox_info, shape_classes):
    # lrt list is N x 19
    num_objects = 0
    lrtlist = []
    shapelist = []

    for object_name, object_info in bbox_info.items():
        num_objects += 1
        lenlist = object_info['lenlist']
        world_T_obj = object_info['world_T_obj'] # 4x4
        lrtlist_obj = np.concatenate([lenlist, world_T_obj.flatten()]) # len-19
        lrtlist.append(lrtlist_obj)

        shape_matched = False
        for shape_key, shape_id in shape_classes.items():
            if shape_key in object_name:
                shape_matched = True
                shapelist.append(shape_id)
                break

        assert shape_matched, "unknown shape: {}".format(object_name)        

    lrtlist = np.array(lrtlist)
    shapelist = np.array(shapelist)
    return num_objects, lrtlist, shapelist

def generate_gif(out_file_name, input_file_names):
    images = []
    for filename in input_file_names:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_file_name, images)

def load_camera_info(camera_info, camera_name, frame_id):
    # with open(camera_file_name) as json_file:
    #     camera_info = json.load(json_file)

    # in older version we only store the starting frame camera info
    if not isinstance(camera_info[camera_name]['pix_T_cam'], dict):
        pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'])
        cam_T_world = np.array(camera_info[camera_name]['cam_T_world'])
    else:
        pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'][str(frame_id)])
        cam_T_world = np.array(camera_info[camera_name]['cam_T_world'][str(frame_id)])
        
    return pix_T_cam, cam_T_world

def load_bbox_info(bbox_info, frame_id):
    dict_to_return = dict()
    # with open(meta_file_name) as json_file:
    #     bbox_info = json.load(json_file)

    for object_name, object_info in bbox_info.items():
        lenlist = object_info['3d_dimensions']
        world_T_obj = object_info["world_T_obj"][str(frame_id)]
        dict_to_return[object_name] = dict()
        dict_to_return[object_name]['lenlist'] = np.array(lenlist)
        dict_to_return[object_name]['world_T_obj'] = np.array(world_T_obj)

    return dict_to_return

def load_action_label(scene_info_file, object_name_list, action_classes, total_num_frames=300):
    with open(scene_info_file) as json_file:
        scene_info = json.load(json_file)

    action_label_dict = {}

    for object_name in object_name_list:
        action_label_dict[object_name] = np.zeros(total_num_frames)

    for object_name in object_name_list:
        obj_movement_list = scene_info['movements'][object_name]
        for obj_movement in obj_movement_list:
            action_name, other_obj_name, start_frame, end_frame = obj_movement
            if action_name in action_classes:
                action_label_dict[object_name][start_frame:end_frame+1] = action_classes[action_name]

            if action_name == '_contain': # this is special
                action_label_dict[other_obj_name][start_frame:end_frame+1] = action_classes['_be_contained']

    return action_label_dict

def get_corners_world(lenlist, world_T_obj):
    lx = lenlist[0]
    ly = lenlist[1]
    lz = lenlist[2]
    xs = np.array([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.]) # 1 x 8
    ys = np.array([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.])
    zs = np.array([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.])
    ones = np.ones_like(xs)

    corners_obj = np.stack((xs, ys, zs, ones), axis=0) # 4 x 8
    corners_world = np.dot(world_T_obj, corners_obj)

    return corners_world

def get_edge_combos():
    edge_combos = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    return edge_combos

def generate_seq_dump_file_name(seq_len, video_name, cam_names, start_frame=0):
    cam_name_str = '_cam'
    for cam_name in cam_names:
        cam_name_str = cam_name_str + '_' + cam_name
    return video_name + '_S' + str(seq_len) + cam_name_str + '_startframe' + str(start_frame) + '.npz' 

def split_lrtlist(lrtlist):
    N, D = lrtlist.shape
    assert D==19
    lenlist = lrtlist[:,0:3]
    rt_list = lrtlist[:, 3:]
    rt_list = np.reshape(rt_list, (N, 4, 4))
    r_list = rt_list[:, 0:3, 0:3] # N x 3 x 3
    t_list = rt_list[:, 0:3, 3] # N x 3

    return lenlist, r_list, t_list

def get_key_frame(lrtlist, thres=1e-3):
    S, N, D = lrtlist.shape
    assert D==19
    lenlist_, r_list_, t_list_ = split_lrtlist(lrtlist.reshape((S*N, 19)))
    t_list = t_list_.reshape((S, N, 3)) 
    t_list_this = t_list[:S-1]
    t_list_next = t_list[1:]

    delta_t = np.abs(t_list_this - t_list_next) # S-1 x N x 3
    is_key_frame = np.all(delta_t<thres, axis=(1,2)) # S - 1

    is_key_frame = np.append(is_key_frame, False) # back to S

    return is_key_frame

def sort_keyframe_by_conseq(is_key_frame):
    sorted_keyframe = [0, ]
    key_frames = np.where(is_key_frame)[0]
    for id, frame in enumerate(key_frames):
        if id == (len(key_frames)-1) or (key_frames[id+1] - frame) != 1:
            if frame - sorted_keyframe[-1] > 20: # suppress false positive
                sorted_keyframe.append(frame)

    return np.array(sorted_keyframe)

if __name__ == "__main__":
    
    base_dir = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//images//CLEVR_new_000000'
    camera_info_file = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//camera_info//CLEVR_new_000000.json'
    bbox_info_file = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//object_info//CLEVR_new_000000.json'
    # with open(camera_info_file) as json_file:
    #     camera_info = json.load(json_file)
    # frame_names = ['0000', ]
    frame_id = 15
    frame_names = [str(frame_id).zfill(4)]
    camera_names = ['Camera']
    bbox_info = load_bbox_info(bbox_info_file, frame_id)

    downsample_factor = 0.5

    # img_dir = []
    # for frame in range(50):
    #     img_dir.append(os.path.join(base_dir, 'RGB%s_1.jpg' % str(frame).zfill(4)))

    # generate_gif('video.gif', img_dir)
    for frame_name, camera_name in zip(frame_names, camera_names):
        pix_T_cam, cam_T_world = load_camera_info(camera_info_file, camera_name)
        # pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'])
        # cam_T_world = np.array(camera_info[camera_name]['cam_T_world'])
        
        depth_img_name = 'Depth%s.exr' % frame_name
        rgb_img_name = 'RGB%s.jpg' % frame_name

        depth_img_name = os.path.join(base_dir, depth_img_name)
        rgb_img_name = os.path.join(base_dir, rgb_img_name)

        depth = load_depth(depth_img_name)

        rgb = load_image(rgb_img_name, downsample_factor=downsample_factor)

        xyz_cam = depth2pointcloud(depth, pix_T_cam, downsample_factor=downsample_factor)
        world_T_cam = np.linalg.inv(cam_T_world)
        xyz_world = np.dot(world_T_cam, xyz_cam)

        show_pointcloud(camera_name, xyz_world, rgb, x_lim=[-5, 5], y_lim=[-5, 5], bbox_info=bbox_info)

    plt.show()

    


