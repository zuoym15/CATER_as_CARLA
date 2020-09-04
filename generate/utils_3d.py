import numpy as np
import imageio
import os
import cv2
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img2depth_scale_factor = 1 / 0.05 

# pix_T_cam = np.array([[ 350.,            0.,          160.,            0.,        ],
#                       [   0.,         466,  120.,            0.,        ],
#                       [   0.,            0.,            1.,            0.,        ],
#                       [   0.,            0.,            0.,            1.,        ]])
# pix_T_cam = np.array([[ 350,            0.,            160.,            0.,        ],
#                       [   0.,            350.0,         120.,            0.,        ],
#                       [   0.,            0.,            1.,            0.,        ],
#                       [   0.,            0.,            0.,            1.,        ]])

# cam_T_world =  ([[  1.79110646e-01,  -9.83828962e-01,  -6.92129515e-06,   6.92129515e-05],
#                  [ -9.83827949e-01,  -1.79110453e-01,  -1.46486727e-03,   1.46486727e-02],
#                  [  1.43993914e-03,   2.69182754e-04,  -9.99998927e-01,   9.99998927e+00],
#                  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

# cam_T_world =  ([[  6.63005233e-01,   7.48614728e-01,   9.85652093e-09,  -1.14529834e-02],
#                  [  3.54717642e-01,  -3.14153105e-01,  -8.80615234e-01,  -3.27243269e-03],
#                  [ -6.59241557e-01,   5.83852530e-01,  -4.73832011e-01,   1.07452393e+01],
#                  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

# cam_T_world = ([[ -6.56495214e-01,   7.54330158e-01,  -6.09365181e-09,   7.43821359e-03],
#                 [  3.58130813e-01,   3.11682045e-01,  -8.80111694e-01,  -4.95519886e-03],
#                 [ -6.63894832e-01,  -5.77789128e-01,  -4.74766642e-01,   1.12568166e+01],
#                 [  0.00000000e+00,   0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])

                 

# world_T_cam = np.array([(0.1791, 0.9838, -0.0014,  0.0000),
#               (-0.9838, 0.1791, -0.0003,  0.0000),
#               ( 0.0000, 0.0015,  1.0000, 10.0000),
#               ( 0.0000, 0.0000,  0.0000,  1.0000)])

def img2depth(img):
    # img is a H x W x 3 np array, each value between 0 and 1
    if len(img.shape) == 3: # H x W x C
        img = img[:, :, 0] # 3 channels are the same
    return img2depth_scale_factor*img

def load_image(img_dir, normalize=True):
    img = imageio.imread(img_dir)
    if normalize:
        img = img / 255.0 # normalize to [0,1]
    return img

def load_depth(img_dir):
    return cv2.imread(img_dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

# def meshgrid2d(H, W):
#     x = np.linspace(0, H, H)
#     y = np.linspace(0, W, W)
#     xv, yv = np.meshgrid(x, y, indexing='ij')
#     return xv, yv

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

def depth2pointcloud(depth, pix_T_cam):
    H, W = depth.shape
    y, x = meshgrid2D(H, W)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)

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

def preprocess_bbox_info(bbox_info):
    # lrt list is N x 19
    num_objects = 0
    lrtlist = []
    for object_name, object_info in bbox_info.items():
        num_objects += 1
        lenlist = object_info['lenlist']
        world_T_obj = object_info['world_T_obj'] # 4x4
        lrtlist_obj = np.concatenate([lenlist, world_T_obj.flatten()]) # len-19
        lrtlist.append(lrtlist_obj)

    lrtlist = np.array(lrtlist)
    return num_objects, lrtlist

def generate_gif(out_file_name, input_file_names):
    images = []
    for filename in input_file_names:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_file_name, images)

def load_camera_info(camera_file_name, camera_name):
    with open(camera_file_name) as json_file:
        camera_info = json.load(json_file)

    pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'])
    cam_T_world = np.array(camera_info[camera_name]['cam_T_world'])
        
    return pix_T_cam, cam_T_world

def load_bbox_info(meta_file_name, frame_id):
    dict_to_return = dict()
    # json file
    with open(meta_file_name) as json_file:
        bbox_info = json.load(json_file)

    for object_name, object_info in bbox_info.items():
        lenlist = object_info['3d_dimensions']
        world_T_obj = object_info["world_T_obj"][str(frame_id)]
        dict_to_return[object_name] = dict()
        dict_to_return[object_name]['lenlist'] = np.array(lenlist)
        dict_to_return[object_name]['world_T_obj'] = np.array(world_T_obj)

    return dict_to_return

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

def generate_seq_dump_file_name(seq_len, video_name, cam_name, start_frame=0):
    return video_name + '_S' + str(seq_len) + '_cam' + cam_name + '_startframe' + str(start_frame) + '.npz' 

if __name__ == "__main__":
    
    base_dir = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//images//CLEVR_new_000000'
    camera_info_file = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//camera_info//CLEVR_new_000000.json'
    bbox_info_file = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//object_info//CLEVR_new_000000.json'
    # with open(camera_info_file) as json_file:
    #     camera_info = json.load(json_file)
    # frame_names = ['0000', ]
    frame_id = 15
    frame_names = [str(frame_id).zfill(4)+'_L', str(frame_id).zfill(4)+'_R']
    camera_names = ['Camera_L', 'Camera_R']
    bbox_info = load_bbox_info(bbox_info_file, frame_id)

    img_dir = []
    for frame in range(50):
        img_dir.append(os.path.join(base_dir, 'RGB%s_1.jpg' % str(frame).zfill(4)))

    generate_gif('video.gif', img_dir)
    # for frame_name, camera_name in zip(frame_names, camera_names):
    #     pix_T_cam, cam_T_world = load_camera_info(camera_info_file, camera_name)
    #     # pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'])
    #     # cam_T_world = np.array(camera_info[camera_name]['cam_T_world'])
        
    #     depth_img_name = 'Depth%s.exr' % frame_name
    #     rgb_img_name = 'RGB%s.jpg' % frame_name

    #     depth_img_name = os.path.join(base_dir, depth_img_name)
    #     rgb_img_name = os.path.join(base_dir, rgb_img_name)

    #     depth = load_depth(depth_img_name)

    #     rgb = load_image(rgb_img_name)

    #     xyz_cam = depth2pointcloud(depth, pix_T_cam)
    #     world_T_cam = np.linalg.inv(cam_T_world)
    #     xyz_world = np.dot(world_T_cam, xyz_cam)

    #     show_pointcloud(camera_name, xyz_world, rgb, x_lim=[-5, 5], y_lim=[-5, 5], bbox_info=bbox_info)

    # plt.show()

    


