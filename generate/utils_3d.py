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

def load_image(img_dir):
    img = imageio.imread(img_dir) / 255.0 # normalize to [0,1]
    return img

def unp_depth(depth, cam_T_world, pix_T_cam):
    # cam_T_world and pix_T_cam are 4x4 matrix
    # depth is a H x W tensor
    pass

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



def show_pointcloud(name, xyz, color=None, x_lim=None, y_lim=None, z_lim=None, top_view=True):
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

    set_axes_equal(ax)

    

if __name__ == "__main__":
    import imageio
    images = []
    filenames = ['../output/Camera_L.png', '../output/Camera_R.png']
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('../output/bev.gif', images)
    # base_dir = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//images//CLEVR_new_000000'
    # camera_info_file = 'C://Users//zuoyi//Documents//GitHub//CATER_as_CARLA//output//camera_info//CLEVR_new_000000.json'
    # with open(camera_info_file) as json_file:
    #     camera_info = json.load(json_file)
    # # frame_names = ['0000', ]
    # frame_names = ['0000_L', '0000_R']
    # camera_names = ['Camera_L', 'Camera_R']
    # for frame_name, camera_name in zip(frame_names, camera_names):
    #     pix_T_cam = np.array(camera_info[camera_name]['pix_T_cam'])
    #     cam_T_world = np.array(camera_info[camera_name]['cam_T_world'])
        
    #     depth_img_name = 'Depth%s.exr' % frame_name
    #     rgb_img_name = 'RGB%s.jpg' % frame_name

    #     depth_img_name = os.path.join(base_dir, depth_img_name)
    #     rgb_img_name = os.path.join(base_dir, rgb_img_name)

    #     depth = cv2.imread(depth_img_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

    #     rgb = load_image(rgb_img_name)

    #     xyz_cam = depth2pointcloud(depth, pix_T_cam)
    #     world_T_cam = np.linalg.inv(cam_T_world)
    #     xyz_world = np.dot(world_T_cam, xyz_cam)

    #     show_pointcloud(camera_name, xyz_world, rgb, x_lim=[-5, 5], y_lim=[-5, 5])

    # plt.show()

    


