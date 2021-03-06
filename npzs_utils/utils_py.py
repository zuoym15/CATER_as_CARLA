import glob, math, cv2
import numpy as np
from scipy import misc
from scipy import linalg
EPS = 1e-6

XMIN = -32 # right (neg is left)
XMAX = 32.0 # right
YMIN = -16.0 # down (neg is up)
YMAX = 16.0 # down
ZMIN = -32 # forward
ZMAX = 32 # forward

def print_stats(name, tensor):
    print('%s min = %.2f, mean = %.2f, max = %.2f' % (name, np.min(tensor), np.mean(tensor), np.max(tensor)))
    
def reduce_masked_mean(x, mask, axis=None, keepdims=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    prod = x*mask
    numer = np.sum(prod, axis=axis, keepdims=keepdims)
    denom = EPS+np.sum(mask, axis=axis, keepdims=keepdims)
    mean = numer/denom
    return mean

def reduce_masked_sum(x, mask, axis=None, keepdims=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    prod = x*mask
    numer = np.sum(prod, axis=axis, keepdims=keepdims)
    return numer

def get_nFiles(path):
    return len(glob.glob(path))

def get_file_list(path):
    return glob.glob(path)

def rotm2eul(R):
    # R is 3x3
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    if sy > 1e-6: # singular
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return x, y, z
            
def rad2deg(rad):
    return rad*180.0/np.pi

def deg2rad(deg):
    return deg/180.0*np.pi
            
def eul2rotm(rx, ry, rz):
    # copy of matlab, but order of inputs is different
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    sinz = np.sin(rz)
    siny = np.sin(ry)
    sinx = np.sin(rx)
    cosz = np.cos(rz)
    cosy = np.cos(ry)
    cosx = np.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = np.stack([r11,r12,r13],axis=-1)
    r2 = np.stack([r21,r22,r23],axis=-1)
    r3 = np.stack([r31,r32,r33],axis=-1)
    r = np.stack([r1,r2,r3],axis=-2)
    return r

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def rot2view(rx,ry,rz,x,y,z):
    # takes rot angles and 3d position as input
    # returns viewpoint angles as output
    # (all in radians)
    # it will perform strangely if z <= 0
    az = wrap2pi(ry - (-np.arctan2(z, x) - 1.5*np.pi))
    el = -wrap2pi(rx - (-np.arctan2(z, y) - 1.5*np.pi))
    th = -rz
    return az, el, th

def invAxB(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)
        
def merge_rt(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4 
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def merge_rts(r, t):
    # r is S x 3 x 3
    # t is S x 3 or maybe S x 3 x 1
    S, D1, D2 = r.shape
    assert(D1 == 3 and D2 == 3)
    t = np.reshape(t, [S, 3, 1])
    rt = np.concatenate((r,t), axis=-1)
    # rt is S x 3 x 4
    br = np.reshape(np.tile(np.array([0,0,0,1], np.float32), (S, 1)), [S, 1, 4])
    # br is S x 1 x 4
    rt = np.concatenate((rt, br), axis=1)
    # rt is S x 4 x 4
    return rt

def split_rt(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def split_rts(rt):
    N, _, _ = rt.shape
    r = rt[:, :3, :3]
    t = rt[:, :3, 3]
    r = np.reshape(r, [N, 3, 3])
    t = np.reshape(t, [N, 3, 1])
    return r, t

def split_lrtlist(lrtlist):
    # splits a BN x 19 tensor
    # into N x 3 (lens)
    # and N x 4 x 4 (rts)
    N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:3]
    ref_T_objs_list = lrtlist[:,3:].reshape(N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is N x 3
    # rtlist is N x 4 x 4
    # merges these into a N x 19 tensor
    N, D = list(lenlist.shape)
    assert(D==3)
    N2, E, F = list(rtlist.shape)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(N, 16)
    lrtlist = np.concatenate([lenlist, rtlist], axis=1)
    return lrtlist

def split_intrinsics(K):
    # K is 3 x 4 or 4 x 4
    fx = K[0,0]
    fy = K[1,1]
    x0 = K[0,2]
    y0 = K[1,2]
    return fx, fy, x0, y0
                    
def merge_intrinsics(fx, fy, x0, y0):
    # inputs are shaped []
    K = np.eye(4)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = x0
    K[1,2] = y0
    # K is shaped 4 x 4
    return K
                            
def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx *= sx
    fy *= sy
    x0 *= sx
    y0 *= sy
    return merge_intrinsics(fx, fy, x0, y0)

# def meshgrid(H, W):
#     x = np.linspace(0, W-1, W)
#     y = np.linspace(0, H-1, H)
#     xv, yv = np.meshgrid(x, y)
#     return xv, yv

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return numpy.linalg.norm(transform[0:3,3])

def radian_l1_dist(e, g):
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    l = np.abs(np.pi - np.abs(np.abs(e-g) - np.pi))
    return l

def apply_4x4(RT, XYZ):
    # RT is 4 x 4
    # XYZ is N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=1)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=1)
    # XYZ1 is N x 4

    XYZ1_t = np.transpose(XYZ1)
    # this is 4 x N

    XYZ2_t = np.dot(RT, XYZ1_t)
    # this is 4 x N
    
    XYZ2 = np.transpose(XYZ2_t)
    # this is N x 4
    
    XYZ2 = XYZ2[:,:3]
    # this is N x 3
    
    return XYZ2


def apply_4x4s(RT, XYZ):
    # RT is B x 4 x 4
    # XYZ is B x N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=2)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=2)
    # XYZ1 is B x N x 4

    XYZ1_t = np.transpose(XYZ1, (0, 2, 1))
    # this is B x 4 x N

    XYZ2_t = np.matmul(RT, XYZ1_t)
    # this is B x 4 x N

    XYZ2 = np.transpose(XYZ2_t, (0, 2, 1))
    # this is B x N x 4

    XYZ2 = XYZ2[:, :, :3]
    # this is B x N x 3

    return XYZ2


def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    N, D = list(lrtlist_X.shape)
    assert (D == 19)
    N2, E, F = list(Y_T_X.shape)
    assert (N2 == N)
    assert (E == 4 and F == 4)

    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is N x 4 x 4

    rtlist_Y = np.matmul(Y_T_X, rtlist_X)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    N, C = xyz.shape
    
    x, y, z = np.split(xyz, 3, axis=-1)

    EPS = 1e-4
    z = np.clip(z, EPS, None)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = np.concatenate([x, y], axis=-1)
    return xy

def Ref2Mem(xyz, Z, Y, X):
    # xyz is N x 3, in ref coordinates
    # transforms ref coordinates into mem coordinates
    N, C = xyz.shape
    assert(C==3)
    mem_T_ref = get_mem_T_ref(Z, Y, X)
    xyz = apply_4x4(mem_T_ref, xyz)
    return xyz

# def Mem2Ref(xyz_mem, MH, MW, MD):
#     # xyz is B x N x 3, in mem coordinates
#     # transforms mem coordinates into ref coordinates
#     B, N, C = xyz_mem.get_shape().as_list()
#     ref_T_mem = get_ref_T_mem(B, MH, MW, MD)
#     xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
#     return xyz_ref

def get_mem_T_ref(Z, Y, X):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[0,3] = -XMIN
    center_T_ref[1,3] = -YMIN
    center_T_ref[2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(X)
    VOX_SIZE_Y = (YMAX-YMIN)/float(Y)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(Z)
    
    # scaling
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0,0] = 1./VOX_SIZE_X
    mem_T_center[1,1] = 1./VOX_SIZE_Y
    mem_T_center[2,2] = 1./VOX_SIZE_Z
    
    mem_T_ref = np.dot(mem_T_center, center_T_ref)
    return mem_T_ref

def safe_inverse(a):
    r, t = split_rt(a)
    t = np.reshape(t, [3, 1])
    r_transpose = r.T
    inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    inv = np.concatenate([inv, bottom_row], 0)
    return inv

def get_ref_T_mem(Z, Y, X):
    mem_T_ref = get_mem_T_ref(X, Y, X)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = np.linalg.inv(mem_T_ref)
    return ref_T_mem

def voxelize_xyz(xyz_ref, Z, Y, X):
    # xyz_ref is N x 3
    xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    # this is N x 3
    voxels = get_occupancy(xyz_mem, Z, Y, X)
    voxels = np.reshape(voxels, [Z, Y, X, 1])
    return voxels

def get_inbounds(xyz, Z, Y, X, already_mem=False):
    # xyz is H*W x 3
    # proto is MH x MW x MD
    
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)
    
    x_valid = np.logical_and(
        np.greater_equal(xyz[:,0], -0.5), 
        np.less(xyz[:,0], float(X)-0.5))
    y_valid = np.logical_and(
        np.greater_equal(xyz[:,1], -0.5), 
        np.less(xyz[:,1], float(Y)-0.5))
    z_valid = np.logical_and(
        np.greater_equal(xyz[:,2], -0.5), 
        np.less(xyz[:,2], float(Z)-0.5))
    inbounds = np.logical_and(np.logical_and(x_valid, y_valid), z_valid)
    return inbounds

def sub2ind3D_zyx(depth, height, width, d, h, w):
    # same as sub2ind3D, but inputs in zyx order
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def sub2ind3D_yxz(height, width, depth, h, w, d):
    return h*width*depth + w*depth + d

def get_occupancy(xyz_mem, Z, Y, X):
    # xyz_mem is N x 3
    # we want to fill a voxel tensor with 1's at these inds

    inbounds = get_inbounds(xyz_mem, Z, Y, X, already_mem=True)
    inds = np.where(inbounds)

    xyz_mem = np.reshape(xyz_mem[inds], [-1, 3])
    # xyz_mem is N x 3

    # this is more accurate than a cast/floor, but runs into issues when Y==0
    xyz_mem = np.round(xyz_mem).astype(np.int32)
    x = xyz_mem[:,0]
    y = xyz_mem[:,1]
    z = xyz_mem[:,2]

    voxels = np.zeros([Z, Y, X], np.float32)
    voxels[z, y, x] = 1.0

    return voxels

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    H, W = z.shape
    
    fx = np.reshape(fx, [1,1])
    fy = np.reshape(fy, [1,1])
    x0 = np.reshape(x0, [1,1])
    y0 = np.reshape(y0, [1,1])
    
    # unproject
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    z = np.reshape(z, [-1])
    xyz = np.stack([x,y,z], axis=1)
    return xyz

def depth2pointcloud(z, pix_T_cam):
    H = z.shape[0]
    W = z.shape[1]
    y, x = meshgrid2D(H, W)
    z = np.reshape(z, [H, W])
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def meshgrid2D(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    # outputs are Y x X
    return grid_y, grid_x

def gridcloud3D(Y, X, Z):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    z_ = np.linspace(0, Z-1, Z)
    y, x, z = np.meshgrid(y_, x_, z_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    z = np.reshape(z, [-1])
    xyz = np.stack([x,y,z], axis=1).astype(np.float32)
    return xyz

def gridcloud2D(Y, X):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    y, x = np.meshgrid(y_, x_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    xy = np.stack([x,y], axis=1).astype(np.float32)
    return xyz

def normalize(im):
    im = im - np.min(im)
    im = im / np.max(im)
    return im

def wrap2pi(rad_angle):
    # rad_angle can be any shape
    # puts the angle into the range [-pi, pi]
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def convert_occ_to_height(occ, axis=1):
    Z, Y, X, C = occ.shape
    assert(C==1)
    
    height = np.linspace(float(Y), 1.0, Y)
    height = np.reshape(height, [1, Y, 1, 1])
    height = np.max(occ*height, axis=axis)/float(Y)
    height = np.reshape(height, [Z, X, C])
    return height

def create_depth_image(xy, Z, H, W):

    # turn the xy coordinates into image inds
    xy = np.round(xy)

    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (Z > 0)
    valid = (xy[:,0] < W-1) & (xy[:,1] < H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (Z[:] > 0)

    # gather these up
    xy = xy[valid]
    Z = Z[valid]
    
    inds = sub2ind(H,W,xy[:,1],xy[:,0])
    depth = np.zeros((H*W), np.float32)

    for (index, replacement) in zip(inds, Z):
        depth[index] = replacement
    depth[np.where(depth == 0.0)] = 70.0
    depth = np.reshape(depth, [H, W])

    return depth

def vis_depth(depth, maxdepth=80.0, log_vis=True):
    depth[depth<=0.0] = maxdepth
    if log_vis:
        depth = np.log(depth)
        depth = np.clip(depth, 0, np.log(maxdepth))
    else:
        depth = np.clip(depth, 0, maxdepth)
    depth = (depth*255.0).astype(np.uint8)
    return depth

def preprocess_color(x):
    return x.astype(np.float32) * 1./255 - 0.5

def convert_box_to_ref_T_obj(boxes):
    shape = boxes.shape
    boxes = boxes.reshape(-1,9)
    rots = [eul2rotm(rx,ry,rz)
            for rx,ry,rz in boxes[:,6:]]
    rots = np.stack(rots,axis=0)
    trans = boxes[:,:3]
    ref_T_objs = [merge_rt(rot,tran)
                  for rot,tran in zip(rots,trans)]
    ref_T_objs = np.stack(ref_T_objs,axis=0)
    ref_T_objs = ref_T_objs.reshape(shape[:-1]+(4,4))
    ref_T_objs = ref_T_objs.astype(np.float32)
    return ref_T_objs

def convert_boxlist_to_lrtlist(boxlist):
    N, D = list(boxlist.shape)
    assert(D==9)
    rtlist = convert_box_to_ref_T_obj(boxlist)
    lenlist = boxlist[:,3:6]
    lenlist = np.clip(lenlist, a_min=0.01, a_max=np.inf)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist

def parse_boxes(box_camRs, origin_T_camRs):
    # box_camRs is S x 9
    # origin_T_camRs is S x 4 x 4
    S, D = box_camRs.shape
    assert (D == 9)
    # in this data, the last three elements are rotation angles,
    # and these angles are wrt the world origin

    rots = deg2rad(box_camRs[:, 6:])
    roll = rots[:, 0]
    pitch = rots[:, 1]
    yaw = rots[:, 2]
    pitch_ = pitch.reshape(-1)
    yaw_ = yaw.reshape(-1)
    roll_ = roll.reshape(-1)
    rots = eul2rotm(-pitch_ - np.pi / 2.0, -roll_, yaw_ - np.pi / 2.0)
    # this is S x 3 x 3
    ts = np.zeros([S, 3], dtype=np.float32)
    rts = merge_rts(rots, ts)
    # this S x 4 x 4

    camRs_T_origin = np.linalg.inv(origin_T_camRs)
    rts = np.matmul(camRs_T_origin, rts)

    lrt_camRs = convert_boxlist_to_lrtlist(box_camRs)
    lenlist, rtlist = split_lrtlist(lrt_camRs)
    _, tlist = split_rts(rtlist.reshape(-1, 4, 4))
    rlist, _ = split_rts(rts)
    rtlist = merge_rts(rlist, tlist).reshape(S, 4, 4)
    # this is S x 4 x 4
    lrt_camRs = merge_lrtlist(lenlist, rtlist)
    return lrt_camRs

def get_clist_from_lrtlist(lrtlist):
    # lrtlist is N x 19
    N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is N x 3
    # rtlist is N x 4 x 4

    xyzlist_obj = np.zeros([N, 1, 3], dtype=np.float32)
    # xyzlist_obj is N x 8 x 3

    xyzlist_cam = apply_4x4s(rtlist, xyzlist_obj)
    return xyzlist_cam

def get_pts_inbound_lrt(xyz, lrt, mult_pad=1.0):
    N, D = list(xyz.shape)
    C, = lrt.shape
    assert(C == 19)
    assert(D == 3)
    lens, cam_T_obj = split_lrtlist(lrt.reshape(1, 19))
    lens = lens.reshape(3)
    cam_T_obj = cam_T_obj.reshape(4, 4)

    obj_T_cam = safe_inverse(cam_T_obj)
    xyz_obj = apply_4x4(obj_T_cam, xyz) # B x N x 3

    x = xyz_obj[:, 0] # N
    y = xyz_obj[:, 1]
    z = xyz_obj[:, 2]
    lx = lens[0] * mult_pad # float
    ly = lens[1] * mult_pad # float
    lz = lens[2] * mult_pad # float


    x_valid = np.logical_and((x > -lx/2.0), (x < lx/2.0))
    #print('xvalid', np.sum(x_valid))
    y_valid = np.logical_and((y > -ly/2.0), (y < ly/2.0))
    #print('yvalid', np.sum(y_valid))
    z_valid = np.logical_and((z > -lz/2.0), (z < lz/2.0))
    #print('zvalid', np.sum(z_valid))
    inbounds = np.logical_and(np.logical_and(x_valid, y_valid), z_valid) # N
    #print('inbounds', np.sum(inbounds))
    return inbounds