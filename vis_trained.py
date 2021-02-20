import utils
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manopth import manolayer, demo
import torch
import json
import model
from model import *
import rotate2angle

######################################
MODELS_ROOT = '/Disk1/guyi/obj_models'
TWO_FINGERS_ROOT = '/Disk1/guyi/two_finger'
HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'
MODEL_ROOT = 'model_result/random_split'
DIR, HAND, OBJ, IDX = "GSF14", 392, 9, 2241
#######################################

layer = manolayer.ManoLayer(
    joint_rot_mode='axisang',
    use_pca=False,
    mano_root='assets/mano',
    center_idx=None,
    flat_hand_mean=True,
)
cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

five_finger = np.load(os.path.join(HO3D_ROOT, DIR, f'hand_info_{HAND:04d}.npz'))
five_verts, five_joints = layer(torch.Tensor(five_finger['pose']).unsqueeze(0), torch.Tensor(five_finger['shape']).unsqueeze(0))
five_verts = five_verts[0].numpy() / 1000 + five_finger['trans']
five_verts = cam_extr[:3, :3].dot(five_verts.transpose()).transpose()
five_verts = torch.Tensor(five_verts).unsqueeze(0)
demo.display_hand({'verts': five_verts, 'joints': five_joints}, mano_faces=layer.th_faces, ax=ax, show=False)

pose = np.load(os.path.join(HO3D_ROOT, DIR, f'obj_pose_{HAND:04d}.npy'))
print('pose', pose)

joints = np.load(os.path.join(HO3D_ROOT, DIR, f'joints_{HAND:04d}.npy'))
rotate, trans = model.eval(os.path.join(MODEL_ROOT, 'model.pkl'), os.path.join(MODEL_ROOT, 'norm.npz'), joints)
trans += joints.mean(axis=0)
hand_vertices = utils.vis_training_result(0.05, 0.05, rotate, trans)
for i in [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [3, 5],
    [2, 4],
    [8, 9],
]:
    ax.plot(hand_vertices[i, 0], hand_vertices[i, 1], hand_vertices[i, 2], c='blue')
print(rotate)
print(trans)

# rotate, trans = rotate2angle.grasp_raw2learn(*utils.two_finger_transpose(TWO_FINGERS_ROOT, OBJ, IDX), pose)
# hand_vertices = utils.vis_training_result(0.05, 0.1, rotate, trans)
# for i in [
#     [0, 1],
#     [1, 2],
#     [2, 3],
#     [3, 0],
#     [4, 5],
#     [5, 6],
#     [6, 7],
#     [7, 4],
#     [3, 5],
#     [2, 4],
#     [8, 9],
# ]:
#     ax.plot(hand_vertices[i, 0], hand_vertices[i, 1], hand_vertices[i, 2], c='violet')
# print(rotate)
# print(trans)

object_cloud = utils.models_transpose(MODELS_ROOT, OBJ, pose)
ax.scatter(object_cloud[:, 0], object_cloud[:, 1], object_cloud[:, 2], s=0.0001, c='cyan')

plt.show()
