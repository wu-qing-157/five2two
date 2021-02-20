import utils
import mapping
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manopth import manolayer, demo
import torch
import json

######################################
MODELS_ROOT = '/Disk1/guyi/obj_models'
TWO_FINGERS_ROOT = '/Disk1/guyi/two_finger'
HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'
# CATE = 'SMu41'
# IDX = 320
CATE, IDX = "ShSu10", 892
#######################################

layer = manolayer.ManoLayer(
    joint_rot_mode='axisang',
    use_pca=False,
    mano_root='assets/mano',
    center_idx=None,
    flat_hand_mean=True,
)
cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

with open(os.path.join(HO3D_ROOT, CATE, 'mapping.json')) as f:  
    mapped = json.loads(f.read())[f'{IDX}']
    print(f"grasp_index: {mapped['grasp_index']}    five_dis: {mapped['five_dis']}    mapping_dis: {mapped['mapping_dis']}")
    obj_index = mapped['obj_index']
    grasp_index = mapped['grasp_index']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

five_finger = np.load(os.path.join(HO3D_ROOT, CATE, f'hand_info_{IDX:04d}.npz'))
five_verts, five_joints = layer(torch.Tensor(five_finger['pose']).unsqueeze(0), torch.Tensor(five_finger['shape']).unsqueeze(0))
five_verts = five_verts[0].numpy() / 1000 + five_finger['trans']
five_verts = cam_extr[:3, :3].dot(five_verts.transpose()).transpose()
five_verts = torch.Tensor(five_verts).unsqueeze(0)
demo.display_hand({'verts': five_verts, 'joints': five_joints}, mano_faces=layer.th_faces, ax=ax, show=False)

pose = np.load(os.path.join(HO3D_ROOT, CATE, f'obj_pose_{IDX:04d}.npy'))

# depth, width, rotate, target_point = utils.two_finger_transpose(TWO_FINGERS_ROOT, obj_index, grasp_index)
# hand_vertices = utils.two_finger_vis_graspnet_vision(depth, width, rotate, target_point, pose)
# for group in (hand_vertices[0:8], hand_vertices[8:16], hand_vertices[16:24], hand_vertices[24:32]):
#     for i in range(8):
#         for j in range(i + 1, 8):
#             ax.plot(group[[i, j], 0], group[[i, j], 1], group[[i, j], 2], c='blue')
#ax.scatter(hand_vertices[[25, 27, 29, 31], 0], hand_vertices[[25, 27, 29, 31], 1], hand_vertices[[25, 27, 29, 31], 2], c='purple')

object_cloud = utils.models_transpose(MODELS_ROOT, obj_index, pose)
ax.scatter(object_cloud[:, 0], object_cloud[:, 1], object_cloud[:, 2], s=0.0001, c='cyan')

ffive = mapping.single_pair((CATE, IDX))
for five in ffive:
    ax.plot(five[:, 0], five[:, 1], five[:, 2], c='red')

plt.show()
