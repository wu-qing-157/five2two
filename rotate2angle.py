import numpy as np
from cv2 import Rodrigues
import json
import utils
import os
from tqdm import tqdm
from manopth import manolayer, demo
import torch

TWO_FINGERS_ROOT = '/Disk1/guyi/two_finger'
HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'

layer = manolayer.ManoLayer(
    joint_rot_mode='axisang',
    use_pca=False,
    mano_root='assets/mano',
    center_idx=None,
    flat_hand_mean=True,
)
cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def rotate2angle(R, pose):
    con = pose[:3, :3] @ R
    return con
    theta, _ = Rodrigues(con)
    return theta[:, 0]

def transposed(center, pose):
    # print(np.square(pose[:3, :3] @ center.T + pose[:3, 3]).sum())
    return pose[:3, :3] @ center.T + pose[:3, 3]

def hand_raw2learn(pose, shape, trans):
    return [*pose, *shape, *trans]

def grasp_raw2learn(depth, width, rotate, target, pose):
    return rotate2angle(rotate, pose), transposed(target, pose)

if __name__ == "__main__":
    with open('filtered.json') as f:
        j = json.loads(f.read())
    hands = []
    grasps_rotate = []
    grasps_trans = []
    idxs = []
    for d, h, o, g, _, _ in tqdm(j):
        # hand = np.load(os.path.join(HO3D_ROOT, d, f'hand_info_{h:04d}.npz'))
        # hands.append(hand_raw2learn(hand['pose'], hand['shape'], hand['trans']))
        pose = np.load(os.path.join(HO3D_ROOT, d, f'obj_pose_{h:04d}.npy'))
        joints = np.load(os.path.join(HO3D_ROOT, d, f'joints_{h:04d}.npy'))
        pose[:3, 3] -= joints.mean(axis=0)
        hands.append(joints)
        grasp_rotate, grasp_trans = grasp_raw2learn(*utils.two_finger_transpose(TWO_FINGERS_ROOT, o, g), pose)
        grasps_rotate.append(grasp_rotate)
        grasps_trans.append(grasp_trans)
        idxs.append((d, h, o, g))
    hands = np.array(hands)
    grasps_rotate = np.array(grasps_rotate)
    grasps_trans = np.array(grasps_trans)
    shuffle = np.arange(len(hands))
    # np.random.shuffle(shuffle)
    # train = shuffle[:int(len(shuffle) * 0.8)]
    # val = shuffle[int(len(shuffle) * 0.8):]
    # train = np.where(['SMu' not in d for d, _, _, _ in idxs])[0]
    # val = np.where(['SMu' in d for d, _, _, _ in idxs])[0]
    train = np.zeros((0,), dtype=int)
    val = np.zeros((0,), dtype=int)
    ids = {}
    for i, (d, _, _, _) in enumerate(idxs):
        ids.setdefault(d, []).append(i)
    for k, v in ids.items():
        train = np.concatenate((train, v[:int(len(v) * 0.8)]))
        val = np.concatenate((val, v[int(len(v) * 0.8):]))
    if not os.path.isdir('model_data'):
        os.mkdir('model_data')
    np.savez('model_data/train.npz', hand=hands[train], grasp_rotate=grasps_rotate[train], grasp_trans=grasps_trans[train])
    np.savez('model_data/val.npz', hand=hands[val], grasp_rotate=grasps_rotate[val], grasp_trans=grasps_trans[val])
    val_idxs = []
    for i in val:
        val_idxs.append(idxs[i])
    with open('model_data/val_idx.json', 'w') as f:
        f.write(json.dumps(val_idxs))
    # np.save('model_test/train.npy', con)
    # np.save('model_test/val.npy', con)
    # np.save('model_test/train.npy', con[:int(len(con) * 0.8), :])
    # np.save('model_test/val.npy', con[int(len(con) * 0.8):, :])