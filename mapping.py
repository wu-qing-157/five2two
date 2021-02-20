import utils
import numpy as np
import os
import json
from tqdm import tqdm
from threading import Thread, Lock

MODELS_ROOT = '/Disk1/guyi/obj_models'
TWO_FINGERS_ROOT = '/Disk1/guyi/two_finger'
HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'

obj_cate_mapping = [
    ('MC', 0),
    ('ShSu', 1),
    ('SS', 1),
    ('SiBF', 1),
    ('SMu', 7),
    ('SM', 3),
    ('GPMF', 4),
    ('SiBF', 5),
    ('BB', 5),
    ('MDF', 8),
    ('GSF', 9),
]

obj_mapping = {}

for dir_name in os.listdir(HO3D_ROOT):
    for cate, ind in obj_cate_mapping:
        if dir_name.startswith(cate):
            obj_mapping[dir_name] = ind
            break

two_vertices_dict = {}
two_vertices_dict_lock = Lock()

def single_pair(five_index, threshold=0.03):
    pose = np.load(os.path.join(HO3D_ROOT, five_index[0], f'obj_pose_{five_index[1]:04d}.npy'))
    obj_index = obj_mapping[five_index[0]]

    cloud = utils.models_transpose(MODELS_ROOT, obj_index, pose)
    joints = np.load(os.path.join(HO3D_ROOT, five_index[0], f'joints_{five_index[1]:04d}.npy'))
    joints = joints[2:21, :]
    joints_dis = np.square(joints[:, None, :] - cloud[None, :, :]).sum(axis=-1).min(axis=-1)**.5
    finger_joints = [
        [0, 1, 2],
        [3, 4, 5, 6],
        [7, 8, 9, 10],
        [11, 12, 13, 14],
        [15, 16, 17, 18],
    ]
    finger_touch = [joints_dis[idx].min() <= threshold for idx in finger_joints]
    if not finger_touch[0] and not finger_touch[1]:
        return [-1] * 4
    finger_center = [joints[idx].mean(axis=0) for idx in finger_joints]
    five_l = np.zeros((0, 3), dtype=float)
    five_b = np.zeros((0, 3), dtype=float)
    five_r = np.zeros((0, 3), dtype=float)
    if finger_touch[0]:
        five_l = np.concatenate((five_l, joints[1:3]), axis=0)
        print('L0')
    second_left = False
    if (not finger_touch[0] and finger_touch[1]) or (finger_touch[0] and finger_touch[1] and finger_touch[2] and
            np.square(finger_center[1] - finger_center[0]).sum(axis=-1) <=
            4 * np.square(finger_center[2] - finger_center[1]).sum(axis=-1)):
        second_left = True
        five_l = np.concatenate((five_l, joints[4:7]), axis=0)
        print('L1')
    for i in range(1, 5):
        if not finger_touch[i]:
            continue
        five_b = np.concatenate((five_b, joints[finger_joints[i][0]][None, :]), axis=0)
        if i == 1 and second_left:
            continue
        five_r = np.concatenate((five_r, joints[finger_joints[i][1:]]), axis=0)
        print(f'R{i}')
    if five_l.shape[0] == 0 or five_b.shape[0] == 0 or five_r.shape[0] == 0:
        return [-1] * 4
    five_l = five_l.mean(axis=0)
    five_b = five_b.mean(axis=0)
    five_r = five_r.mean(axis=0)
    five = np.vstack((five_l, five_b, five_r))
    
    with two_vertices_dict_lock:
        if obj_index in two_vertices_dict:
            two = two_vertices_dict[obj_index]
        else:
            depth, width, rotates, target_points = utils.two_finger_transpose_all(TWO_FINGERS_ROOT, obj_index)
            #print(depth, width)
            two = np.zeros((depth.shape[0], 3, 3))
            for i in range(depth.shape[0]):
                two_vertices = utils.two_finger_vis_graspnet_vision_no_pose(depth[i], width[i], rotates[i], target_points[i])
                two[i, 0, :] = np.mean(two_vertices[0:8, :], axis=0)
                two[i, 1, :] = np.mean(two_vertices[0:16:2, :], axis=0)
                two[i, 2, :] = np.mean(two_vertices[8:16, :], axis=0)
            two_vertices_dict[obj_index] = two
    # assert np.nan not in [*five.reshape(-1)], 'five'
    # assert np.nan not in [*two.reshape(-1)], 'two'
    two = utils.two_finger_vis_apply_pose(two, pose)

    vertices_dis = np.square(five[None, :, :] - two).sum(axis=-1)
    mapping_dis_t = vertices_dis.sum(axis=-1)
    #print(five_index, mapping_dis_t.argmin())
    return five[None, ...]
    return np.concatenate((five[None, ...], two[mapping_dis_t.argmin()][None, ...]), axis=0)
    return obj_index, int(mapping_dis_t.argmin()), float(max(joints_dis[:2].min(), joints_dis[2:].min()))**.5, (float(mapping_dis_t.min()) / 4)**.5

def single_dir(tqdm_index, dir_name):
    raw_l = os.listdir(os.path.join(HO3D_ROOT, dir_name))
    l = []
    ret = {}
    for raw in raw_l:
        if raw.startswith('joints'):
            l.append(int(raw[7:11]))
    l = sorted(l)
    for i in tqdm(l, position=tqdm_index, desc=dir_name):
        obj_index, grasp_index, five_dis, mapping_dis = single_pair((dir_name, i))
        if mapping_dis == float('nan'):
            print(dir_name, i, obj_index, grasp_index, 'nan')
        if obj_index != -1:
            ret[i] = {
                'obj_index': obj_index,
                'grasp_index': grasp_index,
                'five_dis': five_dis,
                'mapping_dis': mapping_dis,
            }
    with open(os.path.join(HO3D_ROOT, dir_name, 'mapping3.json'), 'w') as f:
        f.write(json.dumps(ret))

if __name__ == "__main__":
    single_pair(("MC1", 419))
    quit()
    raw_dirs = sorted(os.listdir(HO3D_ROOT))
    dirs = []
    for raw in raw_dirs:
        if raw in obj_mapping:
            dirs.append(raw)
    #dirs = dirs[dirs.index('SMu40'):][:1]
    #dirs = ['BB11']
    #threads = []
    #for dir_id, dir_name in enumerate(dirs[::5]):
    #    thread = Thread(target=single_dir, args=(dir_id, dir_name))
    #    thread.start()
    #    threads.append(thread)
    #for thread in threads:
    #    thread.join()
    for dir_name in tqdm(dirs, position=0):
        single_dir(1, dir_name)
