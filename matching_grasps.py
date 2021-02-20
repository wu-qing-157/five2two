from graspnetAPI import GraspNet
from graspnetAPI import GraspNetEval
import numpy as np
import os
from tqdm import tqdm
import json

if __name__ == '__main__':
#####################################################
    TWO_FINGER_ROOT = '/hddisk2/guyi/two_finger'
    FIVE_FINGER_ROOT = '/hddisk2/guyi/HO3D_guyi'
    OUTPUT_ROOT = '/home/guyi/five2two/label'
#####################################################
    # consta
    OBJ_IDS = [0,1,3,4,5,7,8,9]
    FIVE_NUM = 1240
    OBJ_MAP = (
        ('MC', 0),
        ('ShSu', 1),
        ('SS', 1),
        ('SMu', 7),
        ('SM', 3),
        ('GPMF', 4),
        ('SiBF', 5),
        ('BB', 5),
        ('MDF', 8),
        ('GSF', 9),
    )
    FILE_DICT = {}
    for filename in os.listdir(FIVE_FINGER_ROOT):
        for k, v in OBJ_MAP:
            if k in filename:
                FILE_DICT.setdefault(v, []).append(filename)
                break

    for obj_id in OBJ_IDS:
        # load data
        two_sampled_points = np.load(os.path.join(
            TWO_FINGER_ROOT, 'obj_{}_targetpoints.npy'.format(obj_id)))  # (GRASP_NUM, 3)
        two_sampled_points = np.column_stack((two_sampled_points,np.ones((two_sampled_points.shape[0],1)))) # (GRASP_NUM, 4)
        # print(f'targetpoints shape is {two_sampled_points.shape}')
        # two_grasp_paras = np.load(os.path.join(
        #     TWO_FINGER_ROOT, 'obj_{}_grippers.npy'.format(obj_id)))  # (GRASP_NUM, 2)
        # print(f'grasp_paras shape is {two_grasp_paras.shape}')
        # two_rotate_matrix = np.load(os.path.join(
        #     TWO_FINGER_ROOT, 'obj_{}_rotates.npy'.format(obj_id))) #  (GRASP_NUM, 3, 3)
        # print(f'rotate_matrix shape is {two_rotate_matrix.shape}')


        for file_dict in FILE_DICT[obj_id]:
        # 寻找最好的pose
            label_index = {}
            files_ = sorted(os.listdir(os.path.join(FIVE_FINGER_ROOT, file_dict)))
            files = []
            for file in files_:
                if 'obj_pose' in file:
                    files.append(file)
            for file in tqdm(files,desc= f'loading five finger poses {obj_id}....'):
                obj_6d_pose = np.load(
                    os.path.join(FIVE_FINGER_ROOT, file_dict, file))
                # print(obj_6d_pose.shape)
                # 计算two finger的6d pose
                # print(two_sampled_points.shape)

                new_sampled_points = np.matmul(obj_6d_pose, two_sampled_points.T).T
                new_sampled_points = np.delete(new_sampled_points, 3, axis=1)
                # print(two_sampled_points)


                five_info = np.load(
                    os.path.join(FIVE_FINGER_ROOT, file_dict, file.replace('obj_pose', 'hand_info').replace('npy', 'npz')))

                # print(five_info)

                five_trans = five_info['trans']
                # print(five_trans)
                # print(five_trans.shape)
                # quit()

                # 用矩阵运算求一个loss，得到最好的下标
                five_tile = np.tile(five_trans, (new_sampled_points.shape[0], 1))
                dist_arr = np.sum(np.square(new_sampled_points - five_tile), axis=1)
                dist = np.min(dist_arr)
                index = np.argmin(dist_arr)
                # TODO:if dist > alpha
                label_index[file.replace('obj_pose_', '').replace('.npy', '')] = {
                    'index': int(index),
                    'dist': float(dist),
                }

            print(label_index)
            if not os.path.exists(OUTPUT_ROOT):
                os.makedirs(OUTPUT_ROOT)

            #np.save(os.path.join(OUTPUT_ROOT, file_dict + '_label.npy'), np.asarray(label_index))
            with open(os.path.join(OUTPUT_ROOT, file_dict + '_labels.json'), 'w') as f:
                f.write(json.dumps(label_index))



