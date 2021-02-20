__author__ = 'fulingyue'

from graspnetAPI import GraspNet
from graspnetAPI import GraspNetEval
from tqdm import tqdm
import os
import numpy as np
import open3d as o3d
import utils

if __name__ == '__main__':

    ####################################################################
    graspnet_root = '/ssd1/graspnet'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################
    output_root = '/home/lingyue/two_finger'

    obj_Ids = [0,1,3,4,5,7,8,9]

    camera = 'kinect'
    th = 0.5
    num_grasp = 10000

    graspnet = GraspNetEval(graspnet_root, camera, 'all')

    # graspnet.showObjGrasp(objIds=obj_Ids, numGrasp= 30, show=False)
    # if not os.path.exists(output_root):
    #     os.makedir(output_root)

    for obj_id in obj_Ids:
        plyfile = os.path.join(graspnet_root, 'models', '%03d' % obj_id, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)

        num_views, num_angles, num_depths = 300, 12, 4
        # 随机生成观察角度
        views = utils.generate_views(num_views)


        # vis = o3d.visualization.Visualizer()
        # vis.create_window(width=1280, height=720)
        # ctr = vis.get_view_control()


        param = utils.get_camera_parameters(camera='kinect')

        cam_pos = np.load(os.path.join(graspnet_root, 'scenes', 'scene_0000', 'kinect', 'cam0_wrt_table.npy'))
        param.extrinsic = np.linalg.inv(cam_pos).tolist()

        # 读取抓取姿势
        sampled_points, offsets, scores, _ = utils.get_model_grasps(
            '%s/grasp_label/%03d_labels.npz' % (graspnet_root, obj_id))

        # print(offsets.shape)

        # 随机选点
        cnt = 0
        point_inds = np.arange(sampled_points.shape[0])

        # 每个物体有多少抓取姿势
        # print(sampled_points.shape[0])
        np.random.shuffle(point_inds)
        grippers = []
        target_points = []
        grip_rotates = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind] # 物体上的抓取点 && 抓取的center点
            offset = offsets[point_ind]
            score = scores[point_ind] # confidence

            view_inds = np.arange(300)
            np.random.shuffle(view_inds)

            flag = False
            for v in view_inds:
                if flag: break
                view = views[v]
                angle_inds = np.arange(12)
                np.random.shuffle(angle_inds)
                for a in angle_inds:
                    if flag: break
                    depth_inds = np.arange(4)
                    np.random.shuffle(depth_inds)
                    for d in depth_inds:
                        if flag: break
                        angle, depth, width = offset[v, a, d]
                        if score[v, a, d] > th or score[v, a, d] < 0:
                            continue
                        R = utils.viewpoint_params_to_matrix(-view, angle)
                        t = target_point
                        target_points.append(target_point)
                        grippers.append([depth,width,angle])
                        grip_rotates.append(R)

                        # print(f'angle is :{angle}')
                        # print(f'depth shape is :{depth.shape}')
                        # print(f'width shape is : {width.shape}')
                        # gripper = utils.plot_gripper_pro_max(t, R, width, depth, 1.1 - score[v, a, d])
                        # grippers.append(gripper)
                        flag = True
            if flag:
                cnt += 1
                if cnt % 100 == 0:
                    print(f'{cnt} grasps have been loaded...')
            if cnt == num_grasp:
                break

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        # print(np.asarray(grippers).shape)
        np.save(os.path.join(output_root, 'obj_{}_targetpoints.npy'.format(obj_id)),target_points)
        np.save(os.path.join(output_root, 'obj_{}_grippers.npy'.format(obj_id)), grippers)
        np.save(os.path.join(output_root, 'obj_{}_rotates.npy'.format(obj_id)), grip_rotates)

        # vis.add_geometry(model)
        # for gripper in grippers:
        #     vis.add_geometry(gripper)
        # ctr.convert_from_pinhole_camera_parameters(param)
        # vis.poll_events()
        # filename = os.path.join(output_root, 'object_{}_grasp.png'.format(obj_id))
        # vis.capture_screen_image(filename, do_render=True)


