import utils
import os
import open3d as o3d
import numpy as np
import graspnetAPI

MODEL_ROOT = '/Users/fulingyue/Desktop/obj_model'
FINGER_ROOT = '/Users/fulingyue/Desktop/two_finger'
DATASET_ROOT = '/Users/fulingyue/Desktop/graspnet'
model = np.load(MODEL_ROOT + '/obj_001.npy')
color = np.load(MODEL_ROOT + '/color_001.npy')
# pose = np.random.randn(4,4)
# model = utils.models_transpose(MODEL_ROOT,1,pose)
# model_vis = o3d.geometry.PointCloud()
# model_vis.points = o3d.utility.Vector3dVector(model)
# model_vis.colors = o3d.utility.Vector3dVector(color)

# utils.visObjGrasp(DATASET_ROOT,1,10,show=True)

# depth, width, rotate, target_point = utils.two_finger_transpose(FINGER_ROOT, 1, 1)
# vis = o3d.visualization.Visualizer()
# gripper = utils.plot_gripper_pro_max(target_point,rotate,width,depth,1)
# gripper = o3d.geometry.PointCloud()
# gripper.points = \
#     o3d.utility.Vector3dVector(utils.two_finger_vis_graspnet_vision(depth,width,rotate,target_point,pose))
# o3d.visualization.draw_geometries([model_vis,gripper])

utils.visObjGrasp(DATASET_ROOT, 1, 3000, show=True)


