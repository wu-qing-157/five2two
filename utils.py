import numpy as np
import open3d as o3d
import os
import cv2

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views


def get_camera_parameters(camera='kinect'):
    '''
    author: Minghao Gou

    **Input:**

    - camera: string of type of camera: 'kinect' or 'realsense'

    **Output:**

    - open3d.camera.PinholeCameraParameters
    '''

    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = np.eye(4, dtype=np.float64)
    # param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    return param

def get_model_grasps(datapath):
    # print(1)
    label = np.load(datapath)
    # print(2)
    points = label['points']
    offsets = label['offsets']
    scores = label['scores']
    collision = label['collision']
    # print(points, offsets, scores)
    return points, offsets, scores, collision

def viewpoint_params_to_matrix(towards, angle):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1):
    '''
        center: target point
        R: rotation matrix
    '''
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    color_r = score  # red for high score
    color_b = 1 - score  # blue for low score
    color_g = 0
    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper


def two_finger_vis_graspnet_vision_no_pose(depth, width, R, center):
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02


    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)

    vertices = np.dot(R, vertices.T).T + center
    return vertices

def two_finger_vis_apply_pose(vertices, pose_matrix):
    old_shape = vertices.shape
    vertices = vertices.reshape((-1, 3))
    vertices = np.column_stack((vertices,np.ones((vertices.shape[0],1))))
    vertices = np.matmul(pose_matrix, vertices.T).T
    vertices = np.delete(vertices, 3, axis=1)
    vertices = vertices.reshape(old_shape)
    return vertices

def two_finger_vis_graspnet_vision(depth, width, R, center, pose_matrix):
    '''
    :return: (32,3), vertexes of grasping
    '''
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02


    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)

    vertices = np.dot(R, vertices.T).T + center
    vertices = np.column_stack((vertices,np.ones((vertices.shape[0],1))))
    vertices = np.matmul(pose_matrix, vertices.T).T
    vertices = np.delete(vertices, 3, axis=1)
    return vertices


def models_transpose(models_root, obj_id, pose):
    '''

    models_root: the root of models
    obj_id: the obj index
    pose: 6d pose of the object
    return: a point cloud numpy array (n*3)
    '''
    model = np.load(os.path.join(models_root, 'obj_%03d.npy' %obj_id))  #(n,3)
    model = np.column_stack((model,np.ones((model.shape[0],1)))) # (n, 4)
    new_model_point_cloud = np.matmul(pose, model.T).T
    new_model_point_cloud = np.delete(new_model_point_cloud, 3 ,axis=1)
    return new_model_point_cloud


def two_finger_transpose_all(two_finger_root, obj_id):
    grippers = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_grippers.npy'))
    rotates = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_rotates.npy'))
    target_points = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_targetpoints.npy'))
    return grippers[:, 0], grippers[:, 1], rotates, target_points


def two_finger_transpose(two_finger_root, obj_id, grasp_index):
    grippers = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_grippers.npy'))
    depth = grippers[grasp_index][0]
    width = grippers[grasp_index][1]
    rotates = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_rotates.npy'))
    rotate = rotates[grasp_index]
    target_points = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_targetpoints.npy'))
    target_point = target_points[grasp_index]
    return depth, width, rotate, target_point


def two_finger_gripper_num(two_finger_root, obj_id):
    grippers = np.load(os.path.join(two_finger_root, f'obj_{obj_id}_grippers.npy'))
    return grippers.shape[0]

def visObjGrasp(dataset_root, obj_idx, num_grasp=10, th=0.5, save_folder='save_fig', show=False):
    plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
    model = o3d.io.read_point_cloud(plyfile)

    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1280, height = 720)
    ctr = vis.get_view_control()
    param = get_camera_parameters(camera='kinect')

    cam_pos = np.load(os.path.join(dataset_root, 'scenes', 'scene_0000', 'kinect', 'cam0_wrt_table.npy'))
    param.extrinsic = np.linalg.inv(cam_pos).tolist()
    sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz'%(dataset_root, obj_idx))
    print(3)
    cnt = 0
    point_inds = np.arange(sampled_points.shape[0])
    np.random.shuffle(point_inds)
    grippers = []

    for point_ind in point_inds:
        target_point = sampled_points[point_ind]
        offset = offsets[point_ind]
        score = scores[point_ind]
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
                    R = viewpoint_params_to_matrix(-view, angle)
                    t = target_point
                    gripper = plot_gripper_pro_max(t, R, width, depth, 1.1-score[v, a, d])
                    grippers.append(gripper)
                    flag = True
        if flag:
            cnt += 1
        if cnt == num_grasp:
            break

    vis.add_geometry(model)
    for gripper in grippers:
        vis.add_geometry(gripper)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    filename = os.path.join(save_folder, 'object_{}_grasp.png'.format(obj_idx))
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([model, *grippers])


def draw_line(p1, p2, a=3e-3, color=np.array((1.0, 0.0, 0.0))):
    '''
    **Input:**

    - p1: np.array of shape(3), the first point.

    - p2: np.array of shape(3), the second point.

    - a: float of the length of the square of the bottom face.

    **Output**

    - open3d.geometry.TriangleMesh of the line
    '''
    d = np.linalg.norm(p1 - p2)
    v1 = (p2 - p1) / d
    v2 = np.cross(np.array((0, 0, 1.0)), v1)
    v3 = np.cross(v1, v2)
    R = np.stack((v3, v2, v1)).astype(np.float64).T

    box = o3d.geometry.TriangleMesh.create_box(width=a, height=a, depth=d)
    box = box.translate(np.array((-a / 2, -a / 2, 0)))
    box = box.rotate(R, (0, 0, 0))
    box = box.translate(p1)
    box.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
    return box


def vis_training_result(depth, width, R, center):
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    # rotate, _ = cv2.Rodrigues(np.asarray(R))
    rotate = np.asarray(R).reshape(3, 3)
    # print(rotate)

    gripper = np.zeros((10,3))
    #left_points
    gripper[0,0] = depth
    gripper[0,2] = -height/2
    gripper[1,0] = depth
    gripper[1,2] = height/2
    gripper[2,0] = -depth_base
    gripper[2,2] = height/2
    gripper[3,0] = -depth_base
    gripper[3,2] = -height/2

    gripper[0:4, 1] = -width / 2
    #right_points
    gripper[4,0] = -depth_base
    gripper[4,2] = height/2
    gripper[5,0] = -depth_base
    gripper[5,2] = -height/2
    gripper[6, 0] = depth
    gripper[6, 2] = -height / 2
    gripper[7, 0] = depth
    gripper[7, 2] = height / 2

    gripper[4:8, 1] = width/2

    gripper[8,0] = -depth_base
    gripper[9,0] = -depth_base - tail_length

    gripper = np.matmul(rotate, gripper.T).T
    gripper = gripper + center

    # print(gripper)
    # left: gripper[0:4], right: gripper[4:8], bottom: gripper[2:6], tail: gripper[8:10]
    return gripper

# gripper = vis_training_result(0.1,0.22, [0.0,0.0,0.0], np.asarray([1,1,1]))
# lines = []
#
# lines.append(draw_line(gripper[0],gripper[1]))
# lines.append(draw_line(gripper[1],gripper[2]))
# lines.append(draw_line(gripper[2],gripper[3]))
# lines.append(draw_line(gripper[3],gripper[0]))
# lines.append(draw_line(gripper[4],gripper[5]))
# lines.append(draw_line(gripper[5],gripper[6]))
# lines.append(draw_line(gripper[6],gripper[7]))
# lines.append(draw_line(gripper[7],gripper[4]))
# lines.append(draw_line(gripper[3],gripper[5]))
# lines.append(draw_line(gripper[2],gripper[4]))
# lines.append(draw_line(gripper[8],gripper[9]))
#
# gripper = vis_training_result(0.1,0.22, [0.0,0.0,0.4], np.asarray([1,1,1]))
# lines.append(draw_line(gripper[0],gripper[1]))
# lines.append(draw_line(gripper[1],gripper[2]))
# lines.append(draw_line(gripper[2],gripper[3]))
# lines.append(draw_line(gripper[3],gripper[0]))
# lines.append(draw_line(gripper[4],gripper[5]))
# lines.append(draw_line(gripper[5],gripper[6]))
# lines.append(draw_line(gripper[6],gripper[7]))
# lines.append(draw_line(gripper[7],gripper[4]))
# lines.append(draw_line(gripper[3],gripper[5]))
# lines.append(draw_line(gripper[2],gripper[4]))
# lines.append(draw_line(gripper[8],gripper[9]))
#
# o3d.visualization.draw_geometries([*lines])
