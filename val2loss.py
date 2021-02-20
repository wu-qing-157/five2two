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
from tqdm import tqdm

######################################
MODELS_ROOT = '/Disk1/guyi/obj_models'
TWO_FINGERS_ROOT = '/Disk1/guyi/two_finger'
HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'
RESULT_ROOT = 'model_result/random_split'
#######################################

with open(os.path.join(RESULT_ROOT, 'val_idx.json')) as f:
    vals = json.loads(f.read())
joints = []
rotates = []
transes = []
for d, h, o, g in tqdm(vals):
    if 'GSF' not in d: continue
    joints.append(np.load(os.path.join(HO3D_ROOT, d, f'joints_{h:04d}.npy')))
    pose = np.load(os.path.join(HO3D_ROOT, d, f'obj_pose_{h:04d}.npy'))
    pose[:3, 3] -= joints[-1].mean(axis=0)
    rotate, trans = rotate2angle.grasp_raw2learn(*utils.two_finger_transpose(TWO_FINGERS_ROOT, o, g), pose)
    rotates.append(rotate)
    transes.append(trans)
joints = np.array(joints)
rotates = np.array(rotates)
transes = np.array(transes)
print(model.eval_loss(os.path.join(RESULT_ROOT, 'model.pkl'), os.path.join(RESULT_ROOT, 'norm.npz'), joints, rotates, transes))