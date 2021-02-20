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
RESULT_ROOT = 'model_result/unseen_5'
#######################################

with open(os.path.join(RESULT_ROOT, 'val_idx.json')) as f:
    vals = json.loads(f.read())
poses = []
joints = []
objs = []
for d, h, o, g in tqdm(vals):
    poses.append(np.load(os.path.join(HO3D_ROOT, d, f'obj_pose_{h:04d}.npy')))
    joints.append(np.load(os.path.join(HO3D_ROOT, d, f'joints_{h:04d}.npy')))
    objs.append(o)
joints = np.array(joints)
rotates, transes = model.eval(os.path.join(RESULT_ROOT, 'model.pkl'), os.path.join(RESULT_ROOT, 'norm.npz'), joints)
transes += joints.mean(axis=1)
np.savez(os.path.join(RESULT_ROOT, 'val_eval.npz'), pose=poses, rotate=rotates, trans=transes, obj=objs)