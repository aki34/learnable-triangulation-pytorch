import os
import json
import re
import cv2
import numpy as np
from tqdm import trange

CAMNUM = 8
scenario_list = ['I_crane', 'I_jumping', 'I_march', 'I_squat', 'D_bouncing', 'D_handstand', 'D_march', 'D_squat', 'T_swing', 'T_samba']

with open('imgnum_dict.json', 'r') as f:
    imgnum_dict = json.load(f)

for scenario in scenario_list:
    calib_dir = scenario+'/calibration/'
    intrinsic_out_dir = calib_dir+'intrinsic/'
    extrinsic_out_dir = calib_dir+'extrinsic/'
    os.makedirs(intrinsic_out_dir, exist_ok=True)
    os.makedirs(extrinsic_out_dir, exist_ok=True)

    for cam in range(1, CAMNUM+1):
        print(f'{scenario} cam{cam}')
        cammat_file = calib_dir+f'Camera{cam}.rad'
        if scenario in ['I_march', 'I_squat', 'D_march', 'D_squat']:
            projmat_file = calib_dir+f'camera{cam}.Pmat.cal'
        else:
            projmat_file = calib_dir+f'Camera{cam}.Pmat.cal'

        with open(cammat_file, 'r') as f:
            cammat = []
            for _ in range(9):
                line = f.readline()
                # param = re.sub('\\D', '', line)
                param = float(line[6:-2])
                cammat.append(param)
        cammat = np.array(cammat).reshape(3, 3)
        np.savetxt(intrinsic_out_dir+f'cammat{cam}.txt', cammat)

        focal_x, focal_y = cammat[0, 0], cammat[1, 1]
        cx, cy = cammat[0, 2], cammat[1, 2]
        width, height = 1600, 1200
        aspect = focal_y/focal_x
        camparam = np.array([focal_x, 0, 1, aspect, cx, cy, width, height])
        np.savetxt(intrinsic_out_dir+f'camparam{cam:02d}.txt', camparam)

        projmat = np.loadtxt(projmat_file)
        rt = np.dot(np.linalg.inv(cammat), projmat)

        extrinsic = np.concatenate((rt, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
        np.savetxt(extrinsic_out_dir+f'RT{cam}.txt', extrinsic, fmt='%.5f')
        np.savetxt(extrinsic_out_dir+f'RT{cam}_list.txt', extrinsic.reshape(-1, 1), fmt='%.5f')
        extrinsic = np.linalg.inv(extrinsic)
        np.savetxt(extrinsic_out_dir+f'RTinv{cam}.txt', extrinsic, fmt='%.5f')
        np.savetxt(extrinsic_out_dir+f'RTinv{cam}_list.txt', extrinsic.reshape(-1, 1), fmt='%.5f')
