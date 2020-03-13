import os
import glob
import numpy as np
from tqdm import trange

subject_list = [1, 2, 3, 4, 5]
action_list = ['acting', 'freestyle', 'rom', 'walking']
CAMNUM = 8

for subject in subject_list:
    for action in action_list:
        for idx in [1, 2, 3]:
            scenario = action+str(idx)
            if scenario not in ['acting3', 'freestyle3', 'walking2']:
                continue
            if subject in [4, 5] and scenario in ['acting1', 'acting2', 'freestyle2', 'rom1', 'rom2', 'walking1', 'walking3']:
                continue
            print(f's{subject} {scenario}')
            FRAMENUM = len(glob.glob(f'../s{subject}/gt3Dpose/{scenario}/*.txt'))
            for cam in range(1, 9):
                out_dir = os.path.join('..', f's{subject}', 'gt2Dpose', scenario, f'cam{cam}')
                os.makedirs(out_dir, exist_ok=True)

                cammat = np.loadtxt(f'../calibration/intrinsic/cammat{cam}.txt')
                camparam = np.loadtxt(f'../calibration/intrinsic/camparam_scaled{cam}.txt')
                RT = np.loadtxt(f'../calibration/extrinsic/rt{cam}.txt')
                r = RT[:3, :3]
                t = RT[:3, 3]

                f = camparam[0]
                dist = camparam[1]
                px, py = camparam[2:4]
                cx, cy = camparam[4:6]
                width, height = camparam[6:8]

                for frm in trange(FRAMENUM):
                    pose3d = np.loadtxt(f'../s{subject}/gt3Dpose/{scenario}/pose{frm:04d}.txt')
                    pose3d = np.dot(r, pose3d.T).T + t

                    pose_proj = pose3d[:, :2]/pose3d[:, 2].reshape(-1, 1)
                    pose2d = f*pose_proj/np.array([px, py]).reshape(1, 2)+np.array([cx, cy]).reshape(1, 2)

                    np.savetxt(os.path.join(out_dir, f'pose{frm:04d}.txt'), pose2d, fmt='%.5f')
