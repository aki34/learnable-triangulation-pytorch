import os
import glob
import numpy as np
from tqdm import trange

root_dir = '../'
for subject in range(1, 9):
    for seq in range(1, 3):
        print(f'S{subject} Seq{seq}')
        data_dir = os.path.join(root_dir, f'S{subject}', f'Seq{seq}')
        out_dir = os.path.join(data_dir, 'gt3Dpose')
        os.makedirs(out_dir, exist_ok=True)

        framenum = len(glob.glob(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{0}', '*.txt')))
        pose3D_list = []
        for cam in range(14):
            RT = np.loadtxt(os.path.join(data_dir, 'calibration', 'extrinsic', f'rt{cam}.txt'))
            r = RT[:3, :3]
            t = RT[:3, 3]

            # pose3D_cam = np.loadtxt(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{cam}', f'pose{frm:04d}.txt'))
            pose3D_cam = np.load(os.path.join(data_dir, 'gt3Dpose_cam', f'gtpose_cam{cam}.npy'))
            pose3D_cam = pose3D_cam.reshape(-1, 3)

            pose3D = np.dot(np.linalg.inv(r), (pose3D_cam - t).T).T
            pose3D = pose3D.reshape(-1, 28, 3)
            pose3D_list.append([pose3D])
        pose3D = np.vstack(pose3D_list)
        pose3D = np.average(pose3D, axis=0)
        for frm in trange(framenum):
            np.savetxt(os.path.join(out_dir, f'pose{frm:04d}.txt'), pose3D[frm])
            # for cam in range(14):
            #     print(cam, np.average(np.linalg.norm(pose3D - pose3D_list[cam], axis=1)))