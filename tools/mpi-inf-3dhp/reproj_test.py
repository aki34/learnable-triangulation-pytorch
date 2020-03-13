import os
import glob
import numpy as np

root_dir = '../'
for subject in range(1, 2):
    for seq in range(1, 3):
        data_dir = os.path.join(root_dir, f'S{subject}', f'Seq{seq}')
        for cam in range(14):
            cammat = np.loadtxt(os.path.join(data_dir, 'calibration', 'intrinsic', f'cammat{cam}.txt'))
            camparam = np.loadtxt(os.path.join(data_dir, 'calibration', 'intrinsic', f'camparam_scaled{cam}.txt'))
            RT = np.loadtxt(os.path.join(data_dir, 'calibration', 'extrinsic', f'rt{cam}.txt'))
            r = RT[:3, :3]
            t = RT[:3, 3]

            f = camparam[0]
            dist = camparam[1]
            px, py = camparam[2:4]
            cx, cy = camparam[4:6]
            width, height = camparam[6:8]

            framenum = len(glob.glob(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{cam}', '*.txt')))
            for frm in range(framenum):
                pose3D = np.loadtxt(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{cam}', f'pose{frm:04d}.txt'))
                pose2D = np.loadtxt(os.path.join(data_dir, 'gt2Dpose_cam', f'cam{cam}', f'pose{frm:04d}.txt'))

                # pose3D = np.dot(r, pose3D.T).T + t*1000

                reproj2D = np.dot(cammat, pose3D.T)
                reproj2D = reproj2D[:2] / reproj2D[2]
                reproj2D = reproj2D.T

                print(np.average(np.linalg.norm(pose2D - reproj2D, axis=1)))
