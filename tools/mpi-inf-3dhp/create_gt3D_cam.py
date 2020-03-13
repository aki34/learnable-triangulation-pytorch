import os
from scipy import io
import numpy as np
from tqdm import trange

root_dir = '../'
for subject in range(1, 9):
    for seq in range(1, 3):
        print(f'S{subject} Seq{seq}')
        data_dir = os.path.join(root_dir, f'S{subject}', f'Seq{seq}')
        annot = io.loadmat(os.path.join(data_dir, 'annot.mat'), squeeze_me=True)
        pose3d_tmp = annot['annot3']
        pose_list = []
        for i in range(len(pose3d_tmp)):
            pose_list.append([pose3d_tmp[i].astype(np.float32)])
        gt_pose = np.vstack(pose_list)
        for cam in range(14):
            os.makedirs(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{cam}'), exist_ok=True)
            np.save(os.path.join(data_dir, 'gt3Dpose_cam', f'gtpose_cam{cam}.npy'), gt_pose[cam])
            for frm in trange(gt_pose[cam].shape[0]):
                np.savetxt(os.path.join(data_dir, 'gt3Dpose_cam', f'cam{cam}', f'pose{frm:04d}.txt'), gt_pose[cam, frm].reshape(-1, 3), fmt='%.5f')
