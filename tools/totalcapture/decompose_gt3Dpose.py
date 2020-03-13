import os
import numpy as np
from tqdm import trange

subject_list = [1, 2, 3, 4, 5]
# subject_list = [1]
action_list = ['acting', 'freestyle', 'rom', 'walking']
CAMNUM = 8

for subject in subject_list:
    for action in action_list:
        for idx in [1, 2, 3]:
            scenario = action+str(idx)
            # if scenario not in ['acting3', 'freestyle3', 'walking2']:
            #     continue
            if subject in [4, 5] and scenario in ['acting1', 'acting2', 'freestyle2', 'rom1', 'rom2', 'walking1', 'walking3']:
                continue
            print(f's{subject} {scenario}')

            pose_file = os.path.join(f's{subject}', 'Vicon_pos_ori', scenario, 'gt_skel_gbl_pos.txt')
            out_dir = os.path.join(f's{subject}', 'gt3Dpose', scenario)
            os.makedirs(out_dir, exist_ok=True)
            with open(pose_file, 'r') as f:
                joint_order = f.readline().split()
                pose_list = f.readlines()
                poses = []
                for frm in trange(len(pose_list)):
                    pose = np.array(pose_list[frm].split()).astype(np.float32).reshape(-1, 3)
                    pose *= 0.0254  # inch ==> meter
                    np.savetxt(os.path.join(out_dir, f'pose{frm:04d}.txt'), pose, fmt='%.5f')
                    poses.append([pose])
                poses = np.vstack(poses)
                np.save(os.path.join(out_dir, 'gtpose.npy'), poses)
