import sys
import os
import pickle
from tqdm import trange
import numpy as np

if len(sys.argv) < 3:
    print(f'USAGE: python {sys.argv[0]} <log_dir> <epoch>')
    exit()

log_dir = sys.argv[1]
epoch = int(sys.argv[2])
result_file = os.path.join(log_dir, 'checkpoints', f'{epoch:04d}', 'results.pkl')
pose_out_dir = os.path.join(log_dir, 'pose')
os.makedirs(pose_out_dir, exist_ok=True)

with open(result_file, 'rb') as f:
    result = pickle.load(f)
    print(result.keys())
    pose3d = result['keypoints_3d']
    idx = result['indexes']
    DATANUM = idx.shape[0]

    for i in trange(0, DATANUM):
        np.savetxt(os.path.join(pose_out_dir, f'pose{idx[i]:04d}.txt'), pose3d[i])
