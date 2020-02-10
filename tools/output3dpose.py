import sys
import os
import pickle
from tqdm import trange
import numpy as np

if len(sys.argv) < 3:
    print('usage:')
    exit()

log_dir = sys.argv[1]
epoch = int(sys.argv[2])
result_dir = log_dir+'/checkpoints/{:04d}/'.format(epoch)
result_file = result_dir+'results.pkl'
out_dir = result_dir+'pose/'
os.makedirs(out_dir, exist_ok=True)

with open(result_file, 'rb') as f:
    result = pickle.load(f)
    pose3d = result['keypoints_3d']
    idx = result['indexes']
    DATANUM = idx.shape[0]

    for i in trange(0, DATANUM):
        np.savetxt(out_dir+'pose{0:06d}.txt'.format(idx[i]), pose3d[i], delimiter=';')
