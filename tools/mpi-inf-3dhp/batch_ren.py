import sys
import os
import glob
from tqdm import trange

subject = int(sys.argv[1])
scenario = sys.argv[2]
framenum = int(sys.argv[3])

for c in range(14):
    print(f'S{subject} {scenario} cam{c}')
    for frm in trange(framenum):
        img_dir = os.path.join('..', f'S{subject}', scenario, 'Images', f'cam{c}')
        img_file_old = os.path.join(img_dir, f'frm{frm:04d}_cam{c}.jpg')
        img_file_new = os.path.join(img_dir, f'frm{frm:06d}_cam{c}.jpg')
        try:
            os.rename(img_file_old, img_file_new)
        except FileNotFoundError as e:
            print(e)
            continue