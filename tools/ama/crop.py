import os
import json
import cv2
import numpy as np
from tqdm import trange

CAMNUM = 8
scenario_list = ['I_crane', 'I_jumping', 'I_march', 'I_squat', 'D_bouncing', 'D_handstand', 'D_march', 'D_squat', 'T_swing', 'T_samba']

with open('imgnum_dict.json', 'r') as f:
    imgnum_dict = json.load(f)

for scenario in scenario_list:
    bb_file = scenario+'/boundingbox/BoundingBox{}.txt'
    img_src_dir = scenario+'/images/'
    img_dst_dir = scenario+'/images_crop/cam{}/'
    if scenario in ['D_march', 'D_squat', 'I_march', 'I_squat']:
        img_src_file = 'Camera{}_{:04d}.jpg'
    else:
        img_src_file = 'Image{}_{:04d}.png'
    img_dst_file = 'Image{}_{:04d}.jpg'

    framenum = imgnum_dict[scenario]

    for cam in range(1, CAMNUM+1):
        print(f'{scenario} cam{cam}')
        bb_cam = np.loadtxt(bb_file.format(cam), dtype=int)
        os.makedirs(img_dst_dir.format(cam), exist_ok=True)
        for frm in trange(framenum):
            bb = bb_cam[frm]
            src = cv2.imread(img_src_dir.format(cam)+img_src_file.format(cam, frm))
            dst = src[bb[2]:bb[3], bb[0]:bb[1]]
            cv2.imwrite(img_dst_dir.format(cam)+img_dst_file.format(cam, frm), dst)
