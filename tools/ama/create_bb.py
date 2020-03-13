import json
import os
import cv2
import numpy as np
from tqdm import trange

CAMNUM = 8
scenario_list = ['I_crane', 'I_jumping', 'I_march', 'I_squat', 'D_bouncing', 'D_handstand', 'D_march', 'D_squat', 'T_swing', 'T_samba']

with open('imgnum_dict.json', 'r') as f:
    imgnum_dict = json.load(f)

for scenario in scenario_list:
    sil_src_dir = scenario+'/silhouettes/'
    sil_src_file = 'Silhouette{}_{:04d}.png'
    dst_dir = scenario+'/boundingbox/'
    dst_file = 'BoundingBox{}.txt'
    os.makedirs(dst_dir, exist_ok=True)

    framenum = imgnum_dict[scenario]

    for cam in range(1, CAMNUM+1):
        print(f'{scenario} cam{cam}')
        bb = []
        for frm in trange(framenum):
            sil_img = cv2.imread(sil_src_dir+sil_src_file.format(cam, frm), flags=cv2.IMREAD_GRAYSCALE)
            if cam in [4, 5]:
                sil_img[:, 1455:] = 0
            if 'march' in scenario and cam == 2:
                sil_img[1000:] = 0
            if 'march' in scenario and cam == 5:
                sil_img[:, 1420:] = 0
            if 'squat' in scenario and cam == 2:
                sil_img[1000:] = 0
            if scenario == 'D_squat' and cam == 5:
                sil_img[:, 1420:] = 0
            HEIGHT, WIDTH = sil_img.shape
            h, w = np.where(sil_img==255)
            x_0, y_0 = max(w.min()-20, 0), max(h.min()-20, 0)
            x_1, y_1 = min(w.max()+20, WIDTH), min(h.max()+20, HEIGHT)
            bb.append([x_0, x_1, y_0, y_1])
        bb = np.vstack(bb)
        np.savetxt(dst_dir+dst_file.format(cam), bb, fmt='%d')
