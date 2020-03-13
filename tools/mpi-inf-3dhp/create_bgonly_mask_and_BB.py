import os
import glob
import cv2
import numpy as np
from tqdm import trange

for subject in range(2, 3):
    for scenario in ['Seq1']:
        video_list = sorted(glob.glob(os.path.join('..', f'S{subject}', scenario, 'FGmasks', '*.avi')))
        out_video_dir = os.path.join('..', f'S{subject}', scenario, 'BGmasks')
        out_bb_dir = os.path.join('..', f'S{subject}', scenario, 'BoundingBox')
        os.makedirs(out_video_dir, exist_ok=True)
        os.makedirs(out_bb_dir, exist_ok=True)
        for i, video_file in enumerate(video_list[::-1]):
            print(video_file)
            file_name_base = os.path.splitext(os.path.basename(video_file))[0]
            cam = int(file_name_base.split('_')[1])
            in_video = cv2.VideoCapture(video_file)
            FRAMENUM = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
            FPS = in_video.get(cv2.CAP_PROP_FPS)
            WIDTH = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            HEIGHT = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            # out_video_file = os.path.join(out_video_dir, file_name_base+'.mp4')
            # out_video = cv2.VideoWriter(out_video_file, fourcc, FPS, (WIDTH, HEIGHT))

            mask = cv2.imread(os.path.join(out_video_dir, f'{file_name_base}_mask.jpg'), cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32)

            kernel = np.ones((25, 25), dtype=np.uint8)
            in_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            bb = []
            for frm in trange(FRAMENUM):
                _, src = in_video.read()
                src = src[:, :, 2]
                rows, cols = np.where(src < 1)
                dst = src.astype(np.float32) - mask
                dst[rows, cols] = 0
                dst[dst < 240] = 0
                dst[dst >= 240] = 255
                dst = cv2.morphologyEx(dst.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                # cv2.imwrite(os.path.join(out_video_dir, f'{file_name_base}_bg.png'), dst)
                # out_video.write(dst)

                h, w = np.where(dst == 255)
                try:
                    left, top = max(w.min()-20, 0), max(h.min()-20, 0)
                    right, bottom = min(w.max()+20, WIDTH), min(h.max()+20, HEIGHT)
                except ValueError:
                    left, top, right, bottom = 0, 0, 0, 0
                bb.append([left, right, top, bottom])
            in_video.release()
            # out_video.release()
            bb = np.vstack(bb)
            np.savetxt(os.path.join(out_bb_dir, f'BoundingBox{cam}.txt'), bb, fmt='%d')
