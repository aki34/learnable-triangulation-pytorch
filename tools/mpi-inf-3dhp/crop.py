import os
import cv2
import numpy as np
from tqdm import trange

MARGIN = 100

for subject in range(3, 4):
    for seq in range(2, 3):
        data_dir = os.path.join('../', f'S{subject}', f'Seq{seq}')
        for cam in list(range(0, 14)):
            out_img_dir = os.path.join(data_dir, 'Images_cropped', f'cam{cam}')
            os.makedirs(out_img_dir, exist_ok=True)

            bb = np.loadtxt(os.path.join(data_dir, 'BoundingBox', f'BoundingBox{cam}.txt'), dtype=int)

            in_video_file = os.path.join(data_dir, 'imageSequence', f'video_{cam}.avi')
            if not os.path.isfile(in_video_file):
                print(f"Couldn't find {in_video_file}")
                continue

            video = cv2.VideoCapture(in_video_file)
            FRAMENUM = int(min(bb.shape[0], int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
            WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f'S{subject} Seq{seq} cam{cam} FRAMENUM:{FRAMENUM}')
            for frm in trange(FRAMENUM):
                _, src = video.read()
                left, right, top, bottom = bb[frm]
                if left+right+top+bottom < 1:
                    dst = np.zeros((1, 1, 3))
                else:
                    left, top = max(0, left-MARGIN), max(0, top-MARGIN)
                    right, bottom = min(WIDTH, right+MARGIN), min(HEIGHT, bottom+MARGIN)
                    dst = src[top:bottom, left:right]
                cv2.imwrite(os.path.join(out_img_dir, f'img{frm:04d}.jpg'), dst)
