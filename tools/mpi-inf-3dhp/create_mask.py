import os
import glob
import cv2
import numpy as np
from tqdm import trange

for subject in range(1, 9):
    for scenario in ['Seq1', 'Seq2']:
        video_list = glob.glob(os.path.join('..', f'S{subject}', scenario, 'FGmasks', '*.avi'))
        out_video_dir = os.path.join('..', f'S{subject}', scenario, 'BGmasks')
        os.makedirs(out_video_dir, exist_ok=True)
        for i, video_file in enumerate(video_list):
            print(video_file)
            in_video = cv2.VideoCapture(video_file)
            FRAMENUM = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))

            mask = []
            for frm in trange(0, FRAMENUM, 100):
                in_video.set(cv2.CAP_PROP_POS_FRAMES, frm)
                _, src = in_video.read()
                mask.append([src])
            mask = np.vstack(mask)
            mask = np.average(mask, axis=0).astype(np.uint8)
            # _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
            mask[mask > 120] = 255
            mask[mask <= 120] = 0
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask[mask < 70] = 0
            # mask[mask > 70] = 255

            cv2.imwrite(os.path.join(out_video_dir, f'{os.path.splitext(os.path.basename(video_file))[0]}_mask.jpg'), mask[:, :, 2])
