import os
import glob
import cv2
from tqdm import trange

for subject in range(1, 9):
    for scenario in [1, 2]:

        video_dir = f'../S{subject}/Seq{scenario}/imageSequence/'
        res = (384, 384)

        video_list = glob.glob(video_dir+'*.avi')
        for i, video_file in enumerate(video_list):
            print(video_file)
            in_video = cv2.VideoCapture(video_file)
            FRAMENUM = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
            FPS = in_video.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
            # fourcc = cv2.VideoWriter_fourcc('x', '2', '6', '4')
            # fourcc = cv2.VideoWriter_fourcc('d', 'i', 'v', 'x')
            out_video = cv2.VideoWriter(os.path.splitext(video_file)[0]+'.mp4', fourcc, FPS, res)

            for frm in trange(FRAMENUM):
                _, src = in_video.read()
                dst = cv2.resize(src, res)
                out_video.write(dst)
            in_video.release()
            out_video.release()
