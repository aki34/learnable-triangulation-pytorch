import sys
import os
import cv2
import random
import json
import numpy as np
from tqdm import trange
from tqdm import tqdm

camIdx = [f'cam{i}' for i in range(1, 9)]
draw_flag = False
id_flag = False
crop_flag = False
unidst_flag = False
path_camparam = './code/Release-v1.1/'


def video2imgs(subject, scenario, frm_list):
    for cam, camID in enumerate(camIdx):
        print(f's{subject} {scenario:>12s} {camID:>10s}', flush=True)
        if crop_flag:
            out_dir = f'../s{subject}/Images_cropped/{scenario}/{camID}/'
        else:
            out_dir = f'../s{subject}/Images/{scenario}/{camID}/'
        os.makedirs(out_dir, exist_ok=True)

        if unidst_flag:
            camparam = np.loadtxt(path_camparam+f'camparam/camparam_S{subject}_{camID}.txt')
            R = np.loadtxt(path_camparam+f'Rotmat/Rw2c_S{subject}_{camID}.txt')
            t = camparam[3:6]
            f = camparam[6:8]
            c = camparam[8:10]
            k = camparam[10:13]
            p = camparam[13:15]

            A = np.array(((f[0], 0, c[0]), (0, f[1], c[1]), (0, 0, 1)))
            dist = np.array((k[0], k[1], p[0], p[1], k[2]))

        video = f'../s{subject}/Videos/{scenario}/TC_S{subject}_{scenario}_{camID}.mp4'
        if not os.path.isfile(video):
            print("Couldn't find {0}".format(video))
            continue

        cap = cv2.VideoCapture(video)
        FRAMENUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if crop_flag:
            bb_file = f'../s{subject}/Boundingbox_fix_manual/{scenario}/BoundingBox{cam+1}.txt'
            bb = np.loadtxt(bb_file, dtype=int)
            FRAMENUM = min(FRAMENUM, bb.shape[0])

        print(FRAMENUM, end=' ', flush=True)
        if frm_list is None:
            frm_list = list(range(FRAMENUM))
        for frm in tqdm(frm_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
            out_file = f'frm{frm:04d}'
            ret, img = cap.read()

            if unidst_flag:
                img = cv2.undistort(img, A, dist, newCameraMatrix=A)
                out_file += '_undist'

            if draw_flag:
                points = np.loadtxt(f's{subject}/gt2Dpose/{scenario}/{camID}/pose{frm:04d}.txt')
                for i in range(points.shape[0]):
                    pos = (int(points[i][0]), int(points[i][1]))
                    img = cv2.circle(img, pos, 3, (0, 255, 0), thickness=-1)
                    if id_flag:
                        randPos = (int(points[i][0]) + random.randint(-10, 10), int(points[i][1]) + random.randint(-10, 10))
                        b = random.randint(0, 255)
                        g = random.randint(0, 255)
                        r = random.randint(0, 255)
                        cv2.putText(img, str(i), randPos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (b, g, r))
                        img = cv2.line(img, pos, randPos, (255, 255, 255))
                out_file += '_draw_jnt'

            if crop_flag:
                x0 = max(0, bb[frm, 0]-100)
                y0 = max(0, bb[frm, 1]-100)
                x1 = min(img.shape[1], bb[frm, 2]+100)
                y1 = min(img.shape[0], bb[frm, 3]+100)
                img = img[y0:y1, x0:x1]
                out_file += '_crop'

            cv2.imwrite(out_dir+out_file+f'_{camID}.jpg', img)

        cap.release()


if __name__ == "__main__":
    action_list = ['acting', 'freestyle', 'rom', 'walking']

    if len(sys.argv) == 1:
        subject_list = [1, 2, 3, 4, 5]
        for subject in subject_list:
            for action in action_list:
                for idx in [1, 2, 3]:
                    scenario = action+str(idx)
                    if scenario not in ['acting3', 'freestyle3', 'walking2']:
                        continue
                    if subject in [4, 5] and scenario in ['acting1', 'acting2', 'freestyle2', 'rom1', 'rom2', 'walking1', 'walking3']:
                        continue
                    video2imgs(subject, scenario, frm_list=None)

    else:
        subject = int(sys.argv[1])
        scenario = sys.argv[2]
        if len(sys.argv) < 4:
            frm_list = None
        else:
            frm_list = []
            for frm in sys.argv[3:]:
                frm_list.append(int(frm))
        video2imgs(subject, scenario, frm_list)
