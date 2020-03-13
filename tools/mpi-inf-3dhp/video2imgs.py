import sys
import os
import cv2
import random
import json
import numpy as np
from tqdm import trange
from tqdm import tqdm

draw_flag = False
id_flag = False
crop_flag = True
unidst_flag = False
path_camparam = './code/Release-v1.1/'


def video2imgs(subject, scenario, frm_list):
    for cam in list(range(14))[::-1]:
        print(f's{subject} {scenario:>12s} cam{cam}', flush=True)
        if crop_flag:
            out_dir = f'../S{subject}/{scenario}/Images_cropped/cam{cam}/'
        else:
            out_dir = f'../S{subject}/{scenario}/Images/cam{cam}/'
        os.makedirs(out_dir, exist_ok=True)

        if unidst_flag:
            camparam = np.loadtxt(path_camparam+f'camparam/camparam_S{subject}_cam{cam}.txt')
            R = np.loadtxt(path_camparam+f'Rotmat/Rw2c_S{subject}_cam{cam}.txt')
            t = camparam[3:6]
            f = camparam[6:8]
            c = camparam[8:10]
            k = camparam[10:13]
            p = camparam[13:15]

            A = np.array(((f[0], 0, c[0]), (0, f[1], c[1]), (0, 0, 1)))
            dist = np.array((k[0], k[1], p[0], p[1], k[2]))

        video = f'../S{subject}/{scenario}/imageSequence/video_{cam}.avi'
        if not os.path.isfile(video):
            print("Couldn't find {0}".format(video))
            continue

        cap = cv2.VideoCapture(video)
        FRAMENUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if crop_flag:
            bb_file = f'../S{subject}/{scenario}/BoundingBox/BoundingBox{cam}.txt'
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
                points = np.loadtxt(f'../S{subject}/{scenario}/gt2Dpose_cam/cam{cam}/pose{frm:04d}.txt')
                for i in range(points.shape[0]):
                    pos = (int(points[i][0]), int(points[i][1]))
                    img = cv2.circle(img, pos, 7, (0, 255, 0), thickness=-1)
                    if id_flag:
                        randPos = (int(points[i][0]) + random.randint(-100, 100), int(points[i][1]) + random.randint(-100, 100))
                        b = random.randint(0, 255)
                        g = random.randint(0, 255)
                        r = random.randint(0, 255)
                        cv2.putText(img, str(i), randPos, cv2.FONT_HERSHEY_SIMPLEX, 2, (b, g, r), thickness=2)
                        img = cv2.line(img, pos, randPos, (255, 255, 255))
                out_file += '_draw_jnt'

            if crop_flag:
                left, right, top, bottom = bb[frm]
                x0 = max(0, left-100)
                y0 = max(0, top-100)
                x1 = min(img.shape[1], right+100)
                y1 = min(img.shape[0], bottom+100)
                img = img[y0:y1, x0:x1]
                out_file += '_crop'

            cv2.imwrite(out_dir+out_file+f'_cam{cam}.jpg', img)

        cap.release()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        subject_list = list(range(1, 9))
        for subject in subject_list:
            for seq in range(1, 3):
                video2imgs(subject, f'Seq{seq}', frm_list=None)

    else:
        subject = int(sys.argv[1])
        scenario = sys.argv[2]
        if len(sys.argv) < 4:
            frm_list = list(range(0, 12430, 5))
        else:
            frm_list = []
            for frm in sys.argv[3:]:
                frm_list.append(int(frm))
        video2imgs(subject, scenario, frm_list)
