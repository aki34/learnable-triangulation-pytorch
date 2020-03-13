import os
import cv2
import random
import numpy as np
from tqdm import trange

draw_flag = False
id_flag = False
crop_flag = False
unidst_flag = False

CAMNUM = 3
subject_list = [1, 2, 3]
action_list = ['Box', 'Gestures', 'Jog', 'ThrowCatch', 'Walking']
trial_list = [1]


for subject in subject_list:
    print('subject', subject)
    for action in action_list:
        for trial in trial_list:
            for cam in range(1, CAMNUM+1):
                print(cam, '...', end=' ', flush=True)
                out_dir = './S{}/imageSequence/{}_{}/{}/'.format(subject, action, trial, cam)
                os.makedirs(out_dir, exist_ok=True)

                if unidst_flag:
                    camparam = np.loadtxt('./S{}/Calibration_Data/intrinsic/camparam_scaled{}.txt'.format(subject, cam))
                    R = np.loadtxt('./S{}/Calibration_Data/extrinsic/RT{}.txt'.format(subject, cam))
                    t = camparam[3:6]
                    f = camparam[6:8]
                    c = camparam[8:10]
                    k = camparam[10:13]
                    p = camparam[13:15]

                    A = np.array(((f[0], 0, c[0]), (0, f[1], c[1]), (0, 0, 1)))
                    dist = np.array((k[0], k[1], p[0], p[1], k[2]))

                video = './S{}/Image_Data/{}_{}_(C{}).avi'.format(subject, action, trial, cam)
                if not os.path.isfile(video):
                    print("Couldn't find {}".format(video))
                    continue

                cap = cv2.VideoCapture(video)
                FRAMENUM = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if crop_flag:
                    bb_file = '../../code/PoseEstimation/data/Human3.6M/S{0}/{1}/{2}/BoundingBox.txt'.format(subject, scenario, camIdx[cam])
                    bb = np.loadtxt(bb_file, dtype=int)
                    FRAMENUM = min(FRAMENUM, bb.shape[0])

                print(FRAMENUM, end=' ', flush=True)
                for frm in trange(FRAMENUM):
                    out_file = 'img_{0:06d}'.format(frm)
                    # Capture frame-by-frame
                    ret, img = cap.read()

                    if unidst_flag:
                        img = cv2.undistort(img, A, dist, newCameraMatrix=A)
                        out_file += '_undist'

                    if draw_flag:
                        points = np.loadtxt('subject/S{0}/MyPoseFeatures/D2_Positions/{1}.{2}/frm{3:04d}_undist.txt'.format(subject, scenario, camIdx[cam], frm))
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

                    cv2.imwrite(out_dir+out_file+'.jpg', img)

                # When everything done, release the capture
                cap.release()
                # cv2.destroyAllWindows()

            # if i > 0 and j > 0:
            #     print('EXIT')
            #     exit(0)
