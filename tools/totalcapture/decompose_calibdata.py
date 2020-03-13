import os
import numpy as np

with open('calibration.cal') as f:
    f.readline()
    for cam in range(8):
        top, bottom, left, right = f.readline().split()
        fx, fy, cx, cy = f.readline().split()
        dist = f.readline()
        R00, R01, R02 = f.readline().split()
        R10, R11, R12 = f.readline().split()
        R20, R21, R22 = f.readline().split()
        t0, t1, t2 = f.readline().split()

        K = np.eye(3)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)
        aspect = float(fy) / float(fx)
        camparam = np.array([float(fx), 0, 1, aspect, float(cx), float(cy), 1920, 1080])
        R = np.array([[float(R00), float(R01), float(R02)],
                      [float(R10), float(R11), float(R12)],
                      [float(R20), float(R21), float(R22)]])
        t = np.array([float(t0), float(t1), float(t2)])

        RT = np.concatenate((R, t.reshape(3, 1)), axis=1)
        RT = np.concatenate((RT, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

        out_dir = os.path.join('calibration')
        os.makedirs(os.path.join(out_dir, 'intrinsic'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'extrinsic'), exist_ok=True)
        np.savetxt(os.path.join(out_dir, 'intrinsic', f'cammat{cam+1}.txt'), K, fmt='%.5f')
        np.savetxt(os.path.join(out_dir, 'intrinsic', f'camparam_scaled{cam+1}.txt'), camparam.reshape(1, -1), fmt='%.5f')
        np.savetxt(os.path.join(out_dir, 'intrinsic', f'camparam{cam+1:02d}.txt'), camparam.reshape(1, -1), fmt='%.5f')
        np.savetxt(os.path.join(out_dir, 'extrinsic', f'rt{cam+1}.txt'), RT, fmt='%.5f')
        np.savetxt(os.path.join(out_dir, 'extrinsic', f'rt{cam+1}_list.txt'), RT.reshape(-1, 1), fmt='%.5f')
