import os
import numpy as np

root_dir = '../'
for subject in range(1, 9):
    for seq in range(1, 3):
        data_dir = os.path.join(root_dir, f'S{subject}', f'Seq{seq}')
        calib_dir = os.path.join(data_dir, 'calibration')
        os.makedirs(os.path.join(calib_dir, 'intrinsic'), exist_ok=True)
        os.makedirs(os.path.join(calib_dir, 'extrinsic'), exist_ok=True)
        with open(os.path.join(data_dir, 'camera.calibration'), 'r') as f:
            f.readline()
            for cam in range(14):
                line = f.readline()  # name
                line = f.readline()  # sensor
                W, H = np.array(f.readline().split()[1:]).astype(np.float32)  # size
                line = f.readline()  # animated
                K = np.array(f.readline().split()[1:]).reshape(4, 4).astype(np.float32)  # intrinsic
                RT = np.array(f.readline().split()[1:]).reshape(4, 4).astype(np.float32)  # extrinsic
                line = f.readline()  # radial

                focas = K[0, 0]
                aspect = K[1, 1] / K[0, 0]
                camparam = np.array([focas, 0, 1, aspect, K[0, 2], K[1, 2], W, H])

                np.savetxt(os.path.join(calib_dir, 'intrinsic', f'cammat{cam}.txt'), K[:3, :3], fmt='%.5f')
                np.savetxt(os.path.join(calib_dir, 'intrinsic', f'camparam{cam:02d}.txt'), camparam.reshape(1, -1), fmt='%.5f')
                np.savetxt(os.path.join(calib_dir, 'intrinsic', f'camparam_scaled{cam}.txt'), camparam.reshape(1, -1), fmt='%.5f')
                np.savetxt(os.path.join(calib_dir, 'extrinsic', f'rt{cam}.txt'), RT, fmt='%.5f')
                np.savetxt(os.path.join(calib_dir, 'extrinsic', f'rt{cam}_list.txt'), RT.reshape(-1, 1), fmt='%.5f')
