"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy-multiview.py <path/to/Human3.6M-root> <path/to/una-dinosauria-data/h36m> <path/to/bboxes-Human36M-squared.npy>`
"""
import os, sys
import numpy as np
import h5py

retval = {
    # 'subject_names': ['S1', 'S2', 'S3'],
    'subject_names': ['S1'],
    'camera_names': ['1', '2', '3'],
    # 'action_names': [
    #     'Box_1', 'Box_3',
    #     'Gestures_1', 'Gestures_3',
    #     'Jog_1', 'Jog_3',
    #     'ThrowCatch_1', 'ThrowCatch_3',
    #     'Walking_1', 'Walking_3']    
    'action_names': [
        'Box_1'
        # 'Gestures_1',
        # 'Jog_1'
        # 'ThrowCatch_1',
        # 'Walking_1'
        ]
}
retval['cameras'] = np.empty(
    (len(retval['subject_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3, 3)),
        ('t', np.float32, (3, 1)),
        ('K', np.float32, (3, 3)),
        ('dist', np.float32, 5)
    ]
)

table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (20, 3)),  # HumanEva format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4))
])
retval['table'] = []

heva_root = './'  # sys.argv[1]
# assert all(subject.startswith('S') for subject in os.listdir(h36m_root))

# una_dinosauria_root = sys.argv[2]
# cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'), 'r')

# Fill retval['cameras']
for subject_idx, subject in enumerate(retval['subject_names']):
    calib_dir = './{}/Calibration_Data/'.format(subject)
    intrinsic_dir = calib_dir+'intrinsic/'
    extrinsic_dir = calib_dir+'extrinsic/'

    for camera_idx, camera in enumerate(retval['camera_names']):
        # assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        # camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
        camera_retval = retval['cameras'][subject_idx][camera_idx]

        # def camera_array_to_name(array):
        #     return ''.join(chr(int(x[0])) for x in array)
        # assert camera_array_to_name(camera_params['Name']) == camera

        rt = np.loadtxt(extrinsic_dir+'RT{}_list.txt'.format(camera)).reshape(4, 4)

        # camera_retval['R'] = np.array(camera_params['R']).T
        # camera_retval['t'] = -camera_retval['R'] @ camera_params['T']
        camera_retval['R'] = rt[:3, :3]
        camera_retval['t'] = rt[:3, 3].reshape(-1 ,1)

        cam_mat = np.loadtxt(calib_dir+'C{}.cal'.format(camera))

        camera_retval['K'] = 0
        # camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
        # camera_retval['K'][0, 0] = camera_params['f'][0]
        # camera_retval['K'][1, 1] = camera_params['f'][1]
        # camera_retval['K'][2, 2] = 1.0
        camera_retval['K'][:2, 2] = cam_mat[2:4]
        camera_retval['K'][0, 0] = cam_mat[0]
        camera_retval['K'][1, 1] = cam_mat[1]
        camera_retval['K'][2, 2] = 1.0

        # camera_retval['dist'][:2] = camera_params['k'][:2, 0]
        # camera_retval['dist'][2:4] = camera_params['p'][:, 0]
        # camera_retval['dist'][4] = camera_params['k'][2, 0]
        camera_retval['dist'][:2] = cam_mat[5:7]
        camera_retval['dist'][2:5] = np.zeros(3)

# Fill bounding boxes
# bboxes = np.load(sys.argv[3], allow_pickle=True).item()


def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return top, left, bottom, right


# for subject in bboxes.keys():
#     for action in bboxes[subject].keys():
#         for camera, bbox_array in bboxes[subject][action].items():
#             for frame_idx, bbox in enumerate(bbox_array):
#                 bbox[:] = square_the_bbox(bbox)

# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's "ground truth".
BBOXES_SOURCE = 'GT'  # or 'MRCNN' or 'SSD'

if BBOXES_SOURCE != 'GT':
    def replace_gt_bboxes_with_cnn(bboxes_gt, bboxes_detected_path, detections_file_list):
        """
            Replace ground truth bounding boxes with boxes from a CNN detector.
        """
        with open(bboxes_detected_path, 'r') as f:
            import json
            bboxes_detected = json.load(f)

        with open(detections_file_list, 'r') as f:
            for bbox, filename in zip(bboxes_detected, f):
                # parse filename
                filename = filename.strip()
                filename, frame_idx = filename[:-15], int(filename[-10:-4])-1
                filename, camera_name = filename[:-23], filename[-8:]
                slash_idx = filename.rfind('/')
                filename, action_name = filename[:slash_idx], filename[slash_idx+1:]
                subject_name = filename[filename.rfind('/')+1:]

                bbox, _ = bbox[:4], bbox[4] # throw confidence away
                bbox = square_the_bbox([bbox[1], bbox[0], bbox[3]+1, bbox[2]+1]) # LTRB to TLBR
                bboxes_gt[subject_name][action_name][camera_name][frame_idx] = bbox

    detections_paths = {
        'MRCNN': {
            'train': "/Vol1/dbstore/datasets/Human3.6M/extra/train_human36m_MRCNN.json",
            'test': "/Vol1/dbstore/datasets/Human3.6M/extra/test_human36m_MRCNN.json"
        },
        'SSD': {
            'train': "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-train-human36m.json",
            'test': "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-human36m.json"
        }
    }

    replace_gt_bboxes_with_cnn(
        bboxes,
        detections_paths[BBOXES_SOURCE]['train'],
        '/Vol1/dbstore/datasets/Human3.6M/train-images-list.txt')

    replace_gt_bboxes_with_cnn(
        bboxes,
        detections_paths[BBOXES_SOURCE]['test'],
        '/Vol1/dbstore/datasets/Human3.6M/test-images-list.txt')

# fill retval['table']
# from action_to_una_dinosauria import action_to_una_dinosauria

for subject_idx, subject in enumerate(retval['subject_names']):
    subject_path = os.path.join(heva_root, subject)
    # actions = os.listdir(subject_path)
    # try:
    #     actions.remove('MySegmentsMat') # folder with bbox *.mat files
    # except ValueError:
    #     pass

    for action_idx, action in enumerate(retval['action_names']):
        print(subject, action)

        image_path = os.path.join(subject_path, action, 'imageSequence')
        pose_path = os.path.join(subject_path, '3Dpose_TXT_Data_align')
        poset_txt_path = os.path.join(pose_path, action)
        # if not os.path.isdir(action_path):
        #     raise FileNotFoundError(action_path)

        for camera_idx, camera in enumerate(retval['camera_names']):
            camera_path = os.path.join(subject_path, 'imageSequence', action, camera)
        #     if os.path.isdir(camera_path):
        #         frame_idxs = sorted([int(name[4:-4])-1 for name in os.listdir(camera_path)])
        #         assert len(frame_idxs) > 15, 'Too few frames in %s' % camera_path # otherwise WTF
        #         break
        # else:
        #     raise FileNotFoundError(action_path)

        # 16 joints in MPII order + "Neck/Nose"
        # valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
        # with h5py.File(os.path.join(una_dinosauria_root, subject, 'MyPoses', '3D_positions',
        #                             '%s.h5' % action_to_una_dinosauria[subject].get(action, action.replace('-', ' '))), 'r') as poses_file:
        #     poses_world = np.array(poses_file['3D_positions']).T.reshape(-1, 32, 3)[frame_idxs][:, valid_joints]

        frame_idxs = np.loadtxt(pose_path+'/'+action+'_valid_list.txt', dtype=int).tolist()
        poses_world = []
        for frm in frame_idxs:
            pose_world = np.loadtxt(poset_txt_path+'/frm{:04d}.txt'.format(frm))
            poses_world .append([pose_world])
        poses_world = np.vstack(poses_world)

        table_segment = np.empty(len(frame_idxs), dtype=table_dtype)
        table_segment['subject_idx'] = subject_idx
        table_segment['action_idx'] = action_idx
        table_segment['frame_idx'] = frame_idxs
        table_segment['keypoints'] = poses_world
        table_segment['bbox_by_camera_tlbr'] = 0 # let a (0,0,0,0) bbox mean that this view is missing

        for (camera_idx, camera) in enumerate(retval['camera_names']):
            # camera_path = os.path.join(action_path, camera)
            # if not os.path.isdir(camera_path):
            #     print('Warning: camera %s isn\'t present in %s/%s' % (camera, subject, action))
            #     continue

            for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'], frame_idxs):
                # bbox[camera_idx] = bboxes[subject][action][camera][frame_idx]
                # bbox[camera_idx] = (67, 195, 408 ,425)
                bbox[camera_idx] = (0, 0, 480, 640)

        retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print('Total frames in HumanEva:', len(retval['table']))
np.save(f'humaneva-multiview-labels-bboxes.npy', retval)
