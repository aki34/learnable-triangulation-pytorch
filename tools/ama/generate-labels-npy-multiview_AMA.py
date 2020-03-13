"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy-multiview.py <path/to/Human3.6M-root> <path/to/una-dinosauria-data/h36m> <path/to/bboxes-Human36M-squared.npy>`
"""
import os
import sys
import numpy as np
import h5py
import json

retval = {
    # 'subject_names': ['I', 'D', 'T'],
    'camera_names': [f'{i}' for i in range(1, 9)],
    'action_names': ['I_crane', 'I_jumping', 'I_march', 'I_squat',
                     'D_bouncing', 'D_handstand', 'D_march', 'D_squat',
                     'T_samba', 'T_swing']
}
retval['cameras'] = np.empty(
    (len(retval['action_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3, 3)),
        ('t', np.float32, (3, 1)),
        ('K', np.float32, (3, 3)),
        ('dist', np.float32, 5)
    ]
)

table_dtype = np.dtype([
    # ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (21, 3)),  # Currently there are no GT, so the shape is the same as h36m
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4))
])
retval['table'] = []

with open('imgnum_dict.json', 'r') as f:
    imgnum_dict = json.load(f)

root = './'  # sys.argv[1]

# una_dinosauria_root = sys.argv[2]
# cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'), 'r')

# Fill retval['cameras']
for action_idx, action in enumerate(retval['action_names']):
    calib_dir = os.path.join(root, f'{action}', 'calibration')
    intrinsic_dir = os.path.join(calib_dir, 'intrinsic')
    extrinsic_dir = os.path.join(calib_dir, 'extrinsic')

    for camera_idx, camera in enumerate(retval['camera_names']):
        camera_retval = retval['cameras'][action_idx][camera_idx]
        rt = np.loadtxt(os.path.join(extrinsic_dir, f'RT{camera}.txt')).astype(np.float32)

        camera_retval['R'] = rt[:3, :3]
        camera_retval['t'] = rt[:3, 3].reshape(-1, 1)
        camera_retval['K'] = np.loadtxt(os.path.join(intrinsic_dir, f'cammat{camera}.txt')).astype(np.float32)
        camera_retval['dist'] = np.zeros(5, dtype=np.float32)


# Fill bounding boxes
# bboxes = np.load(sys.argv[3], allow_pickle=True).item()
def square_the_bbox(bbox):
    left, right, top, bottom = bbox
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


bboxes = {}
for action_idx, action in enumerate(retval['action_names']):
    bboxes[action] = {}
    for camera_idx, camera in enumerate(retval['camera_names']):
        bboxes[action][camera] = []
        bb_file = os.path.join(root, action, 'boundingbox', f'BoundingBox{camera}.txt')
        bb_tmp = np.loadtxt(bb_file).astype(np.float32)
        for i in range(bb_tmp.shape[0]):
            bboxes[action][camera].append(square_the_bbox(bb_tmp[i]))
        bboxes[action][camera] = np.vstack(bboxes[action][camera])

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

    replace_gt_bboxes_with_cnn(bboxes,
                               detections_paths[BBOXES_SOURCE]['train'],
                               '/Vol1/dbstore/datasets/Human3.6M/train-images-list.txt')

    replace_gt_bboxes_with_cnn(bboxes,
                               detections_paths[BBOXES_SOURCE]['test'],
                               '/Vol1/dbstore/datasets/Human3.6M/test-images-list.txt')

for action_idx, action in enumerate(retval['action_names']):
    print(action)

    image_path = os.path.join(root, action, 'images')
    # pose_path = os.path.join(root, '3Dpose')
    # poset_txt_path = os.path.join(pose_path, action)
    # if not os.path.isdir(action_path):
    #     raise FileNotFoundError(action_path)

    # RETRIEVE AVALIABLE FRAME INDEXES
    # for camera_idx, camera in enumerate(retval['camera_names']):
    #     camera_path = os.path.join(root, 'imageSequence', action, camera)
    #     if os.path.isdir(camera_path):
    #         frame_idxs = sorted([int(name[4:-4])-1 for name in os.listdir(camera_path)])
    #         assert len(frame_idxs) > 15, 'Too few frames in %s' % camera_path # otherwise WTF
    #         break
    # else:
    #     raise FileNotFoundError(action_path)

    # RETRIEVE GT 3D POSE
    # 16 joints in MPII order + "Neck/Nose"
    # valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
    # with h5py.File(os.path.join(una_dinosauria_root, subject, 'MyPoses', '3D_positions',
    #                             '%s.h5' % action_to_una_dinosauria[subject].get(action, action.replace('-', ' '))), 'r') as poses_file:
    #     poses_world = np.array(poses_file['3D_positions']).T.reshape(-1, 32, 3)[frame_idxs][:, valid_joints]

    frame_idxs = list(range(imgnum_dict[action]))
    poses_world = []
    for frm in frame_idxs:
        # pose_world = np.loadtxt(poset_txt_path+'/frm{:04d}.txt'.format(frm))
        pose_world = np.zeros((21, 3), dtype=np.float32)
        poses_world.append([pose_world])
    poses_world = np.vstack(poses_world)

    table_segment = np.empty(len(frame_idxs), dtype=table_dtype)
    table_segment['action_idx'] = action_idx
    table_segment['frame_idx'] = frame_idxs
    table_segment['keypoints'] = poses_world
    table_segment['bbox_by_camera_tlbr'] = 0  # let a (0,0,0,0) bbox mean that this view is missing

    for (camera_idx, camera) in enumerate(retval['camera_names']):
        # camera_path = os.path.join(root, action, camera)
        # if not os.path.isdir(camera_path):
        #     print(f'Warning: camera {camera} isn\'t present in {action}')
        #     continue

        for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'], frame_idxs):
            bbox[camera_idx] = bboxes[action][camera][frame_idx]

    retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print('Total frames in ArticulatedMeshAnimation:', len(retval['table']))
np.save(f'ama-multiview-labels-bboxes.npy', retval)
