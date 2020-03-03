import os
import pickle
import numpy as np

# label = np.load('../data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', allow_pickle=True).item()
label = np.load('../data/mpi_inf_3dhp/mpiinf3dhp-multiview-labels-bboxes.npy', allow_pickle=True).item()
# print(label.keys())
# print(label['table'].dtype)
# print(label['table'].shape)
# print(label['table']['subject_idx'])
subject_map = label['subject_names']
action_map = label['action_names']
table = label['table']
# print(label['action_names'])

# with open('sequence_mappings.pkl', 'rb') as f:
#     scequence_mapping = pickle.load(f)
# for key, value in scequence_mapping.items():
#     print(key, value)

# with open('../logs/eval_human36m_alg_AlgebraicTriangulationNet@02.03.2020-12:38:56/checkpoints/0000/resutls_test.pkl', 'rb') as f:
# with open('../logs/algebraic-precalulated-result/train.pkl', 'rb') as f:
with open('../logs/eval_human36m_alg_AlgebraicTriangulationNet@03.03.2020-23:17:39/checkpoints/0000/results_test.pkl', 'rb') as f:
    result = pickle.load(f)

# print(table[::300]['action_idx'])
print(result.keys())
for i, index in enumerate(result['indexes']):
    subject = subject_map[table[index]['subject_idx']]  # np.concatenate(result['subject'])[i]  # subject_map[table[index]['subject_idx']]
    action = action_map[table[index]['action_idx']]  # np.concatenate(result['action'])[i]  # action_map[table[index]['action_idx']]
    frm = table[index]['frame_idx']  # result['frame_idx'][i]  # table[index]['frame_idx']
    pose = result['keypoints_3d'][i]
    LT_dir = os.path.join('../data/mpi-inf-3dhp', subject, action, '3Dpose_LT')
    os.makedirs(LT_dir, exist_ok=True)
    np.savetxt(os.path.join(LT_dir, f'pose{frm:04d}.txt'), pose, fmt='%.5f')
    # print(subject, action, frm, np.concatenate(result['subject'])[i], np.concatenate(result['action'])[i], result['frame_idx'][i])
    print(subject, action, frm)

    # with open(os.path.join(subject, action, 'frm_idx_LT.txt'), 'a') as f:
    #     f.write(f'{frm}\n')
