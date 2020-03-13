import os
from scipy import io
import h5py
import numpy as np

# annot = h5py.File(os.path.join('..', 'mpi_inf_3dhp_test_set', 'TS1', 'annot_data.mat'), 'r')
annot = io.loadmat(os.path.join('..', 'S1', 'Seq1', 'annot.mat'), squeeze_me=True)

for key in annot.keys():
    print(key)
#     # print(annot[key])
#     print(np.array(annot[key])[0])
#     print(np.array(annot[key])[10])
#     print(np.array(annot[key])[100])
# print(np.array(annot['bb_crop']))
# bb = annot['bb_crop']
# print(bb)
# print(type(bb))
# print(dir(bb))
# print(bb.len)
# # print(bb.size)
# # print(bb.value)
# # print(bb.ref)
# print(type(bb[0]))
# print(bb.shape)
# print(annot[bb[0]][:])

pose3d = annot['univ_annot3']
print(type(pose3d))
