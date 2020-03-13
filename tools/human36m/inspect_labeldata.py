import numpy as np

labels = np.load('human36m-multiview-labels-GTbboxes.npy', allow_pickle=True).item()
FRAMENUM = len(labels['table'])
print('FRAMENUM:', FRAMENUM)
# exit()

subj_idx = 0
act_idx = 0
count = 0

data_list = []
subj_list = []
act_list = []
print(count)
count = 0
i = 0
print(f'******{i:>8d}', end = ' | ')
# print(labels['table'][i])
# print(labels['subject_names'][labels['table'][i]['subject_idx']], end = ' | ')
# print(labels['action_names'][labels['table'][i]['action_idx']], end = ' | ')
print(labels['table'][i]['subject_idx'], end = ' | ')
print(labels['table'][i]['action_idx'], end = ' | ')
print(labels['table'][i]['frame_idx'], end = ' | ')
subj_idx = labels['table'][i]['subject_idx']
act_idx = labels['table'][i]['action_idx']
for i in range(FRAMENUM):
    # if labels['table'][i]['subject_idx'] != subj_idx or labels['table'][i]['action_idx'] != act_idx:
    if labels['table'][i]['action_idx'] != act_idx:
        subj_list.append(act_list)
        act_list = []
        act_idx = labels['table'][i]['action_idx']

        print(count)
        count = 0
        print(f'******{i:>8d}', end = ' | ')
        # print(labels['table'][i])
        print(labels['subject_names'][labels['table'][i]['subject_idx']], end = ' | ')
        print(labels['action_names'][labels['table'][i]['action_idx']], end = ' | ')
        print(labels['table'][i]['frame_idx'], end = ' | ')
    if labels['table'][i]['subject_idx'] != subj_idx:
        data_list.append(subj_list)
        subj_list = []
        subj_idx = labels['table'][i]['subject_idx']

    act_list.append(labels['table'][i]['frame_idx'])
    
    count += 1

subj_list.append(act_list)
data_list.append(subj_list)
print(count)

for i in range(len(data_list)):
    for j in range(len(data_list[i])):
        print(i, '-', j, '-', len(data_list[i][j]))

np.save('learnable_triangulation_train_list.npy', data_list)

# # i = 0
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])
# i = 100
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])
# i = 1000
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])
# i = 10000
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])
# i = 100000
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])
# i = 1000000
# print(f'******{i}')
# print(labels['table'][i])
# print(labels['table'][i]['subject_idx'])
# print(labels['table'][i]['action_idx'])
# print(labels['table'][i]['frame_idx'])


# for i in range(0, 1000, 20):
#     print(i, labels['table'][i].item()[0])