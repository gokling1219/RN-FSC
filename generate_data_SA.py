import numpy as np
import h5py
from scipy.io import loadmat


def Patch(data,height_index,width_index,PATCH_SIZE):   # PATCH_SIZE为一个patch（边长-1）的一半    data维度(H,W,C)
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    # 由height_index和width_index定位patch中心像素
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    # print(patch.shape)                  #为一行  (1, 243) 243 = 9*9*3
    return patch

import sys
seed_number = sys.argv[1]
np.random.seed(int(seed_number))

img = loadmat('D:\hyperspectral_data\Salinas_corrected.mat')['salinas_corrected']
gt = loadmat('D:\hyperspectral_data\Salinas_gt.mat')['salinas_gt']
# print(img.shape)  #(610, 340, 103)

# 只选取前100个波段
#band = list(range(0,200,2))
img = img[:, :, 0:100]

# 归一化
img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )

[m, n, b] = img.shape
label_num = gt.max()  #最大为9，即除0外包括9类
PATCH_SIZE = 14   #每一个patch边长大小为9

# padding the hyperspectral images
img_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE, b), dtype=np.float32)
img_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE), :] = img[:, :, :]

for i in range(PATCH_SIZE):
    img_temp[i, :, :] = img_temp[2 * PATCH_SIZE - i, :, :]
    img_temp[m + PATCH_SIZE + i, :, :] = img_temp[m + PATCH_SIZE - i - 2, :, :]

for i in range(PATCH_SIZE):
    img_temp[:, i, :] = img_temp[:, 2 * PATCH_SIZE - i, :]
    img_temp[:, n + PATCH_SIZE + i, :] = img_temp[:, n + PATCH_SIZE  - i - 2, :]

img = img_temp

gt_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE), dtype=np.int8)
gt_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE)] = gt[:, :]
gt = gt_temp

[m, n, b] = img.shape
# count = 0 #统计有多少个中心像素类别不为0的patch


def preparation():

    f = open('./data/gt_index_SA.txt', 'w')
    f1 = open('./data/SA_label.txt', 'w')
    data = []
    label = []

    for i in range(PATCH_SIZE, m - PATCH_SIZE):
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                # count += 1
                temp_data = Patch(img, i, j, PATCH_SIZE)
                #temp_label = np.zeros((1, label_num), dtype=np.int8)  # temp_label为一行九列[0,1,2,....,7,8]表示类别
                #temp_label[0, gt[i, j] - 1] = 1
                temp_label = gt[i, j] - 1 # 直接用0-8表示，不用独热编码
                data.append(temp_data)  # 每一行表示一个patch
                label.append(temp_label)
                gt_index = ((i - PATCH_SIZE) * 217 + j - PATCH_SIZE)  # 记录坐标，用于可视化分类预测结果
                f.write(str(gt_index) + '\n')
                f1.write(str(temp_label) + '\n')

    # print(count)  #42776

    data = np.array(data)
    print(data.shape)  # (42776, 1, 8100)
    data = np.squeeze(data)
    print("squeeze : ", data.shape)  # squeeze :  (42776, 8100)
    label = np.array(label)
    print(label.shape)  # (42776, 1, 9)
    label = np.squeeze(label)
    print("squeeze : ", label.shape)  # squeeze :  (42776, 9)
    print(np.unique(label)) # [0 1 2 3 4 5 6 7 8]
    #
    f = h5py.File('.\data\SA_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_test.h5', 'w')
    f['data'] = data
    f['label'] = label
    f.close()

    # 每类随机采样num_s个生成支撑样本集
    num_s = 5  # 支撑样本集数量
    indices = np.arange(data.shape[0])  # list [0,.....,42775]
    shuffled_indices = np.random.permutation(indices)
    data = data[shuffled_indices]
    label = label[shuffled_indices]
    data_s = []
    label_s = []

    for i in range(label_num): # 类别顺换 0123...8
        count = 0
        for j in range(54129): # 数量循环
            if label[j] == i and count <= num_s-1: # 如果标记为第i类
                data_s.append(data[j, :])
                label_s.append(label[j])
                count += 1
    data_s = np.array(data_s)
    label_s = np.array(label_s)
    print(data_s.shape)
    print(np.unique(label_s))
    print(label_s.shape)

    PATH = 'D:/PycharmProjects/02_RN/data/SA_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_support' + str(num_s) + '.h5'
    f = h5py.File(PATH, 'w')
    f['data_s'] = data_s # (45, 8100)
    f['label_s'] = label_s # (45, 9)
    f.close()


preparation()