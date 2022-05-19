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

img = loadmat('d:\hyperspectral_data\DFC2013_Houston.mat')['DFC2013_Houston']
print(img.shape)
gt = loadmat('d:\hyperspectral_data\DFC2013_Houston_gt.mat')['DFC2013_Houston_gt']
print(gt.shape)  #349,1905,144

img = img[:, :, 0:100]
img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )


[m, n, b] = img.shape
label_num = gt.max()  #最大为9，即除0外包括9类
PATCH_SIZE = 14

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

    #f = open(r'./label_HS.txt', 'w')
    data = []
    label = []

    for i in range(PATCH_SIZE, m - PATCH_SIZE): # PATCH_SIZE, 50, 100, 150, 200, 250, 300, 330, m - PATCH_SIZE
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                # count += 1
                temp_data = Patch(img, i, j, PATCH_SIZE)
                temp_label = np.zeros((1, label_num), dtype=np.int8)  # temp_label为一行九列[0,1,2,....,7,8]表示类别
                temp_label = gt[i, j] - 1

                data.append(temp_data)  # 每一行表示一个patch
                label.append(temp_label)
            #gt_index = ((i - PATCH_SIZE) * 1905 + j - PATCH_SIZE)  # 记录坐标，用于可视化分类预测结果
            #f.write(str(temp_label) + '\n')

    # print(count)  #42776

    data = np.array(data)
    print(data.shape)  # (9234, 1, 867)
    data = np.squeeze(data)
    print("squeeze : ", data.shape)  # squeeze :  (9234, 867)
    label = np.array(label)
    print(label.shape)  # (9234, 1, 9)
    label = np.squeeze(label)
    print("squeeze : ", label.shape)  # squeeze :  (9234, 9)

    # 测试数据集
    #f = h5py.File(r'./data/HS-17-17-100-test-all-8.h5', 'w')
    #f['data'] = data
    #f['label'] = label
    #f.close()
    
    indices = np.arange(data.shape[0])  # list [0,.....,42775]
    shuffled_indices = np.random.permutation(indices)
    images = data[shuffled_indices]
    labels = label[shuffled_indices]  # 打乱顺序

    y = labels  # 布尔索引
    # y 为一个list (42776,)  [0 1 6 ... 0 7 1]

    n_classes = y.max() + 1  # y.max() = 8  n_classes代表类别数为9
    t_labeled = []

    '''
    dict_k = {}
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] in range(0, 16):
                if gt[i][j] not in dict_k:
                    dict_k[gt[i][j]] = 0
                dict_k[gt[i][j]] += 1

    print(dict_k)  # {0: 164624, 1: 6631, 4: 3064, 2: 18649, 8: 3682, 5: 1345, 9: 947, 6: 5029, 3: 2099, 7: 1330}


    # 事先保存的样本情况
    dict_sample = {'0': 686136, '9': 1252, '1': 1251, '11': 1235, '2': 1254, '7': 1268, '4': 1244,
    '10': 1227, '8': 1244, '12': 1233, '5': 1242, '6': 325, '13': 469, '14': 428, '15': 660, '3': 697}
    '''

    for c in range(n_classes):  # 对于每一类     每类的pixel数量：6631 18649 2099 3064 1345 5029 1330 3682 947
        #if dict_sample[str(c+1)]<200:
        #    pass
        #else:
        i = indices[y == c][:200]
        t_labeled += list(i)  # 列表中元素增加


    # print(np.array(t_labeled).shape)  #(1620,)
    # print(np.array(v_labeled).shape)  #(180,)

    # 将其划分分训练和检验两个数据集
    t_images = images[t_labeled]
    print('t_images', t_images.shape)
    t_labels = labels[t_labeled]
    print('t_labels', t_labels.shape)

    # 训练数据集
    #f = h5py.File(r'./data/HS-17-17-100.h5', 'w')
    f = h5py.File(r'./data/HS-' + str(PATCH_SIZE * 2) + '-' + str(PATCH_SIZE * 2) + '-100.h5', 'w')  # 每类200个
    f['data'] = t_images
    f['label'] = t_labels
    f.close()





preparation()