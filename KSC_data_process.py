import numpy as np
import h5py
from scipy.io import loadmat
from functools import reduce

def Patch(data,height_index,width_index,PATCH_SIZE):   # PATCH_SIZE为一个patch（边长-1）的一半    data维度(H,W,C)
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    # 由height_index和width_index定位patch中心像素
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    # print(patch.shape)                  #为一行  (1, 243) 243 = 9*9*3
    return patch

img = loadmat('D:\hyperspectral_data\KSC.mat')['KSC']
gt = loadmat('D:\hyperspectral_data\KSC_gt.mat')['KSC_gt']
print(img.shape)  #(1096, 715, 102)

# 统计每类样本所含个数
dict_k = {}
for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        if gt[i][j] in range(0,gt.max()+1):
            if gt[i][j] not in dict_k:
                dict_k[gt[i][j]]=0
            dict_k[gt[i][j]] +=1

print(dict_k) #{0: 649816, 9: 1252, 1: 1251, 11: 1235, 2: 1254, 7: 1268, 4: 1244, 10: 1227, 8: 1244, 12: 1233, 5: 1242, 6: 325, 13: 469, 14: 428, 15: 660, 3: 697}
# print(reduce(lambda x,y:x+y,dict_k.values())) #207400

img = img[:, :, :100]
img = (img * 1.0 - img.min()) / (img.max() - img.min())

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

label_num = gt.max()  #最大为9，即除0外包括9类
# print(label_num) # 9
data = []
label = []

count = 0 #统计有多少个类别不为0的pixel

#f = open('gt_index.txt', 'w')
for i in range(0, m):
    for j in range(0, n):
        if gt[i, j] == 0:
            continue
        else:
            count += 1
            temp_data = Patch(img, i, j, PATCH_SIZE)
            temp_label = np.zeros((1, label_num), dtype=np.int8)    #temp_label为一行九列[0,1,2,....,7,8]表示类别
            #temp_label[0, gt[i, j] - 1] = 1
            temp_label = gt[i, j] - 1 # 0-15
            data.append(temp_data)                # 每一行表示一个pixel
            label.append(temp_label)
            #gt_index = i * n + j            #  记录坐标，用于可视化分类预测结果
            #f.write(str(gt_index) + '\n')
            # print(i, j)

print(count)  #148152
data = np.array(data)
data = np.squeeze(data)
print(data.shape) #(148152, 100)

label = np.array(label)
print(label.shape) #(148152, 1, 9)
label = np.squeeze(label)
print("squeeze : ",label.shape) #squeeze :  (148152, 9)

# f = h5py.File('.\KSC-16-16-176-test.h5', 'w') # 全部标记样本！
# f['data'] = data
# f['label'] = label
# f.close()

##################################### 每类5/200个样本 #####################################

indices = np.arange(data.shape[0])  # list [0,.....,42775]
shuffled_indices = np.random.permutation(indices)
images = data[shuffled_indices]
labels = label[shuffled_indices]  # 打乱顺序
y = labels  # 布尔索引
print(y)
# y 为一个list (42776,)  [0 1 6 ... 0 7 1]

#n_classes = y.max() + 1  # y.max() = 8  n_classes代表类别数为9
s_labeled = []
#
# ################# 微调数据集 8-15每类5个 #################

sample = {'8': 431, '12': 503, '13': 927, '11': 419, '5': 161, '1': 761, '4': 252, '6': 229, '2': 243, '3': 256, '10': 404, '7': 105, '9': 520}
for c in range(y.max() + 1):  # 对于每一类     每类的pixel数量：6631 18649 2099 3064 1345 5029 1330 3682 947
    if sample[str(c + 1)] < 200:
        pass
    else:
        i = indices[y == c][:200]
        s_labeled += list(i)  # 列表中元素增加



s_images = images[s_labeled]
print('s_images', s_images.shape)
s_labels = labels[s_labeled]
print('s_labels', s_labels.shape)

f = h5py.File(r'./data/KSC-' + str(PATCH_SIZE*2) + '-' + str(PATCH_SIZE*2) + '-100.h5', 'w') # 每类200个
f['data'] = s_images # (1800, 100)
f['label'] = s_labels # (1800, 9)
f.close()