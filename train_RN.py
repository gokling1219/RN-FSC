import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time
#from thop import profile

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 512)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 128)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 20)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class
parser.add_argument("-e","--episode",type = int, default= 10000)
#-----------------------------------------------------------------------------------#
#parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
#FEATURE_DIM = args.feature_dim
#RELATION_DIM = args.relation_dim
n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 200  # 训练数据集中每类200个样本
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(1,8,kernel_size=3,padding=1),
            # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
                        nn.BatchNorm3d(8),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))

        self.layer2 = nn.Sequential(
                        nn.Conv3d(8,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer3 = nn.Sequential(
                        nn.Conv3d(16,32,kernel_size=3,padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(32,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU())



    def forward(self,x):
        out = self.layer1(x)
        #print(list(out.size()))  # [20, 8, 51, 15, 15]
        out = self.layer2(out)
        #print(list(out.size()))  # [20, 16, 13, 8, 8]
        out = self.layer3(out)
        #print(list(out.size()))  # [20, 32, 3, 5, 5]
        out = self.layer4(out)
        #print(list(out.size()))  # [20, 64, 3, 5, 5]
        #out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        #print(list(out.size())) # [100, 32, 6, 3, 3]
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(256, 128,kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())


        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p = 0.5)                                                                              # 测试的时候需要修改....？？？

    def forward(self,x): # [7600, 128, 2, 2]
        out = self.layer1(x)
        #print(list(out.size())) # [7600, 128, 3, 3]
        out = self.layer2(out)
        #out = self.layer3(out)
        #print(list(out.size())) # [7600, 64, 2, 2]
        out = out.view(out.size(0),-1) # flatten
        #print(list(out.size())) # [7600, 256]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.sigmoid(self.fc2(out))
        #print("ssss", list(out.size())) # [6000, 1]
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def train(im_width, im_height, depth):


    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    
    feature_encoder.train()
    relation_network.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=5000, gamma=0.5)
    # 每过step_size次,更新一次学习率;每经过100000次，学习率折半
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=5000, gamma=0.5)



    # 训练数据集
    f = h5py.File(r'D:\PycharmProjects\02_RN\data\meta_train_11000_78400.h5', 'r')
    train_dataset = f['data'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth)  # 划分成了78类，每类200个样本
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    print("train_dataset.shape", train_dataset.shape) # (78, 200, 1, 100, 9, 9) #
    n_train_classes = train_dataset.shape[0]


    accuracy_ = []
    loss_ = []
    a = time.time()
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # start:每一个episode的采样过程##########################################################################################
        epi_classes = np.random.permutation(n_train_classes)[:n_way]  # 在78个数里面随机抽取前20个 78为类别数量 随机抽取20个类别，例如15 69 23 ....
        support = np.zeros([n_way, n_shot, 1, depth, im_height, im_width], dtype=np.float32)  # n_shot = 5
        query = np.zeros([n_way, n_query,  1, depth, im_height, im_width], dtype=np.float32)  # n_query= 15
        # (N,C_in,D_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, 1, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, 1, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        #print(labels)
        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.LongTensor(labels)
        # end:每一个episode的采样过程##########################################################################################

        # calculate features
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
        # flops, params = profile(feature_encoder, inputs=(Variable(support_tensor).cuda(GPU),))
        # print('feature_encoder flops', flops, 'params', params)
        #print( list(sample_features.size()) ) # [100, 32, 6, 3, 3]
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4], list(sample_features.size())[-3],
                                               list(sample_features.size())[-2], list(sample_features.size())[
                                                   -1])  # view函数改变shape: 5way, 5shot, 64, 19, 19
        #sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
        sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
        #print( list(sample_features.size()) ) # [20, 32, 6, 3, 3]
        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  # 20x64*5*5
        #print(list(batch_features.size())) # [300, 32, 6, 3, 3]

        ################################################################################################################
        sample_features = sample_features.view(n_way, list(sample_features.size())[1]*list(sample_features.size())[2],
                                               list(sample_features.size())[-2], list(sample_features.size())[-1])
        batch_features = batch_features.view(n_way*n_query, list(batch_features.size())[1] * list(batch_features.size())[2],
                                               list(batch_features.size())[-2], list(batch_features.size())[-1])
        #print(list(sample_features.size())) # [20, 128, 5, 5]
        #print(list(batch_features.size())) # [380, 128, 5, 5]
        ################################################################################################################

        # calculate relations
        # 支撑样本和查询样本进行连接
        #print('relation_pairs.size() = ',list(relation_pairs.size()))  # [6000, 384, 3, 3]


        sample_features_ext = sample_features.repeat(n_query * n_way, 1, 1, 1, 1)  # # repeat函数沿着指定的维度重复tensor
        #print(list(sample_features_ext.size())) # [380, 20, 128, 5, 5]
        batch_features_ext = batch_features.repeat(n_way, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        #print(list(batch_features_ext.size())) # [380, 20, 128, 5, 5]

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        #print(list(relation_pairs.size())) # [380, 20, 256, 5, 5]
        relation_pairs = relation_pairs.view(-1,  list(relation_pairs.size())[-3], list(relation_pairs.size())[-2], list(relation_pairs.size())[-1])
        #print(list(relation_pairs.size())) # [7600, 256, 5, 5]



        relations = relation_network(relation_pairs)
        # flops, params = profile(relation_network, inputs=(relation_pairs,))
        # print('relation_network flops', flops, 'params', params)
        #print(list(relations.size())) # [6000, 1]
        relations = relations.view(-1, n_way)
        #print(list(relations.size())) # [300, 20]

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(n_query * n_way, n_way).scatter_(dim=1, index=label_tensor.view(-1, 1), value=1).cuda(GPU))
        # scatter中1表示按照行顺序进行填充，labels_tensor.view(-1,1)为索引，1为填充数字
        loss = mse(relations, one_hot_labels)

        # training
        # 把模型中参数的梯度设为0
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        # 进行单次优化，参数更新
        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%1 == 0:
            print("episode:",episode+1,"loss",loss)
            #################调试#################
            _, predict_label = torch.max(relations.data, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            #print(predict_label)
            #print(labels)
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            # print(rewards)
            total_rewards = np.sum(rewards)
            # print(total_rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)
            accuracy_.append(accuracy)
            loss_.append(loss.item())
    print('time = ',time.time()-a)



    torch.save(feature_encoder.state_dict(),str('./model/meta_training_feature_encoder_' +str(n_way) + 'way_' + str(n_shot) + 'shot_newmodel.pkl'))
    torch.save(relation_network.state_dict(),str('./model/meta_training_relation_network_' +str(n_way) + 'way_' + str(n_shot) + 'shot_newmodel.pkl'))


    f = open('./result/meta_training_loss_' +str(n_way) + 'way_' + str(n_shot) + 'shot.txt', 'w')
    for i in range(np.array(loss_).shape[0]):
        f.write(str(loss_[i]) + '\n')
    f = open('./result/meta_training_accuracy_' +str(n_way) + 'way_' + str(n_shot) + 'shot.txt', 'w')
    for i in range(np.array(accuracy_).shape[0]):
        f.write(str(accuracy_[i]) + '\n')



if __name__ == '__main__':
    train(im_width, im_height, depth)



"""Pytorch中神经网络模块化接口nn的了解"""
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)

    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.

    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：

    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28

"""