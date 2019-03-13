# -*- coding: utf-8 -*-
"""
Created on 2018/12/04 12:08
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
# st-gcn
import warnings

warnings.filterwarnings("ignore")
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from st_gcn import Model
from gendata import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import shutil
import os


class Basic_block(nn.Module):
    def __init__(self, in_feature=400, hidden=128, out_feature=12):
        super().__init__()
        self.fcn1 = nn.Linear(in_feature, out_feature)
        self.relu1 = nn.PReLU(out_feature)
        self.fcn2 = nn.Linear(out_feature, hidden)
        self.relu2 = nn.PReLU(hidden)
        self.fcn3 = nn.Linear(hidden, out_feature)
        self.relu3 = nn.PReLU(out_feature)
        self.drop_out = nn.Dropout(0.4)

    def forward(self, x):
        x1 = self.drop_out(self.relu1(self.fcn1(x)))
        x = self.drop_out(self.relu2(self.fcn2(x1)))
        x = self.drop_out(self.relu3(self.fcn3(x))) + x1
        return x


class Model_to_train(nn.Module):
    def __init__(self, in_feature=400, class_num=12):
        super().__init__()
        self.in_feature = in_feature
        self.class_num = class_num
        self.b = []
        self.b1 = Basic_block(self.in_feature, 128, 256)
        self.b2 = Basic_block(256, 128, 512)
        self.b3 = Basic_block(512, 128, 1024)
        self.out = nn.Linear(1024, self.class_num)
        self.b.append(self.b1)
        self.b.append(self.b2)
        self.b.append(self.b3)
        self.b.append(self.out)

    def forward(self, x):
        for i, f in enumerate(self.b):
            if i == len(self.b) - 1:
                continue
            x = f(x)
        x = x.view(x.size(0), -1)
        return x


def load_data(data_split_type):
    return get_all_data_from_txt('./train_val_list/%s_train_list.txt' % data_split_type,
                                 './train_val_list/%s_val_list.txt' % data_split_type, flist=True)


if __name__ == "__main__":
    data_split_type = 'cv2'
    print('generate feature.', data_split_type)
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    PATH = "/home/dongqian/code/stgcn/models/kinetics-st_gcn.pt"
    net1 = Model(in_channels=3, num_class=400, edge_importance_weighting=True,
                 graph_args=graph_args)
    net1.load_state_dict(torch.load(PATH))
    dev = "cuda:1"
    net1 = net1.to(dev)
    net2 = torch.load("mydataset_%s_model_best.pkl" % data_split_type, map_location=dev)
    print('model load success')
    net1.eval()
    net2.eval()
    Train_x, train_list, Test_x, val_list = load_data(data_split_type)
    for i, x in enumerate(Train_x):
        x = np.array([x])
        data = torch.from_numpy(x)
        data = data.float().to(dev)

        output = net1(data)
        feature = net2(output)
        feature = feature.data.cpu().numpy().copy().reshape(1024)
        print(train_list[i])
        if not os.path.exists(train_list[i].replace('keypoints', '%s_features' % data_split_type)):
            os.makedirs(train_list[i].replace('keypoints', '%s_features' % data_split_type))
        out = os.path.join(train_list[i].replace('keypoints', '%s_features' % data_split_type), 'st-gcn-feature.npy')
        np.save(out, feature)
    for i, x in enumerate(Test_x):
        x = np.array([x])
        data = torch.from_numpy(x)
        data = data.float().to(dev)

        output = net1(data)
        feature = net2(output)
        feature = feature.data.cpu().numpy().copy().reshape(1024)
        print(val_list[i])
        if not os.path.exists(val_list[i].replace('keypoints', '%s_features' % data_split_type)):
            os.makedirs(val_list[i].replace('keypoints', '%s_features' % data_split_type))
        out = os.path.join(val_list[i].replace('keypoints', '%s_features' % data_split_type), 'st-gcn-feature.npy')
        np.save(out, feature)
