# -*- coding: utf-8 -*-
"""
Created on 2019/01/11 14:08
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import warnings

warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import os
from dataset import TSNDataSet
from models import TSN
from transforms import *
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--data_split_type', action='store', default="cv2")
args = parser.parse_args()
workers = len(args.gpus)

data_split_type = args.data_split_type
val_list = "../tools/train_val_list/%s_val_list.txt" % data_split_type
pretrained_model = "./%s_Flow_num_seg7_dropout_08_flow_model_best.pth.tar" % data_split_type
num_segments = 7
modality = "Flow"
flow_prefix = "flow_"
arch = "BNInception"
consensus_type = 'avg'
batch_size = 1
num_class = 12
base_lr = 0.001
dev = 2


net = TSN(num_class, num_segments, modality,
          base_model=arch,
          consensus_type=consensus_type,
          dropout=0.8,
          extra_feature=False)
checkpoint = torch.load(pretrained_model)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
print('model load done..%s' % data_split_type)
workers = len(args.gpus)
train_augmentation = net.get_augmentation()
loader2 = torch.utils.data.DataLoader(
    TSNDataSet("", val_list, num_segments=num_segments,
               new_length=1 if modality == "RGB" else 5,
               modality=modality,
               image_tmpl="img_{:05d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix + "{}_{:05d}.jpg",
               transform=torchvision.transforms.Compose([
                   train_augmentation,
                   Stack(roll=arch == 'BNInception'),
                   ToTorchFormatTensor(div=arch != 'BNInception'),
                   GroupNormalize(net.input_mean, net.input_std),
               ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

val_data_gen = enumerate(loader2)
devices = [args.gpus[i] for i in range(workers)]
net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.eval()
val_list = open(val_list).readlines()
y_pred = []
y_true = []
for i, (data, label) in val_data_gen:
    p = val_list[i].split(' ')[0]
    print(p)
    input_var = torch.autograd.Variable(data)
    output = net(input_var)
    output = output.data
    _, pred = output.topk(1, 1, True, True)
    pred = pred.cpu().numpy().copy().reshape(-1)[0]
    y_pred.append(pred)
    y_true.append(label.cpu().numpy().copy()[0])
np.save("%s_y_pred.npy" % data_split_type, y_pred)
np.save("%s_y_true.npy" % data_split_type, y_true)
print(data_split_type)
