# -*- coding: utf-8 -*-
"""
Created on 2019/01/11 14:08
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
from dataset import TSNDataSet
from models import TSN
from transforms import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser

optParser = OptionParser()
optParser.add_option('-t', '--train_list', action='store', type="string", dest='train_list',
                     help="train list's path", default="../tools/train_val_list/cs_train_list.txt")
optParser.add_option('-o', '--val_list', action='store', type="string", dest='val_list',
                     help="val list's path", default="../tools/train_val_list/cs_val_list.txt")
optParser.add_option('-m', '--pretrained_model', action='store', type="string", dest='pretrained_model',
                     help="pretrained model's path", default="./cv1_Flow_num_seg7_dropout_08_flow_model_best.pth.tar")

option, args = optParser.parse_args()
train_list = option.train_list
val_list = option.val_list
pretrained_model = option.pretrained_model
num_segments = 7
modality = "Flow"
flow_prefix = "flow_"
arch = "BNInception"
consensus_type = 'avg'
batch_size = 1
num_class = 12
base_lr = 0.001
dev = 2
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--gpus', nargs='+', type=int, default=None)
args = parser.parse_args()
workers = len(args.gpus)

net = TSN(num_class, num_segments, modality,
          base_model=arch,
          consensus_type=consensus_type,
          dropout=0.8,
          extra_feature=True)
checkpoint = torch.load(pretrained_model)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
print(net)
print(net.input_size, net.scale_size, net.input_mean, net.input_std)
print('model load done..cv1')
workers = len(args.gpus)
train_augmentation = net.get_augmentation()
loader = []
loader1 = torch.utils.data.DataLoader(
    TSNDataSet("", train_list, num_segments=num_segments,
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
loader.append(loader1)
loader.append(loader2)
val_data_gen = enumerate(loader2)
devices = [args.gpus[i] for i in range(workers)]
net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.eval()
val_list = open(val_list).readlines()
train_list = open(train_list).readlines()
for i, (data, label) in val_data_gen:
    p = val_list[i].split(' ')[0]
    print(p)
    input_var = torch.autograd.Variable(data)
    feature = net(input_var)
    feature = feature.data.cpu().numpy().copy()
    feature = feature.reshape(1024)
    out = p.replace('optical_flow_data', 'cv1_features')
    if not os.path.exists(out):
        os.makedirs(out)
    np.save(os.path.join(out, 'TSN-feature.npy'), feature)

data_gen = enumerate(loader1)
for i, (data, label) in data_gen:
    p = train_list[i].split(' ')[0]
    print(p)
    input_var = torch.autograd.Variable(data)
    feature = net(input_var)
    feature = feature.data.cpu().numpy().copy()
    feature = feature.reshape(1024)
    out = p.replace('optical_flow_data', 'cv1_features')
    if not os.path.exists(out):
        os.makedirs(out)
    np.save(os.path.join(out, 'TSN-feature.npy'), feature)
print('cv1')
