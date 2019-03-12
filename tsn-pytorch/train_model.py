import warnings

warnings.filterwarnings("ignore")
import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
train_list = "./dq_tools/train_val_list/cs_train_list.txt"
val_list = "./dq_tools/train_val_list/cs_val_list.txt"
pretrained_model="./pretrained_model/kinetics_rgb.pth"
num_segments = 3
modality = "RGB"
flow_prefix = "flow_"
arch = "BNInception"
consensus_type = 'avg'
batch_size = 64
num_class = 400
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
          dropout=0.7,
          extra_feature=False)

checkpoint = torch.load(pretrained_model)
count = 0
base_dict = {}
for k, v in checkpoint.items():
    count = count + 1
    print(count, k)
    if 415 > count > 18:
        base_dict.setdefault(k[7:], checkpoint[k])
    if count < 19:
        base_dict.setdefault(k, checkpoint[k])
# base_dict.setdefault('new_fc.weight', checkpoint['base_model.fc-action.1.weight'])
# base_dict.setdefault('new_fc.bias', checkpoint['base_model.fc-action.1.bias'])
base_dict.setdefault('new_fc.weight', checkpoint['base_model.fc_action.1.weight'])
base_dict.setdefault('new_fc.bias', checkpoint['base_model.fc_action.1.bias'])

net.load_state_dict(base_dict)
print(net.input_size, net.scale_size, net.input_mean, net.input_std)
devices = [args.gpus[i] for i in range(workers)]
train_augmentation = net.get_augmentation()
net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.eval()
print(net)
cropping = torchvision.transforms.Compose([
    GroupOverSample(224, 256)
])



def extract_feature(video_data):
    data = video_data
    input_var = torch.autograd.Variable(data, volatile=False)
    feature = net(input_var)
    return feature.view(feature.size(0), -1)


class Basic_block(nn.Module):
    def __init__(self, in_feature=400, hidden=128, out_feature=12):
        super().__init__()
        self.fcn1 = nn.Linear(in_feature, out_feature)
        self.relu1 = nn.PReLU(out_feature)
        self.fcn2 = nn.Linear(out_feature, hidden)
        self.relu2 = nn.PReLU(hidden)
        self.fcn3 = nn.Linear(hidden, out_feature)
        self.relu3 = nn.PReLU(out_feature)
        self.drop_out = nn.Dropout(0.5)

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
        # self.b1 = Basic_block(self.in_feature, 512, 256)
        # self.b2 = Basic_block(256, 128, 512)
        # self.b3 = Basic_block(512, 128, 64)
        # self.b4 = Basic_block(64, 128, 256)
        # self.out = nn.Linear(256, self.class_num)
        # self.b.append(self.b1)
        # self.b.append(self.b2)
        # self.b.append(self.b3)
        # self.b.append(self.b4)
        # self.b.append(self.out)
        self.fcn1 = nn.Linear(self.in_feature, 512)
        self.relu1 = nn.PReLU(512)
        self.drop_out = nn.Dropout(0.5)
        self.fcn2 = nn.Linear(512, 12)
        self.b.append(self.fcn1)
        self.b.append(self.relu1)
        self.b.append(self.drop_out)
        self.b.append(self.fcn2)

    def forward(self, x):
        for f in self.b:
            x = f(x)
        x = x.view(x.size(0), -1)
        return x


model = Model_to_train()
model = model.cuda(dev)
loss_f = nn.CrossEntropyLoss().cuda(dev)
optimizer = optim.Adam(
                model.parameters(),
                lr=base_lr,
                weight_decay=0.0001)

epochs = []
all_loss = []
all_acc = []
train_loss = []
train_acc = []
f, axs = plt.subplots(4, 1, figsize=(10, 5))
for epoch in range(5000):
    training_losses = []
    training_accs = []
    model.train()
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", train_list, num_segments=num_segments,
                   new_length=1 if modality == "RGB" else 5,
                   modality=modality,
                   image_tmpl="img{:04d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix + "{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=arch == 'BNInception'),
                       ToTorchFormatTensor(div=arch != 'BNInception'),
                       GroupNormalize([104, 117, 128], [1]),
                   ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", val_list, num_segments=num_segments,
                   new_length=1 if modality == "RGB" else 5,
                   modality=modality,
                   image_tmpl="img{:04d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix + "{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=arch == 'BNInception'),
                       ToTorchFormatTensor(div=arch != 'BNInception'),
                       GroupNormalize([104, 117, 128], [1]),
                   ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    data_gen = enumerate(train_loader)
    val_data_gen = enumerate(val_loader)
    for i, (data, label) in data_gen:
        feature = extract_feature(data)
        data = feature.float().cuda(dev)
        label = label.long().cuda(dev)
        label = torch.autograd.Variable(label)
        output = model(data)
        loss = loss_f(output, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = output.data.topk(1, 1, True, True)
        pred = pred.t()
        acc = (pred == label.data).cpu().sum() * 100 / len(data)
        # statistics
        training_losses.append(loss.data.cpu().numpy()[0])
        training_accs.append(acc)
        print("epoch:", epoch, "step:", i, "/", len(train_loader), 'loss', loss.data.cpu().numpy()[0],
              'lr {:.4f}'.format(base_lr),
              "acc:%.1f" % acc + '%')
    losses = []
    accs = []
    model.eval()
    for i, (data, label) in val_data_gen:
        feature = extract_feature(data)
        data = feature.float().cuda(dev)
        label = label.long().cuda(dev)
        label = torch.autograd.Variable(label)
        output = model(data)
        loss = loss_f(output, label)

        _, pred = output.data.topk(1, 1, True, True)
        pred = pred.t()
        acc = (pred == label.data).cpu().sum() * 100 / len(data)
        # statistics
        losses.append(loss.data.cpu().numpy()[0])
        accs.append(acc)
        print("val epoch:", epoch, "step:", i, "/", len(val_loader), 'loss', loss.data.cpu().numpy()[0],
              'lr {:.4f}'.format(base_lr),
              "acc:%.1f" % acc + '%')
    epochs.append(epoch)
    all_loss.append(np.mean(losses))
    all_acc.append(np.mean(accs))
    train_loss.append(np.mean(training_losses))
    train_acc.append(np.mean(training_accs))
    axs[0].plot(epochs, all_loss, c='b', marker='.', label='val loss')
    axs[1].plot(epochs, all_acc, c='r', marker='.', label='val acc')
    axs[2].plot(epochs, train_loss, c='b', marker='.', label='train_loss')
    axs[3].plot(epochs, train_acc, c='r', marker='.', label='train_acc')
    if epoch == 0:
        for i in range(4):
            axs[i].legend(loc='best')
    plt.pause(0.000001)
    plt.savefig('./figs/%s.jpg' % str(epoch).zfill(5))
    print('done')
