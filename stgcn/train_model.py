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
        # model_1
        # self.b1 = Basic_block(self.in_feature, 128, 256)
        # self.b2 = Basic_block(256, 128, 512)
        # self.b3 = Basic_block(512, 128, 64)
        # self.b4 = Basic_block(64, 128, 256)
        # self.out = nn.Linear(256, self.class_num)
        # self.b.append(self.b1)
        # self.b.append(self.b2)
        # self.b.append(self.b3)
        # self.b.append(self.b4)
        # self.b.append(self.out)

        # model_2
        self.b1 = Basic_block(self.in_feature, 128, 256)
        self.b2 = Basic_block(256, 128, 512)
        self.b3 = Basic_block(512, 128, 1024)
        self.out = nn.Linear(1024, self.class_num)
        self.b.append(self.b1)
        self.b.append(self.b2)
        self.b.append(self.b3)
        self.b.append(self.out)

    def forward(self, x):
        for f in self.b:
            x = f(x)
        x = x.view(x.size(0), -1)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, file_name):
    filename = '_'.join(("mydataset", file_name, "model.pkl"))
    torch.save(state['state_dict'], filename)
    if is_best:
        best_name = '_'.join(("mydataset", file_name, 'model_best.pkl'))
        shutil.copyfile(filename, best_name)


class REC_train():
    """
        Processor for Skeleton-based Action Recgnition
    """

    #             m1   m2
    # 128 0.0001 72%  75%
    # 64  0.0001 75%  75%
    # 128 0.001  75%  75%
    # 64  0.001  75%  75%

    def __init__(self, data_set_split_type, optimizer_type="Adam", base_lr=0.0001):
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
        PATH = "models/kinetics-st_gcn.pt"
        self.data_set_split_type = data_set_split_type
        self.bsize = 128
        self.model = Model(in_channels=3, num_class=400, edge_importance_weighting=True,
                           graph_args=graph_args)
        self.model.load_state_dict(torch.load(PATH))
        self.dev = "cuda:%d" % ngpu
        self.model = self.model.to(self.dev)
        self.model.eval()
        # self.model.apply(weights_init)
        self.model_to_train = Model_to_train()
        self.model_to_train = self.model_to_train.to(self.dev)

        self.load_data()
        self.base_lr = base_lr
        self.lr = base_lr
        self.optimizer_type = optimizer_type
        self.load_optimizer()
        self.step = [20, 30, 40, 50]
        self.meta_info = dict(epoch=0, iter=0)
        self.loss = nn.CrossEntropyLoss()

    def train_model_trans(self):
        f, axs = plt.subplots(4, 1, figsize=(10, 5))
        epochs = []
        losses = []
        accs = []
        train_losses = []
        train_accs = []
        self.X_test = torch.from_numpy(self.Test_x)
        self.y_test = torch.from_numpy(self.Test_y)
        best_acc = -1
        for epoch in range(40):
            self.meta_info['epoch'] = epoch
            print('epoch: ', epoch)
            # training
            self.model_to_train.train()
            training_losses = []
            training_accs = []
            j = 0
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.Train_x, self.Train_y,
                                                                                  test_size=0.2,
                                                                                  random_state=np.random.randint(100))
            self.X_train = torch.from_numpy(self.X_train)
            self.X_val = torch.from_numpy(self.X_val)
            self.y_train = torch.from_numpy(self.y_train)
            self.y_val = torch.from_numpy(self.y_val)
            for i in range(len(self.X_train) // self.bsize):
                data = self.X_train[i * self.bsize:i * self.bsize + self.bsize]
                label = self.y_train[i * self.bsize:i * self.bsize + self.bsize]
                # get data
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                # forward
                output = self.model(data)
                output = self.model_to_train(output)
                loss = self.loss(output, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = output.argmax(dim=1)
                acc = (pred == label.data).cpu().sum() * 100 / self.bsize
                # statistics
                loss_val, acc_val = self.train_model_val()
                training_losses.append(loss_val)
                training_accs.append(acc_val)
                print('epoch:', epoch, "step:", i, "/", len(self.X_train) // self.bsize, "- %d" % j, 'loss',
                      loss.data.item(),
                      "acc:%.1f" % acc + '%', 'val acc:%.1f' % acc_val + '%',
                      "batch size:%d lr:%.4f" % (self.bsize, self.lr))

            train_loss, train_acc = np.mean(training_losses), np.mean(training_accs)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print('Eval epoch: {}'.format(epoch))
            loss, acc = self.train_model_test()

            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                'state_dict': self.model_to_train,
                'best_acc': best_acc,
            }, is_best, file_name=self.data_set_split_type)

            epochs.append(epoch)
            losses.append(loss)
            accs.append(acc)
            axs[0].plot(epochs, losses, c='b', marker='.', label='val loss')
            axs[1].plot(epochs, accs, c='r', marker='.', label='val acc')
            axs[2].plot(epochs, train_losses, c='b', marker='.', label='train_loss')
            axs[3].plot(epochs, train_accs, c='r', marker='.', label='train_acc')
            if epoch == 0:
                for i in range(4):
                    axs[i].legend(loc='best')
            plt.pause(0.000001)
            if not os.path.exists('./%s_figs' % self.data_set_split_type):
                os.makedirs('./%s_figs' % self.data_set_split_type)
            plt.savefig('./%s_figs/%s.png' % (self.data_set_split_type, str(epoch).zfill(5)))
            print('Done.')
        np.save('train_losses.npy', train_losses)
        np.save('train_accs.npy', train_accs)
        np.save('losses.npy', losses)
        np.save('accs.npy', accs)

    def train_model_val(self):
        self.model_to_train.eval()
        losses = []
        accs = []
        for _ in range(5):
            i = random.randrange(len(self.X_val) // self.bsize)
            data = self.X_val[i * self.bsize:i * self.bsize + self.bsize]
            label = self.y_val[i * self.bsize:i * self.bsize + self.bsize]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            output = self.model_to_train(output)
            loss = self.loss(output, label)
            pred = output.argmax(dim=1)
            acc = (pred == label.data).cpu().sum() * 100 / len(data)
            losses.append(loss.data.item())
            accs.append(acc)
        return np.mean(losses), np.mean(accs)

    def train_model_test(self):
        self.model_to_train.eval()
        losses = []
        accs = []
        for i in range(len(self.X_test) // self.bsize):
            data = self.X_test[i * self.bsize:i * self.bsize + self.bsize]
            label = self.y_test[i * self.bsize:i * self.bsize + self.bsize]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            output = self.model_to_train(output)
            loss = self.loss(output, label)
            pred = output.argmax(dim=1)
            acc = (pred == label.data).cpu().sum() * 100 / len(data)
            print('eval loss:', loss.data.item(), "acc:%.1f" % acc + '%')
            losses.append(loss.data.item())
            accs.append(acc)
        print('eval loss:', np.mean(losses), "acc:%.1f" % np.mean(accs) + '%')
        return np.mean(losses), np.mean(accs)

    def load_optimizer(self):
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001)
        elif self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                self.model_to_train.parameters(),
                lr=self.base_lr,
                weight_decay=0.0001)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.optimizer == 'SGD' and self.step:
            lr = self.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.base_lr

    def load_data(self):
        self.Train_x, self.Train_y, self.Test_x, self.Test_y = get_all_data_from_txt(
            '../tools/train_val_list/%s_train_list.txt' % self.data_set_split_type,
            '../tools/train_val_list/%s_val_list.txt' % self.data_set_split_type)


if __name__ == "__main__":
    from optparse import OptionParser

    optParser = OptionParser()
    optParser.add_option('-t', '--data_split_type', action='store', type="string", dest='data_split_type',
                         help="data_split_type", default="cv2")
    optParser.add_option('-n', '--number_of_gpu', action='store', type="int", dest='number_of_gpu',
                         help="the number of used gpu", default=0)
    option, args = optParser.parse_args()
    data_split_type = option.data_split_type
    ngpu = option.number_of_gpu
    rec_train = REC_train(data_set_split_type=data_split_type)
    rec_train.train_model_trans()
    print(data_split_type)
