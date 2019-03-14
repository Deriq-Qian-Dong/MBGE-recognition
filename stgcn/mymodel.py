import warnings

warnings.filterwarnings("ignore")
import os
import sys
import torch
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_data(snippets_dir, label, label_index):
    data_numpy = np.concatenate([np.load(os.path.join(snippets_dir, 'st-gcn-feature.npy')),
                                 np.load(os.path.join(snippets_dir, 'TSN-feature.npy'))])
    return data_numpy, label_index


def get_all_data_from_txt(t, v, data_split_type, flist=False):
    ft_lines = open(t).readlines()
    fv_lines = open(v).readlines()
    root_path = "/data5/dongqian/%s_features" % data_split_type
    ft_lines = [line.strip().replace("/data/zqs/data/optical_flow_data", root_path).split(' ') for line in
                ft_lines]
    fv_lines = [line.strip().replace("/data/zqs/data/optical_flow_data", root_path).split(' ') for line in
                fv_lines]
    X_train = []
    y_train = []
    cnt = 0
    all_num = len(ft_lines) + len(fv_lines)
    train_list = []
    for line in ft_lines:
        npys = os.listdir(line[0])
        if len(npys) != 2:
            print(line)
            continue
        data_numpy, label = get_data(line[0], line[0].split('/')[-1], int(line[-1]))
        if len(data_numpy):
            train_list.append(line[0])
        X_train.append(data_numpy)
        y_train.append(label)
        cnt += 1
        # print(cnt, all_num)
        sys.stdout.write('\r')
        sys.stdout.write(
            "data loading... current-%d all-%d %s%% |%s" % (
                cnt, all_num, int(100 * cnt / all_num), int(100 * cnt / all_num) * '#'))
        sys.stdout.flush()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = []
    y_test = []
    val_list = []
    for line in fv_lines:
        npys = os.listdir(line[0])
        if len(npys) != 2:
            continue
        data_numpy, label = get_data(line[0], line[0].split('/')[-1], int(line[-1]))
        if len(data_numpy):
            val_list.append(line[0])
        X_test.append(data_numpy)
        y_test.append(label)
        cnt += 1
        # print(cnt, all_num)
        sys.stdout.write('\r')
        sys.stdout.write(
            "data loading... current-%d all-%d %s%% |%s" % (
                cnt, all_num, int(100 * cnt / all_num), int(100 * cnt / all_num) * '#'))
        sys.stdout.flush()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    if not flist:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, train_list, X_test, val_list


class Basic_block(nn.Module):
    def __init__(self, in_feature, hidden, out_feature):
        super().__init__()
        self.fcn1 = nn.Linear(in_feature, out_feature)
        self.relu1 = nn.PReLU(out_feature)
        self.fcn2 = nn.Linear(out_feature, hidden)
        self.relu2 = nn.PReLU(hidden)
        self.fcn3 = nn.Linear(hidden, out_feature)
        self.relu3 = nn.PReLU(out_feature)
        self.drop_out = nn.Dropout(0.9)

    def forward(self, x):
        x1 = self.drop_out(self.relu1(self.fcn1(x)))
        x = self.drop_out(self.relu2(self.fcn2(x1)))
        x = self.drop_out(self.relu3(self.fcn3(x))) + x1
        return x


class Model(nn.Module):
    def __init__(self, in_feature=2048, class_num=12):
        super().__init__()
        self.in_feature = in_feature
        self.class_num = class_num
        # model 1 87%
        self.b = []
        self.b1 = Basic_block(self.in_feature, 128, 512)
        self.b2 = Basic_block(512, 128, 64)
        self.b3 = Basic_block(64, 128, 32)
        self.out = nn.Linear(32, self.class_num)
        self.b.append(self.b1)
        self.b.append(self.b2)
        self.b.append(self.b3)
        self.b.append(self.out)

    def forward(self, x):
        for f in self.b:
            x = f(x)
        x = x.view(x.size(0), -1)
        return x


class REC_train():
    def __init__(self, data_split_type, base_lr=1e-4):
        self.data_split_type = data_split_type
        self.bsize = 128
        self.dev = "cuda:%d" % ngpu
        self.model = Model()
        self.model = self.model.to(self.dev)
        self.load_data()
        self.lr = base_lr
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        f, axs = plt.subplots(4, 1, figsize=(10, 5))
        epochs = []
        losses = []
        accs = []
        train_losses = []
        train_accs = []
        self.X_test = torch.from_numpy(self.Test_x)
        self.y_test = torch.from_numpy(self.Test_y)
        best_acc = -1
        for epoch in range(100):
            if epoch == 20:
                self.lr = 1e-5
            print('epoch: ', epoch)
            # training
            self.model.train()
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
                loss = self.loss(output, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = output.argmax(dim=1)
                acc = (pred == label.data).cpu().sum() * 100 / self.bsize
                # statistics
                loss_val, acc_val = self.eval()
                training_losses.append(loss_val)
                training_accs.append(acc_val)
                print('epoch:', epoch, "step:", i, "/", len(self.X_train) // self.bsize, "- %d" % j, 'loss',
                      loss.data.item(),
                      "acc:%.1f" % acc + '%', 'val acc:%.1f' % acc_val + '%',
                      "batch size:%d lr:%.5f" % (self.bsize, self.lr))

            train_loss, train_acc = np.mean(training_losses), np.mean(training_accs)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print('Eval epoch: {}'.format(epoch))
            if not os.path.exists("./%s_mfigs" % self.data_split_type):
                os.makedirs("./%s_mfigs" % self.data_split_type)
            loss, acc = self.test()

            # is_best = acc > best_acc
            # best_acc = max(acc, best_acc)
            # save_checkpoint({
            #     'state_dict': self.model_to_train,
            #     'best_acc': best_acc,
            # }, is_best)

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
            plt.title("%s_fusion" % self.data_split_type)
            plt.pause(0.000001)
            plt.savefig('./%s_mfigs/%s.png' % (self.data_split_type, str(epoch).zfill(5)))
            print('Done.')
        # np.save('train_losses.npy', train_losses)
        # np.save('train_accs.npy', train_accs)
        # np.save('losses.npy', losses)
        # np.save('accs.npy', accs)

    def eval(self):
        self.model.eval()
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
            loss = self.loss(output, label)
            pred = output.argmax(dim=1)
            acc = (pred == label.data).cpu().sum() * 100 / len(data)
            losses.append(loss.data.item())
            accs.append(acc)
        return np.mean(losses), np.mean(accs)

    def test(self):
        self.model.eval()
        losses = []
        accs = []
        y_pred = []
        y_true = []
        for i in range(len(self.X_test) // self.bsize):
            data = self.X_test[i * self.bsize:i * self.bsize + self.bsize]
            label = self.y_test[i * self.bsize:i * self.bsize + self.bsize]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            pred = output.argmax(dim=1)
            y_pred.append(pred.cpu().numpy().tolist())
            y_true.append(label.data.cpu().numpy().tolist())
            acc = (pred == label.data).cpu().sum() * 100 / len(data)
            print('eval loss:', loss.data.item(), "acc:%.1f" % acc + '%')
            losses.append(loss.data.item())
            accs.append(acc)
        print('eval loss:', np.mean(losses), "acc:%.1f" % np.mean(accs) + '%')
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        m = confusion_matrix(y_true, y_pred)
        np.save('./%s_mfigs/y_true.npy' % self.data_split_type, y_true)
        np.save('./%s_mfigs/y_pred.npy' % self.data_split_type, y_pred)
        np.save('./%s_mfigs/m.npy' % self.data_split_type, m)
        return np.mean(losses), np.mean(accs)

    def load_data(self):
        self.Train_x, self.Train_y, self.Test_x, self.Test_y = get_all_data_from_txt(
            '../tools/train_val_list/%s_train_list.txt' % self.data_split_type,
            '../tools/train_val_list/%s_val_list.txt' % self.data_split_type, self.data_split_type)


if __name__ == '__main__':
    from optparse import OptionParser
    optParser = OptionParser()
    optParser.add_option('-t', '--data_split_type', action='store', type="string", dest='data_split_type',
                         help="data_split_type", default="cv2")
    optParser.add_option('-n', '--number_of_gpu', action='store', type="int", dest='number_of_gpu',
                         help="the number of used gpu", default=0)
    option, args = optParser.parse_args()
    data_split_type = option.data_split_type
    ngpu = option.number_of_gpu
    rec_train = REC_train(data_split_type)
    print(rec_train.bsize, rec_train.lr)
    rec_train.train()
    print(data_split_type)
