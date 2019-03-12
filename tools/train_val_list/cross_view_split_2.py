import pandas as pd
import random
import numpy as np
# 按照俯仰角划分
lines = open('all_list.txt').readlines()
lines = [line.strip().split(' ') for line in lines]
df = pd.DataFrame(lines)
ips = ['169.254.32.10', '169.254.32.11', '169.254.32.12', '169.254.32.13', '169.254.32.14', '169.254.32.15',
       '169.254.32.1', '169.254.32.2', '169.254.32.3', '169.254.32.4', '169.254.32.5', '169.254.32.6', '169.254.32.7',
       '169.254.32.8', '169.254.32.9']
ips = [ip+'_' for ip in ips]
ips = set(ips)
val_ips = ['169.254.32.9', '169.254.32.1', '169.254.32.4', '169.254.32.2', '169.254.32.8']
val_ips = [ip+'_' for ip in val_ips]
val_ips = set(val_ips)
train_ips = ips - val_ips

train_list = pd.DataFrame()
for name in train_ips:
    train_list = train_list.append(df[[name in t for t in df[0]]])
train_list = np.array(train_list).tolist()
train_list = [' '.join(line) + '\n' for line in train_list]
with open('cv2_train_list.txt', 'w')as f:
    random.shuffle(train_list)
    f.writelines(train_list)
print(len(train_list))
val_list = pd.DataFrame()
for name in val_ips:
    val_list = val_list.append(df[[name in t for t in df[0]]])
val_list = np.array(val_list).tolist()
val_list = [' '.join(line) + '\n' for line in val_list]
with open('cv2_val_list.txt', 'w')as f:
    random.shuffle(val_list)
    f.writelines(val_list)
print(len(val_list))
