import pandas as pd
import random
import numpy as np

lines = open('all_list.txt').readlines()
lines = [line.strip().split(' ') for line in lines]
df = pd.DataFrame(lines)
train_id = set(random.sample(range(0, 80), int(80 * 0.6)))
all_id = set(range(80))
val_id = all_id - train_id
peoples = ['chenhongyu', 'chenjialing', 'chenjiaxing', 'chenjinbao', 'chenyipeng', 'fanxiaoyuan', 'gongliang',
           'gongyue', 'guhaoying', 'guojunguang', 'guoqiong', 'haojingxuan', 'haoluoying', 'huangchi', 'huanghaiyan',
           'huangshuxian', 'huangweijuan', 'huangxiaoqin', 'huangxiaoye', 'huangzihang', 'hujing', 'laiqiqi',
           'laixinjie', 'laiyingru', 'liangxiaosi', 'lianroumin', 'lianshengxiong', 'lihongjian', 'lijiamin',
           'linchuang', 'linjiaqi', 'linjiawei', 'linyanfei', 'linyue', 'liqiang', 'lisinan', 'liujianeng',
           'liujinmeng', 'liuyuan', 'liuyuxiang', 'liwen', 'lukaijie', 'luowei', 'lvdan', 'maijinhui', 'panjiadong',
           'pengxiaobin', 'renpei', 'ruanpanpan', 'shangtianyu', 'shenguibao', 'sunshijie', 'sushuiqing', 'suzhilong',
           'tangwenjie', 'tanyuanqi', 'tianfangbi', 'tianyanling', 'wangchao', 'wangjing', 'wangxinnian', 'weiliyang',
           'wuhongzhen', 'wuxudong', 'xiaofen', 'xiaozhuo', 'xiezhuolin', 'yangruohan', 'yangweijie', 'zhangkuiqing',
           'zhangshenghai', 'zhangyingkui', 'zhaotaoling', 'zhenglimeng', 'zhengshaoxuan', 'zhengzegeng',
           'zhengzhicheng', 'zhongkaiyu', 'zhoulikai', 'zhuyaping']
train_names = [peoples[id] for id in train_id]
val_names = [peoples[id] for id in val_id]
train_list = pd.DataFrame()
for name in train_names:
    train_list = train_list.append(df[[name in t for t in df[0]]])
train_list = np.array(train_list).tolist()
train_list = [' '.join(line) + '\n' for line in train_list]
with open('cs_train_list.txt', 'w')as f:
    random.shuffle(train_list)
    f.writelines(train_list)
val_list = pd.DataFrame()
for name in val_names:
    val_list = val_list.append(df[[name in t for t in df[0]]])
val_list = np.array(val_list).tolist()
val_list = [' '.join(line) + '\n' for line in val_list]
with open('cs_val_list.txt', 'w')as f:
    random.shuffle(val_list)
    f.writelines(val_list)
