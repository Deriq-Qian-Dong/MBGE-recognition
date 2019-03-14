# -*- coding: utf-8 -*-
"""
Created on 2018/12/04 16:51
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import os
import random
from pathlib import Path
import numpy as np
import json
import os
import time
import sys

root_path = r"../dataset/keypoints"

def json_pack(snippets_dir, frame_width=1280, frame_height=720, label='unknown', label_index=-1):
    sequence_info = []
    video_name = os.path.basename(snippets_dir)
    p = Path(snippets_dir)
    for path in p.glob(video_name + '*.json'):
        json_path = str(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i] / frame_width, keypoints[i + 1] / frame_height]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    return video_info


def video_info_parsing(video_info, num_person_in=5, num_person_out=1, max_frame=60):
    data_numpy = np.zeros((3, max_frame, 18, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        if frame_index >= max_frame:
            break
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    label = video_info['label_index']
    return data_numpy, label


def get_data(snippets_dir, label, label_index):
    video_info = json_pack(snippets_dir, label=label, label_index=label_index)
    data_numpy, label = video_info_parsing(video_info)
    return data_numpy, label


def _get_label_dirs(label_name, all_keypoints=True):
    target_dirs = []
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name == label_name:
                if all_keypoints:  # /data5/dongqian/keypoints/chenhongyu/bla2/169.254.32.10_12_11_15_37_20_record/01-angry_throw
                    target_dirs.append(os.path.join(root, dir_name))
                else:  # /data5/szj/chenjialing/bla2/169.254.32.11_12_06_14_16_48_record/01-back/01-back
                    if dir_name in root:
                        target_dirs.append(os.path.join(root, dir_name))
    return target_dirs


def get_all_data():
    data_all = []
    label_name = ['01-angry_throw', '01-back', '01-crouch', '01-disgust_turn', '01-fear_back', '01-happy_jump',
                  '01-jump',
                  '01-retreat', '01-sad_squat', '01-surprise_retreat', '01-throw', '01-turn']
    labels = []
    for i, label in enumerate(label_name):
        print(i + 1, label)
        for dir_name in _get_label_dirs(label):
            data_numpy, label = get_data(dir_name, label, i)
            data_all.append(data_numpy)
            labels.append(label)
    data_all = np.array(data_all)
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(data_all)
    random.seed(randnum)
    random.shuffle(labels)
    return np.array(data_all), np.array(labels)


def get_all_data_from_txt(t, v, flist=False):
    ft_lines = open(t).readlines()
    fv_lines = open(v).readlines()
    # /data/zqs/data/frames/zhengzegeng/bla3/169.254.32.6_12_06_09_35_07_record/01-jump 25 6
    # /data5/dongqian/keypoints/chenhongyu/bla2/169.254.32.10_12_11_15_37_20_record/01-angry_throw
    ft_lines = [line.strip().replace("../dataset/optical_flow_data", root_path).split(' ') for line in ft_lines]
    fv_lines = [line.strip().replace("../dataset/optical_flow_data", root_path).split(' ') for line in
                fv_lines]
    X_train = []
    y_train = []
    cnt = 0
    all_num = len(ft_lines) + len(fv_lines)
    train_list = []
    for line in ft_lines:
        data_numpy, label = get_data(line[0], line[0].split('/')[-1], int(line[-1]))
        if len(data_numpy):
            train_list.append(line[0])
        X_train.append(data_numpy)
        y_train.append(label)
        cnt += 1
        # print(cnt, all_num)
        sys.stdout.write('\r')
        sys.stdout.write(
            "data loading... current-%d all-%d %s%% |%s" % (cnt, all_num, int(100 * cnt / all_num), int(100 * cnt / all_num) * '#'))
        sys.stdout.flush()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = []
    y_test = []
    val_list = []
    for line in fv_lines:
        data_numpy, label = get_data(line[0], line[0].split('/')[-1], int(line[-1]))
        if len(data_numpy):
            val_list.append(line[0])
        X_test.append(data_numpy)
        y_test.append(label)
        cnt += 1
        # print(cnt, all_num)
        sys.stdout.write('\r')
        sys.stdout.write(
            "data loading... current-%d all-%d %s%% |%s" % (cnt, all_num, int(100 * cnt / all_num), int(100 * cnt / all_num) * '#'))
        sys.stdout.flush()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    if not flist:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, train_list, X_test, val_list
