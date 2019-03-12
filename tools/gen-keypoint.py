# -*- coding: utf-8 -*-
"""
Created on 2018/12/21 20:42
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import os
from optparse import OptionParser
optParser = OptionParser()
optParser.add_option('-r', '--root_path', action='store', type="string", dest='root_path',
                     help="read videos' root path", default="../dataset/MBGE_dataset")
optParser.add_option('-o', '--out_path', action='store', type="string", dest='out_path',
                     help="save videos' key point to output path", default="../dataset/keypoints")
option, args = optParser.parse_args()
root_path = option.root_path
out_path = option.out_path
while True:
    all_names = set(os.listdir(root_path))
    done_names = set(os.listdir(out_path))
    left_names = list(all_names - done_names)
    if len(left_names) == 0:
        break
    name = left_names[0]
    print(name, len(left_names))
    dir_name = os.path.join(root_path, name)
    for root, dirs, files in os.walk(dir_name):
        for v_name in files:
            if "csv" in v_name:
                continue
            f_name = os.path.join(root, v_name)
            label = out_path + f_name.replace(root_path, '').replace('.avi', '') + '/'
            print(label)
            print(f_name)
            if not os.path.exists(label):
                os.makedirs(label)
            os.system(
                "./build/examples/openpose/openpose.bin --video %s --display 0 --num_gpu 1 --num_gpu_start 0 --write_json %s --write_video %s" % (
                    f_name, label, label + 'result.avi'))
