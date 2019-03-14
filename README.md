# MBGE-recognition
ICIP2019
other codes pytorch version 0.4.1
## 1.generate features by TSN
tsn-pytorch's pytorch version 0.3.1

a.generate optical flow from video by nvidia-docker

>sudo nvidia-docker run -it -v **absolute path to dataset**:/data bitxiong/tsn:latest bash
>>cp /data/build_of.py ./tools/

>>python ./tools/build_of.py /data/video/ /data/optical_flow_data/

>>exit

b.generate train and val list

>cd tools 
>>python gen_train_test_list.py

>>cd train_val_list
>>>python cross_subject_split.py

>>>python cross_view_split_1.py

>>>python cross_view_split_2.py

c.generate features

>cd tsn-pytorch 
>>**train model**

>>CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py myDataset Flow ../tools/train_val_list/cv2_train_list.txt ../tools/train_val_list/cv2_val_list.txt --num_segments 7  --gd 20 --lr 0.001 --lr_steps 30 60 90 --epochs 100  -b 128 -j 8 --dropout 0.9 --gpus 0 1 2 3 4 --arch BNInception --snapshot_pref cv2_Flow_num_seg7_dropout_08 --flow_prefix flow_

>>**plot confusion matrix**

>>python gen_pred.py --data_split_type cv2 --gpus 0 1

>>python plot_confusion_matrix.py --data_split_type cv2

>>**save features**

>>python gen_features.py -t cv2 -m ./cv2_Flow_num_seg7_dropout_08_flow_model_best.pth.tar -g 0 -g 1
 
## 2.generate features by ST-GCN

