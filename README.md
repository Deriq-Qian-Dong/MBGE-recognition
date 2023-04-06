# MBGE-recognition
ICIP2019

<div align=center><img width="800" height="400" src="https://github.com/Deriq-Qian-Dong/MBGE-recognition/blob/master/image/%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D.png"/></div>

<div align=center>Fig1.The body gestures in MBGD</div>


<div align=center><img width="800" height="250" src="https://github.com/Deriq-Qian-Dong/MBGE-recognition/blob/master/image/%E6%B5%81%E7%A8%8B%E5%9B%BE.png"/></div>

<div align=center>Fig2.Pipeline of our approach</div>

## 1.generate features by TSN
***Note: tsn-pytorch's pytorch version is 0.3.1; This code was modified from yjxiong's [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)***

**a.generate optical flow from video by nvidia-docker**
>```nvidia-docker run -it -v **absolute path to dataset**:/data bitxiong/tsn:latest bash```

>```docker image can be download from [DockerHub](https://hub.docker.com/r/bitxiong/tsn)```

>>```cp /data/build_of.py ./tools/```

>>```python ./tools/build_of.py /data/video/ /data/optical_flow_data/```

>>```exit```

**b.generate train and val list**

>```cd tools ```
>>```python gen_train_test_list.py```

>>```cd train_val_list```
>>>```python cross_subject_split.py```

>>>```python cross_view_split_1.py```

>>>```python cross_view_split_2.py```

**c.generate features**

>```cd tsn-pytorch ```
>>**train model**

>>```CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py myDataset Flow ../tools/train_val_list/cv1_train_list.txt ../tools/train_val_list/cv1_val_list.txt --num_segments 7  --gd 20 --lr 0.001 --lr_steps 30 60 90 --epochs 100  -b 128 -j 8 --dropout 0.9 --gpus 0 1 2 3 4 --arch BNInception --snapshot_pref cv1_Flow_num_seg7_dropout_08 --flow_prefix flow_```

>>**plot confusion matrix**

>>```python gen_pred.py --data_split_type cv1 --gpus 0 1```

>>```python plot_confusion_matrix.py --data_split_type cv1```

>>**save features**

>>```python gen_features.py -t cv1 -m ./cv1_Flow_num_seg7_dropout_08_flow_model_best.pth.tar -g 0 -g 1```
 
## 2.generate features by ST-GCN
***Note: ST-GCN's pytorch version is 0.4.1; This code was modified from yysijie's [st-gcn](https://github.com/yysijie/st-gcn)***

**a.generate key points from video by OpenPose**

>```cp ./dataset/gen-keypoint.py **path to openpose**```

>```python gen-keypoint.py -r ~/dataset/video/ -o ~/dataset/keypoints```

**b.train model**
>```cd stgcn```
>>```python train_model.py --data_split_type cv1 --number_of_gpu 0```

**c.plot confusion matrix**

>```python gen_pred.py --data_split_type cv1 --number_of_gpu 0```

>```python plot_confusion_matrix.py --data_split_type cv1```

**d.generate features**

>```python gen_features.py --data_split_type cv1 --number_of_gpu 0```

## 3.trained by Residual-Fully-Connected-Network
>```cd stgcn```
>>```python mymodel.py --data_split_type cv1 --number_of_gpu 0```

<div align=center><img width="400" height="400" src="https://github.com/Deriq-Qian-Dong/MBGE-recognition/blob/master/image/cv1%E7%9F%A9%E9%98%B5.png"/></div>

<div align=center>Fig3.The confusion matrix of our approach on cv1</div>


<div align=center><img width="800" height="400" src="https://github.com/Deriq-Qian-Dong/MBGE-recognition/blob/master/image/cv1%E5%AF%B9%E6%AF%94.png"/></div>

<div align=center>Fig4.Category and overall accuracies of our approach on cross-view1</div>


<div align=center><img width="600" height="200" src="https://github.com/Deriq-Qian-Dong/MBGE-recognition/blob/master/image/%E6%95%88%E6%9E%9C%E5%AF%B9%E6%AF%94.png"/></div>

<div align=center>Fig5.The overall accuracies of TSN, ST-GCN and our approach</div>

***Dataset is about 360G, if you need to download it, contact Zhijuan Shen with email: zj.shen@siat.ac.cn***
