# MBGE-recognition
ICIP2019

<div align=center><img width="800" height="300" src="https://github.com/DQ0408/MBGE-recognition/blob/master/image/%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D.png"/></div>

## 1.generate features by TSN
***Note: tsn-pytorch's pytorch version is 0.3.1***

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
***Note: ST-GCN's pytorch version is 0.4.1***

**a.generate key points from video by OpenPose**

>```cp ./dataset/gen-keypoint.py **path to openpose**```

>```python gen-keypoint.py -r ~/dataset/video/ -o ~/dataset/keypoints```

**b.train model**

>```python train_model.py --data_split_type cv1 --number_of_gpu 0```

**c.plot confusion matrix**

>```python gen_pred.py --data_split_type cv1 --number_of_gpu 0```

>```python plot_confusion_matrix.py --data_split_type cv1```

**d.generate features**

>```python gen_features.py --data_split_type cv1 --number_of_gpu 0```

## 3.trained by Residual-Fully-Connected-Network

>```python mymodel.py --data_split_type cv1 --number_of_gpu 0```

