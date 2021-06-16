---
layout: post
title: RSURFC-练习赛Baseline实现
toc: true
top: false
categories:
  - 收藏
tags:
  - 练习赛
  - Baseline
  - 城市功能区划分
  - Pytorch
  - 深度学习
date: 2020-06-07T09:26:00+00:00
---

> 最近两天趁着写完小论文不想做事情。尝试玩了一下2019年百度AI搞得城市功能区划分的比赛。比赛提供了遥感影像数据和区域居民活动数据(可惜没有具体的位置信息)，初赛样本量4W，复赛样本量40W，拿初赛的样本玩了玩，实现了Baseline（0.67）。其实也是照抄别人代码，主要是为了熟悉pytorch环境和多模学习。对了，这个练习赛最开始是在本机上实现，但后来模型增大，显存无法承载，显存不足，转到Google 的Colab实现。别说，配合谷歌自带的网盘，这个Colab真的挺好用(主要是免费GPU，前提你的VPN要稳定)。


<!--more-->

### 准备
- 比赛链接: https://aistudio.baidu.com/aistudio/competition/detail/13
- 数据
  - 飞桨基线挑战赛（训练集）:https://aistudio.baidu.com/aistudio/datasetdetail/12529
  - 飞桨基线挑战赛（测试集）:https://aistudio.baidu.com/aistudio/datasetdetail/12530
  - 预处理数据: https://github.com/ABadCandy/BaiDuBigData19-URFC
- 分数:
  - 官方基线Pytorch实现(本文主要参考): https://github.com/ABadCandy/BaiDuBigData19-URFC
  - 团队Expelliarmus(0.88767): https://aistudio.baidu.com/aistudio/projectdetail/195440
  - URFC-top4(本文主要参考): https://github.com/destiny19960207/URFC-top4

- import
```python
!pip install pretrainedmodels
import numpy as np
from functools import partial
import pandas as pd
import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
from skimage import io

from tqdm import tqdm 
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score

from pathlib import Path

from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision

from imgaug import augmenters as iaa



import pretrainedmodels
from pretrainedmodels.models import *



import ssl
ssl._create_default_https_context = ssl._create_unverified_context # 防止ssl报错

```
- Vars init
```python
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')   

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')



# dir & fp
root_dir = Path('./data')
images_dir = Path(root_dir, 'images')
visits_dir = Path(root_dir, 'visits')
train_df_fp = Path(root_dir, 'train.csv')

log_dir = Path('./logs')
checkpoints_dir = Path('./checkpoints')
if not Path.exists(log_dir):
    Path.mkdir(log_dir)
    
if not Path.exists(checkpoints_dir):
    Path.mkdir(checkpoints_dir)

# config 

loss_name ='focal_loss'
pre_train_name = 'densenet121'  # se_resnext101_32x4d
env='default'
model_name = "multimodal_%s_%s" % (pre_train_name, loss_name)
num_classes = 9
img_weight = 100
img_height = 100

fold = 1
lr = 0.004
lr_decay = 0.5
weight_decay =1e-5
batch_size = 256
epochs = 30

```
- 工具函数
```python
# utils

# save best model
def save_checkpoint(state, is_best_acc,is_best_loss,is_best_f1,fold):
    checkpoints_dir = Path("./checkpoints")
    best_models_dir = Path(checkpoints_dir, "best_models")
    model_fp = Path(checkpoints_dir, "{}_{}_checkpoint.pth.tar".format(model_name, str(fold)))
    torch.save(state, model_fp)
    if is_best_acc:
        shutil.copyfile(model_fp,"%s/%s_fold_%s_model_best_acc.pth.tar"%(best_models_dir,model_name,str(fold)))
    if is_best_loss:
        shutil.copyfile(model_fp,"%s/%s_fold_%s_model_best_loss.pth.tar"%(best_models_dir,model_name,str(fold)))
    if is_best_f1:
        shutil.copyfile(model_fp,"%s/%s_fold_%s_model_best_f1.pth.tar"%(best_models_dir,model_name,str(fold)))


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
        

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError

class FCViewer(nn.Module): # 转化为二维tensor((x.size(0), -1))
    def forward(self, x):
        return x.view(x.size(0), -1) # resize/reshape tensor
```

### 数据处理
#### 文件夹信息
预处理数据下载解压后去除test(无标签)数据后，整理为两个文件夹(images和visits)。根据文件夹路径，记录文件id(区域id)、功能区id(0-8)、影像(Image, *.jpg)路径、用户访问(Visit, *.npy)路径属性
```python
train_df = pd.read_csv(train_df_fp, header=0)
image_fp_list = list(images_dir.iterdir())
stem2fp_dict = {x.stem: x for x in image_fp_list}
train_df['image_fp'] = train_df.Id.map(stem2fp_dict)
visit_fp_list = list(visits_dir.iterdir())
stem2fp_dict = {x.stem: x for x in visit_fp_list}
train_df['visit_fp'] = train_df.Id.map(stem2fp_dict)
train_df.to_csv(train_df_fp, header=True, index=False)
train_df.head()
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200607162449.png)

#### torchvision.Dataset & Dataloader
使用torchvision提供的API载入两类数据
```python

# create dataset class
class MultiModalDataset(Dataset):
    def __init__(self,file_df,augument=True):
        self.file_df = file_df.copy() #csv
        self.augument = augument

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        visit=self.read_npy(index).transpose(1,2,0)
        y = self.file_df.iloc[index].Target
        if self.augument:
            X = self.augumentor(X) # 图像增强
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        visit=T.Compose([T.ToTensor()])(visit)
        return X.float(),visit.float(),y

    
    def read_images(self,index):
        row = self.file_df.iloc[index]
        images = io.imread(row.image_fp)
        return images

    def read_npy(self,index):
        row = self.file_df.iloc[index]
        visit=np.load(row.visit_fp)
        return visit

    def augumentor(self,image):# 图像随机增强
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug


# split data
all_files_df = pd.read_csv(train_df_fp)
train_df,val_df = train_test_split(all_files_df, test_size=0.1, random_state = 2050)


# load dataset
train_gen = MultiModalDataset(train_df)
train_loader = DataLoader(train_gen,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=0) #num_worker is limited by shared memory in Docker!

val_gen = MultiModalDataset(val_df, augument=False)
val_loader = DataLoader(val_gen,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=0)



```

Note:  

- `torch.transpose`: 高维向量转置.如原始向量维度(7,26,24), `torch.transpose(1,2,0)`转换后，向量维度根据索引位置改变而变化为(26,24,7)。 (索引位置: 0->7, 1->26, 2->24)

- image_aug: 图像增强，别人都说强；训练集图像增强后有利于缓解模型过拟合，增强模型泛化能力。

- `num_workers`: window10下只能设置为0，不然报错


#### 模型设计
从单一模型(ImageNet, VisitNet)到复合模型(MulitNet)的思路开始尝试，先载入预训练模型进行简单修改，后尝试基于预训练模型进行自定义，最后尝试结合两种模型。
##### 载入预训练模型进行简单修改
- ImageNet
```python
model = torchvision.models.resnet18(pretrained=True)
model = torchvision.models.resnet34(pretrained=True)
# 更改最后输出为9类
model.fc = nn.Linear(model.fc.in_features, num_classes)
```
- VisitNet
```python
model = torchvision.models.resnet152(pretrained=True)
# 改为7通道,原始图片输入为3通道，visit输入为7通道
model.conv1 = nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)
```
##### 基于预训练模型进行自定义
- ImageNet
```python
class ImageModalNet(nn.Module):
    def __init__(self, backbone='densenet121', drop=0.25, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        else:
            img_model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)
        
        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                        nn.BatchNorm1d(img_model.last_linear.in_features),
                                        nn.Dropout(drop),
                                        nn.Linear(img_model.last_linear.in_features, 256))
        
        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.BatchNorm1d(img_model.last_linear.in_features),
                nn.Linear(img_model.last_linear.in_features, 256)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        
        self.cls = nn.Linear(256, num_classes)
    
    def forward(self, x_img):
        x = self.img_encoder(x_img)
        x = self.img_fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.cls(x)
        return x
```
- VisitNet (DPN网络)
```python
class VisitNet(nn.Module): #随机初始化的DPN26: https://github.com/destiny19960207/URFC-top4
    def __init__(self):
        super(VisitNet, self).__init__()
        k = 1
        in_channel = 7
        layer1_1 = []
        layer1_1.append(nn.Conv2d(in_channel, 64 * k, kernel_size=(6, 1), stride=(6, 1)))
        layer1_1.append(nn.BatchNorm2d(64 * k))
        layer1_1.append(nn.ReLU())
        layer1_1.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(1, 7), stride=(1, 7)))
        layer1_1.append(nn.BatchNorm2d(64 * k))
        layer1_1.append(nn.ReLU())
        self.cell_1_1 = nn.Sequential(*layer1_1)
        layer1_2 = []
        layer1_2.append(nn.Conv2d(in_channel, 64 * k, kernel_size=(1, 7), stride=(1, 7), padding=(0, 0)))
        layer1_2.append(nn.BatchNorm2d(64 * k))
        layer1_2.append(nn.ReLU())
        layer1_2.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(6, 1), stride=(6, 1), padding=(0, 0)))
        layer1_2.append(nn.BatchNorm2d(64 * k))
        layer1_2.append(nn.ReLU())
        self.cell_1_2 = nn.Sequential(*layer1_2)
        layer1_3 = []
        layer1_3.append(nn.Conv2d(in_channel, 64 * k, kernel_size=(6, 5), stride=(6, 1), padding=(0, 2)))
        layer1_3.append(nn.BatchNorm2d(64 * k))
        layer1_3.append(nn.ReLU())
        layer1_3.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(5, 7), stride=(1, 7), padding=(2, 0)))
        layer1_3.append(nn.BatchNorm2d(64 * k))
        layer1_3.append(nn.ReLU())
        self.cell_1_3 = nn.Sequential(*layer1_3)
        layer1_4 = []
        layer1_4.append(nn.Conv2d(in_channel, 64 * k, kernel_size=(5, 7), stride=(1, 7), padding=(2, 0)))
        layer1_4.append(nn.BatchNorm2d(64 * k))
        layer1_4.append(nn.ReLU())
        layer1_4.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(6, 5), stride=(6, 1), padding=(0, 2)))
        layer1_4.append(nn.BatchNorm2d(64 * k))
        layer1_4.append(nn.ReLU())
        self.cell_1_4 = nn.Sequential(*layer1_4)


        layer2_1 = []
        layer2_1.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer2_1.append(nn.BatchNorm2d(256 * k))
        layer2_1.append(nn.ReLU())
        layer2_1.append(nn.Dropout(0.1))
        layer2_1.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer2_1.append(nn.BatchNorm2d(256 * k))
        layer2_1.append(nn.ReLU())
        layer2_1.append(nn.Dropout(0.1))
        self.cell_2_1 = nn.Sequential(*layer2_1)
        layer2_2 = []
        layer2_2.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        layer2_2.append(nn.BatchNorm2d(256 * k))
        layer2_2.append(nn.ReLU())
        layer2_2.append(nn.Dropout(0.1))
        layer2_2.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layer2_2.append(nn.BatchNorm2d(256 * k))
        layer2_2.append(nn.ReLU())
        layer2_2.append(nn.Dropout(0.1))
        self.cell_2_2 = nn.Sequential(*layer2_2)


        layer3_1 = []
        layer3_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer3_1.append(nn.BatchNorm2d(512 * k))
        layer3_1.append(nn.ReLU())
        layer3_1.append(nn.Dropout(0.2))
        layer3_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer3_1.append(nn.BatchNorm2d(512 * k))
        layer3_1.append(nn.ReLU())
        layer3_1.append(nn.Dropout(0.2))
        self.cell_3_1 = nn.Sequential(*layer3_1)


        layer4_1 = []
        layer4_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer4_1.append(nn.BatchNorm2d(512 * k))
        layer4_1.append(nn.ReLU())
        layer4_1.append(nn.Dropout(0.2))
        layer4_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer4_1.append(nn.BatchNorm2d(512 * k))
        layer4_1.append(nn.ReLU())
        layer4_1.append(nn.Dropout(0.2))
        self.cell_4_1 = nn.Sequential(*layer4_1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        fc_dim = 4 * 3 * 512 * k 
        self.fc = nn.Sequential(FCViewer(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_dim, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 256)
                                )

    def forward(self, x):
        x1_1 = self.cell_1_1(x)
        x1_2 = self.cell_1_2(x)
        x1_3 = self.cell_1_3(x)
        x1_4 = self.cell_1_4(x)
        x_in = torch.cat([x1_1, x1_2, x1_3, x1_4], 1)

        x_out_1 = self.cell_2_1(x_in)
        x_out_2 = self.cell_2_2(x_in)
        x_in = torch.cat([x_out_1, x_out_2], 1)

        x_out = self.cell_3_1(x_in)
        x_in = x_in + x_out

        x_out = self.cell_4_1(x_in)
        x_in = x_in + x_out

        out = self.fc(x_in)
        return out
```

##### 模型结合 & 多模学习
- 自己写的
```python
class MultiModalNet(nn.Module):
    def __init__(self, backbone, drop, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        else:
            img_model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)
        visit_model = torchvision.models.resnet152(pretrained=True)
        visit_model.conv1 = nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 改为7通道
        visit_model.fc = nn.Linear(visit_model.fc.in_features, 256)
        self.visit_encoder = list(visit_model.children())[:-2]
        self.visit_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.visit_encoder = nn.Sequential(*self.visit_encoder)
        self.fc_viewer = nn.Sequential(FCViewer())

        

        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(3072)
        self.dropout = nn.Dropout(0.5)
        
        self.cls1 = nn.Linear(3072, 1024)
        self.cls2 = nn.Linear(1024, num_classes)

    
    def forward(self, x_img, x_visit):
        x_img = self.img_encoder(x_img)
        x_img = self.fc_viewer(x_img)

        x_visit = self.visit_encoder(x_visit)
        x_visit = self.fc_viewer(x_visit)
        x_cat = torch.cat((x_img, x_visit), 1)
        x = self.relu(x_cat)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.cls1(x)
        x = self.cls2(x)
        return x
```
- 有些修改
```python
class MultiNet(nn.Module):
  def __init__(self,  drop):
    super().__init__()
    img_model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')  # seresnext101
    self.img_encoder = list(img_model.children())[:-2]
    self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
    self.img_encoder = nn.Sequential(*self.img_encoder,
                                      FCViewer(),
                                      nn.Dropout(drop),
                                      nn.Linear(2048, 256),
                                      )

    self.visit_conv = VisitNet()
    #### cat 512->9
    cat_dim = 256 + 256
    self.fc = nn.Sequential(FCViewer(),
                                    nn.ReLU(),
                                    nn.Dropout(drop),
                                    nn.Linear(cat_dim, cat_dim),
                                    nn.ReLU(),
                                    nn.Dropout(drop),
                                    nn.Linear(cat_dim, num_classes)
                                    )
  def forward(self, x_img, x_vis):
      x1 = self.img_encoder(x_img)
      x2 = self.visit_conv(x_vis)
      x3 = torch.cat([x1, x2], 1)
      out = self.fc(x3)
      return out

```
其实调试过程中很多报错是很烦躁的，又不懂模型原理，只能检索后慢慢调试。这里声明一下"Google大法好",解决90%问题。报错内容大体忘记了，通常包含维度不一致等。

### 主函数
#### 训练&验证函数
这里比较通用，主要是模型和参数修改比较大
```python
def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()

    model.train()
    for i,(images,visit,target) in enumerate(train_loader):
        visit=visit.to(device)
        images = images.to(device)
        indx_target=target.clone()
        target = torch.from_numpy(np.array(target)).long().to(device)
        # compute output
        output = model(images,visit)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))
        f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
        acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        f1.update(f1_batch,images.size(0))
        acc.update(acc_score,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                valid_metrics[0], valid_metrics[1],valid_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [acc.avg,losses.avg,f1.avg]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    acc= AverageMeter()
    # switch mode for evaluation
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images,visit,target) in enumerate(val_loader):
            images_var = images.to(device)
            visit=visit.to(device)
            indx_target=target.clone()
            target = torch.from_numpy(np.array(target)).long().to(device)
            
            output = model(images_var,visit)
            loss = criterion(output,target)
            losses.update(loss.item(),images_var.size(0))
            f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
            acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))        
            f1.update(f1_batch,images.size(0))
            acc.update(acc_score,images.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    acc.avg,losses.avg,f1.avg,
                    train_metrics[0], train_metrics[1],train_metrics[2],
                    str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                    time_to_str((timer() - start),'min'))

            print(message, end='',flush=True)
        log.write("\n")
        #log.write(message)
        #log.write("\n")
        
    return [acc.avg,losses.avg,f1.avg]
```

#### main
```python

start_epoch = 0
best_acc=0
best_loss = np.inf
best_f1 = 0
best_results = [0,np.inf,0]
val_metrics = [0,np.inf,0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiNet(0.25)
model.to(device)

optimizer = torch.optim.Adam([{'params': model.parameters()}],
                             lr=lr,
                             weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to(device)


all_files_df = pd.read_csv(train_df_fp)
train_df,val_df = train_test_split(all_files_df, test_size=0.1, random_state = 2050)


# load dataset
train_gen = MultiModalDataset(train_df)
train_loader = DataLoader(train_gen,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=0) #num_worker is limited by shared memory in Docker!

val_gen = MultiModalDataset(val_df, augument=False)
val_loader = DataLoader(val_gen,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=0)



scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)




start = timer()

log = Logger()
log.open("logs/%s_log_train.txt"%model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')



fold = 1
for epoch in range(0, epochs):
    print('lr:', lr / (2 ** np.sqrt(epoch + 0.1)))
    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=lr / (2 ** np.sqrt(epoch + 0.1)),
                                 weight_decay=1e-4)

    scheduler.step(epoch)
    # train
    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
    # val
    val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)
    # check results
    is_best_acc = val_metrics[0] > best_results[0]
    best_results[0] = max(val_metrics[0], best_results[0])
    is_best_loss = val_metrics[1] < best_results[1]
    best_results[1] = min(val_metrics[1], best_results[1])
    is_best_f1 = val_metrics[2] > best_results[2]
    best_results[2] = max(val_metrics[2], best_results[2])
    # save model
    save_checkpoint({
        "epoch": epoch + 1,
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "best_acc": best_results[0],
        "best_loss": best_results[1],
        "optimizer": optimizer.state_dict(),
        "fold": fold,
        "best_f1": best_results[2],
    }, is_best_acc, is_best_loss, is_best_f1, fold)
    # print logs
    print('\r', end='', flush=True)
    log.write(
        '%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % ( \
            "best", epoch, epoch,
            train_metrics[0], train_metrics[1], train_metrics[2],
            val_metrics[0], val_metrics[1], val_metrics[2],
            str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
            time_to_str((timer() - start), 'min'))
    )
    log.write("\n")
    time.sleep(0.01)
  ```


### 文件打包
下载地址:[RSURFC.rar](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/ethan.imfast.io/Codes/RSURFC.rar)


***




<!-- 插入图片 -->

![](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/ethan.imfast.io/imgs/2020/06/abstract-art-background-blur-564908.jpg)
