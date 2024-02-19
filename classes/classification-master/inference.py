#-*-coding:utf-8-*-
# date:2020-04-12
# Author: Eric.Lee
# function: inference

import os
import argparse
import torch
import torch.nn as nn
from data_iter.datasets import letterbox
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
import xml.etree.cElementTree as ET
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_xml_msg(path):
    list_x = []
    tree=ET.parse(path)# 解析 xml 文件
    root=tree.getroot()
    for Object in root.findall('object'):
        name=Object.find('name').text
        #----------------------------
        bndbox=Object.find('bndbox')
        xmin= np.float32((bndbox.find('xmin').text))
        ymin= np.float32((bndbox.find('ymin').text))
        xmax= np.float32((bndbox.find('xmax').text))
        ymax= np.float32((bndbox.find('ymax').text))
        bbox = int(xmin),int(ymin),int(xmax),int(ymax)
        xyxy = xmin,ymin,xmax,ymax
        list_x.append((name,xyxy))
    return list_x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification top1 Test')
    parser.add_argument('--test_model', type=str, default = './model_exp/2021-03-08_02-38-39/resnet_34-192_epoch-10.pth',
        help = 'test_model') # 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_34',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 14,
        help = 'num_classes') #  分类类别个数
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './handpose_x_gesture_v1/',
        help = 'test_path') # 测试集路径
    parser.add_argument('--img_size', type=tuple , default = (192,192),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--have_label_file', type=bool , default = False,
        help = 'have_label_file') # 是否可视化图片
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_18':
        model_=resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_=resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_50':
        model_=resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_=resnet101(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_152':
        model_=resnet152(num_classes=ops.num_classes, img_size=ops.img_size[0])
    else:
        print('error no the struct model : {}'.format(ops.model))

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.test_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))
    #----------------------------------------------------------------
    dict_r = {}
    dict_p = {}
    dict_static = {}
    for idx,doc in enumerate(sorted(os.listdir(ops.test_path), key=lambda x:int(x.split('-')[0]), reverse=False)):
        if doc not in dict_static.keys():
            dict_static[idx] = doc
            dict_r[doc] = 0
            dict_p[doc] = 0
    #---------------------------------------------------------------- 预测图片

    font = cv2.FONT_HERSHEY_SIMPLEX
    with torch.no_grad():
        for idx,doc in enumerate(sorted(os.listdir(ops.test_path), key=lambda x:int(x.split('-')[0]), reverse=False)):

            gt_label = idx
            for file in os.listdir(ops.test_path+doc):
                if ".jpg" not in file:
                    continue
                print('------>>> {} - gt_label : {}'.format(file,gt_label))

                img = cv2.imread(ops.test_path +doc+'/'+ file)
                #---------------
                if ops.have_label_file:
                    xml_ = ops.test_path +doc+'/'+ file.replace(".jpg",".xml")

                    list_x = get_xml_msg(xml_)# 获取 xml 文件 的 object

                    # 绘制 bbox
                    for j in range(min(1,len(list_x))):
                        label_,bbox_ = list_x[j]
                        x1,y1,x2,y2 = bbox_
                        x1 = int(np.clip(x1,0,img.shape[1]-1))
                        y1 = int(np.clip(y1,0,img.shape[0]-1))
                        x2 = int(np.clip(x2,0,img.shape[1]-1))
                        y2 = int(np.clip(y2,0,img.shape[0]-1))
                        img = img[y1:y2,x1:x2,:]

                # 输入图片预处理
                if ops.fix_res:
                    img_ = letterbox(img,size_=ops.img_size[0],mean_rgb = (128,128,128))
                else:
                    img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
                if ops.vis:
                    cv2.namedWindow('image',0)
                    cv2.imshow('image',img_)
                    cv2.waitKey(1)
                img_ = img_.astype(np.float32)
                img_ = (img_-128.)/256.

                img_ = img_.transpose(2, 0, 1)
                img_ = torch.from_numpy(img_)
                img_ = img_.unsqueeze_(0)

                if use_cuda:
                    img_ = img_.cuda()  # (bs, 3, h, w)

                pre_ = model_(img_.float())

                outputs = F.softmax(pre_,dim = 1)
                outputs = outputs[0]

                output = outputs.cpu().detach().numpy()
                output = np.array(output)

                max_index = np.argmax(output)

                score_ = output[max_index]

                print('gt {} - {} -- pre {}     --->>>    confidence {}'.format(doc,gt_label,max_index,score_))
                dict_p[dict_static[max_index]] += 1
                if gt_label == max_index:
                    dict_r[doc] += 1

    cv2.destroyAllWindows()
    # Top1 的每类预测精确度。
    print('\n-----------------------------------------------\n')
    acc_list = []
    for idx,doc in enumerate(sorted(os.listdir(ops.test_path), key=lambda x:int(x.split('-')[0]), reverse=False)):
        fm = dict_p[doc]
        fz = dict_r[doc]
        acc_list.append(fz/fm)
        try:
            print('{}: {}'.format(doc,fz/fm))
        except:
            print('error')
    print("\n MAP : {}".format(np.mean(acc_list)))
    print('\nwell done ')
