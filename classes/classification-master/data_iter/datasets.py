#-*-coding:utf-8-*-
# date:2019-05-20
# author: Eric.Lee
# function: data iter

import glob
import math
import os
import random
import shutil
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import xml.etree.cElementTree as ET

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

# 非形变处理
def letterbox(img_,size_=416,mean_rgb = (128,128,128)):

    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(size_) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (size_ - new_shape_[0]) / 2  # width padding
    dh_ = (size_ - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=mean_rgb)  # padded square
    # print('fix size : ',img_a.shape)
    return img_a
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

def img_agu_crop(img_):
    # scale_ = int(min(img_.shape[0],img_.shape[1])/15)
    scale_ = 5
    x1 = max(0,random.randint(0,scale_))
    y1 = max(0,random.randint(0,scale_))
    x2 = min(img_.shape[1]-1,img_.shape[1] - random.randint(0,scale_))
    y2 = min(img_.shape[0]-1,img_.shape[1] - random.randint(0,scale_))
    # print(img_.shape,'-crop- : ',x1,y1,x2,y2)
    try:
        img_crop_ = img_[y1:y2,x1:x2,:]
    except:
        img_crop_ = img_
        print("img_agu_crop error ")
    return img_crop_
# 图像旋转
def M_rotate_image(image , angle , cx , cy):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    (h , w) = image.shape[:2]
    # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    return cv2.warpAffine(image , M , (nW , nH)) , M

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=(224,224), flag_agu = False,fix_res = True,val_split = [],have_label_file = False):
        print('img_size (height,width) : ',img_size[0],img_size[1])
        labels_ = []
        files_ = []
        for idx,doc in enumerate(sorted(os.listdir(path), key=lambda x:int(x.split('-')[0]), reverse=False)):
        # for idx,doc in enumerate(os.listdir(path)):
            print(' %s label is %s \n'%(doc,idx))

            for file in os.listdir(path+doc):
                if '.jpg' in file and  ((path+doc + '/' + file) not in val_split) :# 同时过滤掉 val 数据集
                    labels_.append(idx)
                    files_.append(path+doc + '/' + file)
            print()
        print('\n')
        cv2.destroyAllWindows()
        self.labels = labels_
        self.files = files_
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.have_label_file = have_label_file

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        label_ = self.labels[index]

        img = cv2.imread(img_path)  # BGR
        #--------------------------------------------
        if self.have_label_file:
            xml_ = img_path.replace(".jpg",".xml")

            list_x = get_xml_msg(xml_)# 获取 xml 文件 的 object

            # 绘制 bbox
            choose_idx = random.randint(0,int(len(list_x)-1))
            for j in range(len(list_x)):
                if j ==choose_idx:
                    _,bbox_ = list_x[j]
                    x1,y1,x2,y2 = bbox_
                    x1 = int(np.clip(x1,0,img.shape[1]-1))
                    y1 = int(np.clip(y1,0,img.shape[0]-1))
                    x2 = int(np.clip(x2,0,img.shape[1]-1))
                    y2 = int(np.clip(y2,0,img.shape[0]-1))
                    img = img[y1:y2,x1:x2,:]
                    break

        #--------------------------------------------

        if self.flag_agu == True and random.random()>0.5:
            img = img_agu_crop(img)

        cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]

        if self.flag_agu == True and random.random()>0.6:
            cx = int(img.shape[1]/2)
            cy = int(img.shape[0]/2)
            # 手势(gesture)分类建议是全角度旋转, 对于 Stanford dogs 数据集适当角度旋转扰动，目的是为了符合真实样本旋转角度样本分布情况。
            angle = random.randint(-180,180)
            range_limit_x = int(min(6,img.shape[1]/16))
            range_limit_y = int(min(6,img.shape[0]/16))
            offset_x = random.randint(-range_limit_x,range_limit_x)
            offset_y = random.randint(-range_limit_y,range_limit_y)
            if not(angle==0 and offset_x==0 and offset_y==0):
                try:
                    img,_  = M_rotate_image(img , angle , cx+offset_x , cy+offset_y)
                except:
                    print("M_rotate_image error ")
                    img = cv2.imread(img_path)



        if self.flag_agu == True and random.random()>0.9:
            resize_idx = random.randint(0,3)

            if self.fix_res:
                img_ = letterbox(img,size_=self.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img, (self.img_size[1],self.img_size[0]), interpolation = cv_resize_model[resize_idx])
        else:
            if self.fix_res:
                img_ = letterbox(img,size_=self.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img, (self.img_size[1],self.img_size[0]), interpolation = cv2.INTER_CUBIC)

        if self.flag_agu == True and random.random()>0.5:
            img_ = cv2.flip(img_, random.randint(-1,1))# 0上下翻转 ，-1，上下+左右翻转 ，1左右翻转
            # print("---->>. flip")

        if self.flag_agu == True:
            if random.random()>0.6:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)

        if self.flag_agu == True:
            if random.random()>0.9:# and (label_ == 15 or label_ == 16 or label_ == 17):
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

        # img_ = prewhiten(img_)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)

        return img_,label_
