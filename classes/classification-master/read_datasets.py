#-*-coding:utf-8-*-
# date:2020-02-08
# author: Eric.Lee
# function: read datasets label files

import os
import cv2
import numpy as np
import xml.etree.cElementTree as ET
import time

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]# color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # filled rectangle
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],\
        thickness=tf, lineType=cv2.LINE_AA)

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
    path = "./datasets/train/"
    idx = 0
    data_dict = {} # 统计每一类的个数

    for doc_ in os.listdir(path):# 遍历文件夹 文件
        for f_ in os.listdir(path + doc_):
            if ".jpg" not in f_:
                continue
            img_ = cv2.imread(path +doc_+"/"+ f_)

            idx += 1
            print('%s) '%(idx)+f_)
            cv2.putText(img_, ('index : ' + str(idx)), (5,img_.shape[0]-5),cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 6)
            cv2.putText(img_, ('index : ' + str(idx)), (5,img_.shape[0]-5),cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 60, 255), 2)

            xml_ =  path + doc_+"/"+f_.replace(".jpg",".xml").replace(".png",".xml")

            list_x = get_xml_msg(xml_)# 获取 xml 文件 的 object

            # 绘制 bbox
            for j in range(len(list_x)):
                label_,bbox_ = list_x[j]
                plot_one_box(bbox_,img_, label=label_, color=(255,0,0))
                if label_ not in data_dict.keys():
                    data_dict[label_] = 1
                else:
                    data_dict[label_] += 1
                cv2.namedWindow('image',0)
                cv2.imshow('image',img_)
                cv2.waitKey(1)

    cv2.destroyAllWindows()

    for key_ in data_dict.keys():
        print(key_,':',data_dict[key_])
