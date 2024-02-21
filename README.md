# 手势交互

## 项目介绍

*  采用python多进程实现。
* 1、实现单手点击，即大拇指和食指捏合时认为点击。
* 2、实现手的轨迹跟踪，可支持动态手势二次开发。
* 3、实现双手配合点击选中目标区域。
* 4、实现基于IOU的手部跟踪。
* 5、支持语音拓展功能。
  （识别功能后续添加）

## 项目配置  
### 1、软件  
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python
* playsound
### 2、硬件
* 普通USB彩色（RGB）摄像头

## 相关项目
### 1、手部检测
* 文件夹：hand_det
* [预训练模型下载地址](链接：https://pan.baidu.com/s/1yugPgFuNvTAKxDqrP0M17Q?pwd=6h3o 
提取码：6h3o)
* 可以根据自己需求替换检测模型。
### 2、手部关键点
* 文件夹：handkeypoint_reg
* [预训练模型下载地址](链接：https://pan.baidu.com/s/1gkhU5dMnbkyOoA3o4S6kQg?pwd=9dpz 
提取码：9dpz)

### 3、检测分类(classification)
* 文件夹：classes
* [imagenet，预训练模型下载地址](链接：https://pan.baidu.com/s/17iBkPeXuO5rPJfHMieVbvQ?pwd=pfui 
提取码：pfui)
* 可以根据自己的需求替换识别模型及对应的语音素材。
  语音素材在文件夹materials

## 项目使用方法  

### 1、下载手部检测模型和21关键点回归模型。
### 2、确定摄像头连接成功。
### 3、打开配置文件 lib/hand_lib/cfg/[handpose.cfg](https://codechina.csdn.net/EricLee/dpcas/-/blob/master/lib/hand_lib/cfg/handpose.cfg) 进行相关参数配置，具体配置参数如下(可根据实际情况进行自定义修改):
```
detect_model_path=./latest_416.pt #手部检测模型地址
detect_model_arch=yolo_v3 #检测模型类型 ，yolo  or yolo-tiny
yolo_anchor_scale=1.0 # yolo anchor 比例，默认为 1
detect_conf_thres=0.5 # 检测模型阈值
detect_nms_thres=0.45 # 检测模型 nms 阈值

handpose_x_model_path=./ReXNetV1-size-256-wingloss102-0.1063.pth # 21点手回归模型地址
handpose_x_model_arch=rexnetv1 # 回归模型结构

classify_model_path=./imagenet_size-256_20210409.pth # 分类识别模型地址
classify_model_arch=resnet_50 # 分类识别模型结构
classify_model_classify_num=1000 # 分类类别数

camera_id = 0 # 相机 ID ，一般默认为0，如果不是请自行确认
vis_gesture_lines = True # True: 点击时的轨迹可视化， False：点击时的轨迹不可视化
charge_cycle_step = 32 # 点击稳定状态计数器，点击稳定充电环。
```
### 4、根目录下运行命令： python main.py



[此项目以https://gitcode.net/EricLee/dpcas 为基础]

  
