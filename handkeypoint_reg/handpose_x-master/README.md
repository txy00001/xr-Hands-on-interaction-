# HandPose X  
手势 21 个关键点检测  


## 项目介绍     
* 1 - 按键操作     
  因为考虑到目前没有三维姿态不好识别按键按下三维动作，所以目前采用二维方式。    
  该示例的原理：通过简单的IOU跟踪，对二维目标如手的边界框或是特定手指的较长时间位置稳定性判断确定触发按键动作的时刻，用特定指尖的二维坐标确定触发位置。    
  （注意：目前示例并未添加到工程，后期整理后会进行发布，只是一个样例，同时希望同学们自己尝试写自己基于该项目的小应用。）     
![keyboard](https://gitcode.net/EricLee/handpose_x/-/raw/master/samples/keyboard.gif)  

* 2 - 手势交互：指定区域物体识别      
  该示例的出发点是希望通过手势指定用户想要识别的物体。那么就要选中物体的准确边界框才能达到理想识别效果。如果待识别目标边界框太大会引入背景干扰，太小又会时目标特征不完全。所以希望通过手势指定较准确的目标边界框。因为边界框涉及左上、右下两个二维坐标，所以通过两只手的特定指尖来确定。且触发逻辑与示例1相同。           
  该示例的原理：通过简单的IOU跟踪，对二维目标如手的边界框或是特定手指的较长时间位置稳定性判断确定触发按键动作的时刻，用特定指尖的二维坐标确定触发位置。         
   

![keyboard](https://gitcode.net/EricLee/handpose_x/-/raw/master/samples/recognize_obj0.gif)    

* 以下是对书上狗的图片进行分类识别的样例，同学们可以根据自己对应的物体识别分类需求替换对应的分类识别模型即可。    

![recoobj_book](https://gitcode.net/EricLee/handpose_x/-/raw/master/samples/recobj_book.gif)    
        

* 3 - 静态手势     
  通过手关键点的二维角度约束关系定义静态手势。  
  示例中手势包括：fist five gun love one six three thumbup yeah    
  目前该示例由于静态手势数据集的限制，目前用手骨骼的二维角度约束定义静态手势，原理如下图,计算向量AC和DE的角度，它们之间的角度大于某一个角度阈值（经验值）定义为弯曲，小于摸一个阈值（经验值）为伸直。    
  注：这种静态手势识别的方法具有局限性，有条件还是通过模型训练的方法进行静态手势识别。   


  视频示例如下图：   
![gesture](https://gitcode.net/EricLee/handpose_x/-/raw/master/samples/gesture.gif)     

* 4 - 静态手势交互（识别）      
  通过手关键点的二维角度约束关系定义静态手势。     
     

  原理：通过二维约束获得静态手势，该示例是通过 食指伸直（one） 和 握拳（fist）分别代表范围选择和清空选择区域。    
  建议最好还是通过分类模型做静态手势识别鲁棒和准确高，目前局限于静态手势训练集的问题用二维约束关系定义静态手势替代。    

![ocrreco](https://gitcode.net/EricLee/handpose_x/-/raw/master/samples/ocrreco.gif)              

## 项目配置  
### 1、开发环境  
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python  
### 2、硬件  
* 普通USB彩色（RGB）网络摄像头    

## 使用方法  
### 模型训练  
* 根目录下运行命令： python train.py       (注意脚本内相关参数配置 )   

### 模型推理  
* 根目录下运行命令： python inference.py        (注意脚本内相关参数配置 )   

### onnx使用  
* step1: 设定相关配置包括模型类型和模型参数路径，根目录下运行命令： python model2onnx.py        (注意脚本内相关参数配置 )
* step2: 设定onnx模型路径，根目录下运行命令： python onnx_inference.py   (注意脚本内相关参数配置 )  
* 建议    
```

检测手bbox后，进行以下的预处理，crop手图片送入手关键点模型进行推理，   
可以参考 hand_data_iter/datasets.py,数据增强的样本预处理代码部分，   
关键代码如下：     
  img 为原图  ，np为numpy  
  x_min,y_min,x_max,y_max,score = bbox  
  w_ = max(abs(x_max-x_min),abs(y_max-y_min))  

  w_ = w_*1.1  

  x_mid = (x_max+x_min)/2  
  y_mid = (y_max+y_min)/2  

  x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)  

  x1 = np.clip(x1,0,img.shape[1]-1)  
  x2 = np.clip(x2,0,img.shape[1]-1)  

  y1 = np.clip(y1,0,img.shape[0]-1)  
  y2 = np.clip(y2,0,img.shape[0]-1)  

```


