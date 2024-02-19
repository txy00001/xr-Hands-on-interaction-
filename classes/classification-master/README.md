# classification    
物体识别分类   

## 项目介绍    
该项目对物体进行识别分类。  

## 项目配置   
* 作者开发环境：   
* Python 3.7   
* PyTorch >= 1.5.1   

## 预训练模型    
### 1、Stanford Dogs 预训练模型
* [预训练模型下载地址(百度网盘 Password: ks87 )](https://pan.baidu.com/s/1tT0wF4N2I9p5JDfCwtM1CQ)   

### 2、静态手势识别预训练模型（handpose_x_gesture_v1）    
* [预训练模型下载地址(百度网盘 Password: igcf )](https://pan.baidu.com/s/1WeoYQ3bfTkpbzPbROm81Ew)   

### 3、imagenet 预训练模型
* [预训练模型下载地址(百度网盘 Password: ct31 )](https://pan.baidu.com/s/1uZsAHF6wK-LOR8j6TFABmQ)   
* 具体分类看json信息即"imagenet_msg.json"，运行 [read_imagenet_msg.py](https://codechina.csdn.net/EricLee/classification/-/blob/master/imagenet/read_imagenet_msg.py) 读取。
* "chinese_name"为类别中文名字，"doc_name"为数据集对应的每一类文件夹名字，前面的数字为模型的类别号从 "0"~"999"，共 1000 类 。

### 4、Stanford_Cars 预训练模型
* [预训练模型下载地址(百度网盘 Password: 7bf7 )](https://pan.baidu.com/s/1JY_ia48e92am6JJ_p-kgQg)

## 项目使用方法  
### 模型训练  
   注意: train.py 中的 3个参数与具体分类任务数据集，息息相关，如下所示：
```
    #---------------------------------------------------------------------------------
    parser.add_argument('--train_path', type=str, default = './handpose_x_gesture_v1/',
        help = 'train_path') # 训练集路径
    parser.add_argument('--num_classes', type=int , default = 14,
        help = 'num_classes') #  分类类别个数,gesture 配置为 14 ， Stanford Dogs 配置为 120 ， imagenet 配置为 1000
    parser.add_argument('--have_label_file', type=bool, default = False,
        help = 'have_label_file') # 是否有配套的标注文件解析才能生成分类训练样本，gesture 配置为 False ， Stanford Dogs 配置为 True
```
* 根目录下运行命令： python train.py       (注意脚本内相关参数配置 )   

### 模型推理  
* 根目录下运行命令： python inference.py        (注意脚本内相关参数配置 )   
  


  
