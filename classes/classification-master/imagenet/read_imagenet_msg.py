#-*-coding:utf-8-*-
# date:2021-04-08
# author: Eric.Lee
# function : read imagenet_msg.json

import json

if __name__ == "__main__":
    #读取 imagenet_ms.json文件
    f = open("imagenet_msg.json", encoding='utf-8')
    dict_msg = json.load(f)
    f.close()
    for k_ in dict_msg.keys():
        print("  label {} : {} ".format(k_,dict_msg[k_]))
