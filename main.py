#-*-coding:utf-8-*-
'''
/-------------------- 手势交互 --------------------/
'''
# date:2024.02.15
# Author: txy
# function: main

import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./components/") # 添加模型组件路径

def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("           << 手势交互 >>         ")
    print("        Apache License 2.0       ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")

if __name__ == '__main__':
    demo_logo()
    parser = argparse.ArgumentParser(description= " DpCas : << Deep Learning Componentized Application System >> ")
    parser.add_argument('-app', type=int, default = 0,
        help = "handpose_x:0, gesture:1 , video_ana:2,  drive:3") # 设置 App Example

    app_dict = {
        0:"handpose_x",
        1:"gesture",
        3:"video_ana",
        4:"drive"}

    args = parser.parse_args()# 解析添加参数

    APP_P = app_dict[args.app]

    if APP_P == "handpose_x": # 手势识别
        from applications.handpose_local_app import main_handpose_x #加载 handpose 应用
        cfg_file = "./lib/hand_lib/cfg/handpose.cfg"
        main_handpose_x(cfg_file)#加载 handpose 应用
    elif APP_P == "gesture": # 手势识别
        from applications.gesture_local_app import main_gesture_x #加载 gesture 应用
        cfg_file = "./lib/gesture_lib/cfg/handpose.cfg"
        main_gesture_x(cfg_file)#加载 handpose 应用


    # elif APP_P == "video_ana":
    #     from applications.VideoAnalysis_app import main_VideoAnalysis #加载 video_analysis 应用
    #     main_VideoAnalysis(video_path = "./video/f3.mp4")#加载 video_analysis  应用
    #
    # elif APP_P == "face_pay":
    #     cfg_file = "./lib/facepay_lib/cfg/facepay.cfg"
    #     from applications.FacePay_local_app import main_facePay #加载 face pay 应用
    #     main_facePay(video_path = 0,cfg_file = cfg_file) # 加载 face pay  应用
    #
    # elif APP_P == "drive":
    #     from applications.DangerousDriveWarning_local_app import main_DangerousDriveWarning #加载 危险驾驶预警 应用
    #     cfg_file = "./lib/dfmonitor_lib/cfg/dfm.cfg"
    #     main_DangerousDriveWarning(video_path = "./video/drive1.mp4",cfg_file = cfg_file)

    print(" well done ~")
