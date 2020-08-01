# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :YOLOV3_ResNet50_TensorFlow2.0
# @File     :backbone.py
# @Date     :2020/7/13 1:18
# @Author   :dyh
# @Email    :dingyihangheu@163.com
# @Software :PyCharm
# Description :Darknet53

-------------------------------------------------
"""

import tensorflow as tf
import core.common as common


def darknet53(input_data):
    """6+23*2个卷积层"""
    input_data = common.convolutional(input_data, (3, 3,  3,  32))   #[416,416,32]
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)  #[208 208 64]

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)    #[208 208 64] ResNet输入输出一致

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True) #[104 104 128]

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)#[104 104 128]

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True) #[52 52 256]

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)  #[52 52 256]

    route_1 = input_data   #第一次输出  [52 52 256]
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True) #[26 26 512]

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512) #[26 26 512]

    route_2 = input_data   #第二次输出[26 26 512]
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True) #[13 13 1024]

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)  #[13 13 1024] 也是第三次输出

    return route_1, route_2, input_data


