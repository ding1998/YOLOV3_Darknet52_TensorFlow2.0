# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :YOLOV3_ResNet50_TensorFlow2.0
# @File     :common.py
# @Date     :2020/7/12 15:20
# @Author   :dyh
# @Email    :dingyihangheu@163.com
# @Software :PyCharm
# Description :Convolutional 结构、Residual 残差模块和 Upsample 结构
-------------------------------------------------
"""
import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    BN就是通过方法将该层特征值分布重新拉回标准正态分布，特征值将落在激活函数对于
    输入较为敏感的区间，输入的小变化可导致损失函数较大的变化，使得梯度变大，避免
    梯度消失，同时也可加快收敛。
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable) #逻辑操作
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:#向下采样步长为2，[h,w]纬度减小一倍
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)#选择对图片的上和左两个边界填充 0，为了适应步长为2的padding
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    """conv和跳远连接需要保证size相同，只要使用卷积操作就需要保证卷积和被卷积对象channel相同"""
    short_cut = input_layer   #跳远连接
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    """
    相加操作时保证两个 feature map 的宽和高相同
    tf.image.resize改变图片尺寸的大小
    method='nearest'采用最近邻插值
    """
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


