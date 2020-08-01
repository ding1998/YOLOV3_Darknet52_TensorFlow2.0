# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :YOLOV3_ResNet50_TensorFlow2.0
# @File     :train
# @Date     :2020/7/16 16:20
# @Author   :dyh
# @Email    :dingyihangheu@163.com
# @Software :PyCharm
# Description :训练文件
-------------------------------------------------
"""
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

trainset = Dataset('train')
logdir = "./data/log"
steps_per_epoch = len(trainset)#每个 epoch 要训练 len(trainset) 次；
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)#用来记录现在是第几次训练；
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch  #在训练过程中我们希望在训练 warmup_steps 次之前学习率有一种变化趋势，在之后有另一种变化趋势；
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch  #训练总次数

# 确定输入张量的shape
input_tensor = tf.keras.layers.Input([416, 416, 3])
# 确定输出张量
conv_tensors = YOLOv3(input_tensor)  # 3个张量（feature map）
output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)   # 未处理的[batch_size, output_size, output_size,255]，表示在 feature map 上的检测框信息。
    output_tensors.append(pred_tensor)   # 处理的[batch_size, output_size, output_size,255]，表示在原始图像上的检测框信息。
# 构建模型
model = tf.keras.Model(input_tensor, output_tensors)
#初始化优化器
optimizer = tf.keras.optimizers.Adam()
#设置保存文件
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)  # 将图片输入模型
        giou_loss = conf_loss = prob_loss = 0
        for i in range(3):  # 3个 feature map
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]  # 包括未经解码处理的输出和已解码处理的输出
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]  # 框回归损失
            conf_loss += loss_items[1]  # 置信度损失
            prob_loss += loss_items[2]  # 分类损失

        total_loss = giou_loss + conf_loss + prob_loss
        # 梯度计算
        gradients = tape.gradient(total_loss, model.trainable_variables)
        # 梯度下降优化
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # 计算学习率（可变的）
        global_steps.assign_add(1)  # global_steps 加 1
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        # 学习率更新到优化器上
        optimizer.lr.assign(lr.numpy())

        # 绘制损失数据
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        train_step(image_data, target)
    model.save_weights("./yolov3")#保存全部模型，

