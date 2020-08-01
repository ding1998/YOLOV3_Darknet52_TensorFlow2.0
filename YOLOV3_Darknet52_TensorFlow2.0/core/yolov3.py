# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :YOLOV3_ResNet50_TensorFlow2.0
# @File     :yolov3.py
# @Date     :2020/7/13 1:51
# @Author   :dyh
# @Email    :dingyihangheu@163.com
# @Software :PyCharm
# Description :YOLOv3（Darknet53的后半部分）、decode（网络解码）、Iou、Giou、Loss

-------------------------------------------------
"""

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES)) #类别
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)  #提前设定的锚框
STRIDES         = np.array(cfg.YOLO.STRIDES)  #三个 feature map 上单位长度所代表的原始图像长度？？？
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH  #交并比阈值

def YOLOv3(input_layer):
    """
    convolutional中padding在没有下采样的情况下采用了same，所以不用考虑[h w]变化
    将Darknet53的三个route_1, route_2, conv，经过再次卷积以及合并等得到三个大中小box的输出
    最后的通道均为每个输出的通道数都是 3x(NUM_CLASS+5)，每个检测框需要有 (x, y, w, h, confidence)
    五个基本参数，然后还要有类别个数个概率
    这一部分也是Darknet53 的部分
    """
    route_1, route_2, conv = backbone.darknet53(input_layer)   #Darknet53  [52 52 256]   [26 26 512]    #[13 13 1024]

    conv = common.convolutional(conv, (1, 1, 1024,  512))   #[13 13 512]
    conv = common.convolutional(conv, (3, 3,  512, 1024))   #[13 13 1024]
    conv = common.convolutional(conv, (1, 1, 1024,  512))   #[13 13 512]
    conv = common.convolutional(conv, (3, 3,  512, 1024))   #[13 13 1024]
    conv = common.convolutional(conv, (1, 1, 1024,  512))   #[13 13 512]

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024)) #[13 13 1024]
    #第一个输出大的目标的Box
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1,  512,  256))
    conv = common.upsample(conv)
    #上一次输出与route_2合并，在最后一个纬度
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256)) #5次卷积输出[26 26 256]

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512)) #[26 26 512]
    # 第二个输出中等的目标的Box
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))#[52 52 256]
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def decode(conv_output, i=0):
    """
    作用：解码 YOLOv3 网络的输出，Input：YOLOv3 网络的输出三个 feature map 中的一个
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]  # 样本数
    output_size      = conv_shape[1]  # 矩阵大小

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    #Reshape 将最后一个纬度，即通道拆开[3,5 + NUM_CLASS]

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # 中心位置的偏移量
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # 预测框长宽的偏移量
    conv_raw_conf = conv_output[:, :, :, :, 4:5]  # 预测框的置信度
    conv_raw_prob = conv_output[:, :, :, :, 5:]  # 预测框的类别概率

    # 1.对每个先验框生成在 feature map 上的相对坐标，以左上角为基准，其坐标单位为格子，即数值表示第几个格子
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])   #tf.newaxis是用来做张量维数扩展的 shape=(52, 52)
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1]) # shape=(52, 52)

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)  # shape=(52, 52, 2)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])# shape=(1, 52, 52, 3, 2)
    xy_grid = tf.cast(xy_grid, tf.float32)

    # 2，计算预测框的绝对坐标以及宽高度
    #中心
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i] # xy_grid表示 feature map 中左上角的位置，即是第几行第几个格子；STRIDES表示格子的长度，即 feature map 中一个格子在原始图像上的长度
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]  # ANCHORS[i] * STRIDES[i]表示先验框在原始图像中的大小
    #宽度和高度
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # 3. 计算预测框的置信值和分类值
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    """交并比"""
    ## 两个检测框的面积
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  #[x,y,h,w]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1) # 第一个检测框的左上角坐标+右下角坐标
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # left_up=[xmin2, ymin2]
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])  # right_down=[xmax1, ymax1]

    inter_section = tf.maximum(right_down - left_up, 0.0)  # 交集区域
    inter_area = inter_section[..., 0] * inter_section[..., 1]  # 交集面积
    union_area = boxes1_area + boxes2_area - inter_area  # 并集面积

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    """ GIoU """
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):

    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size  # 原始图片尺寸
    input_size = tf.cast(input_size, tf.float32)
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 模型输出的置信值与分类
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    pred_xywh = pred[:, :, :, :, 0:4]  # 模型输出处理后预测框的位置
    pred_conf = pred[:, :, :, :, 4:5]  # 模型输出处理后预测框的置信值
    label_xywh = label[:, :, :, :, 0:4]  # 标签图片的标注框位置
    respond_bbox = label[:, :, :, :, 4:5]  # 标签图片的置信值，有目标的为1 没有目标为0
    label_prob = label[:, :, :, :, 5:]  # 标签图片的分类

# 1、框回归损失
	# 计算检测框和真实框的 GIOU 值
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    # bbox_loss_scale 制衡误差 2-w*h
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

# 2、置信度损失
    # 生成负样本
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # [batch_size, output_size, output_size, 1, 1]

    # respond_bgd 形状为 [batch_size, output_size, output_size, anchor_per_scale, x]，当无目标且小于阈值时x为1，否则为0
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            # 正样本误差
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            # 负样本误差
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

# 3.分类损失
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # 误差平均
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss


