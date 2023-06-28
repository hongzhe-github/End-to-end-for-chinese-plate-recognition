# -*- coding:utf-8 -*-
import numpy as np
import os
from keras import layers, models


def unet_train():
    height = 512
    width = 512
    path = 'D:/desktop/unet_datasets/'
    input_name = os.listdir(path + 'train_image')
    n = len(input_name)
    print(n)
    X_train, y_train = [], []
    for i in range(n):
        print("正在读取第%d张图片" % i)
        img = cv2.imread(path + 'train_image/%d.png' % i)
        label = cv2.imread(path + 'train_label/%d.png' % i)
        X_train.append(img)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    inpt = layers.Input(shape=(height, width, 3))
    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    concat1 = layers.concatenate([conv4, convt1], axis=3)
    concat1 = layers.Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))
    conv6 = Conv2d_BN(conv6, 64, (3, 3))

    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    concat2 = layers.concatenate([conv3, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))
    conv7 = Conv2d_BN(conv7, 32, (3, 3))

    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = layers.concatenate([conv2, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))
    conv8 = Conv2d_BN(conv8, 16, (3, 3))

    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = layers.concatenate([conv1, convt4], axis=3)
    concat4 = layers.Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))
    conv9 = Conv2d_BN(conv9, 8, (3, 3))
    conv9 = layers.Dropout(0.5)(conv9)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(inpt, outpt)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    print("开始训练u-net")
    model.fit(X_train, y_train, epochs=100, batch_size=15)  # epochs和batch_size看个人情况调整，batch_size不要过大，否则内存容易溢出
    # 我11G显存也只能设置15-20左右，我训练最终loss降低至250左右，acc约95%左右
    model.save('unet.h5')
    print('unet.h5保存成功!!!')


import cv2
import numpy as np

def unet_predict(unet, img_src_path):
    '''
    使用U-Net模型对图像进行预测并生成掩膜。

    :param unet: U-Net模型
    :param img_src_path: 图像源路径
    :return: 原始图像和预测的掩膜
    '''
    # 从中文路径读取图像
    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)
    # 或者，如果路径为普通路径，可以使用以下代码读取图像：
    # img_src = cv2.imread(img_src_path)

    # 如果图像形状不是(512, 512, 3)，则进行调整大小
    if img_src.shape != (512, 512, 3):
        img_src = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]

    # 为预测调整图像形状，变为(1, 512, 512, 3)
    img_src = img_src.reshape(1, 512, 512, 3)

    # 使用U-Net模型预测掩膜
    img_mask = unet.predict(img_src)  # 预测前将图像归一化，除以255

    # 将原始图像重新调整为3维
    img_src = img_src.reshape(512, 512, 3)

    # 将预测的掩膜重新调整为3维
    img_mask = img_mask.reshape(512, 512, 3)

    # 对掩膜值进行归一化，并将其缩放到范围[0, 255]
    img_mask = img_mask / np.max(img_mask) * 255

    # 将掩膜的三个通道设置为相同的值
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]

    # 将掩膜转换为无符号整数（int）类型
    img_mask = img_mask.astype(np.uint8)

    return img_src, img_mask

