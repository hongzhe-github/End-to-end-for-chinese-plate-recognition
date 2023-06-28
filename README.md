# End-to-end-for-chinese-plate-recognition

# 介绍
### 本项目基于u-net，cv2以及cnn的中文车牌定位，矫正和端到端识别软件，其中unet和cv2用于车牌定位和矫正，cnn进行车牌识别，unet和cnn都是基于tensorflow的keras实现。
### 由于原作者的代码晦涩难懂，耦合性太强，于是我在开源项目的基础上进行了大量代码优化，将逻辑代码和UI界面进行完全的分离，提高了代码复用性，将变量名进行重构，将项目结构进行重构，导出了软件包依赖列表，大大提高代码复用性，使得项目移植和重构变得非常简单。
[原项目地址传送](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition)

# 开发环境搭建
### 环境：python:3.7.6, 软件包见项目根目录 `requirements.txt`
## 实现原理
1. 利用u-net图像分割得到二值化图像，
2. 再使用cv2进行边缘检测获得车牌区域坐标，并将车牌图形矫正，
3. 利用卷积神经网络cnn进行车牌多标签端到端识别， 

### 实现效果：拍摄角度倾斜、强曝光或昏暗环境等都能较好地识别，甚至有些百度AI车牌识别未能识别的图片也能识别
### 注意：若是直接识别类似下图的无需定位的完整车牌，那么请确保图片尺寸小于等于240 * 80，否则会被认为图片中含其余区域而进行定位，反而识别效果不佳

### 学习交流微信号:hongzheweixin

## 效果图
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/lic.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/0.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/1.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/2.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/3.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/4.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/5.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/6.png)
![](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition/blob/master/test_pic/7.png)
