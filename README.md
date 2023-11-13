# End-to-end-for-chinese-plate-recognition

## 介绍：
本项目是一个基于U-Net、OpenCV（cv2）和卷积神经网络（CNN）的中文车牌定位、矫正和端到端识别软件。使用U-Net和OpenCV进行车牌定位和矫正，使用CNN进行车牌识别，而U-Net和CNN都是基于TensorFlow的Keras实现。

为了改进原作者的代码的可读性和可维护性，我进行了大量的代码优化工作。我将逻辑代码和用户界面（UI）完全分离，提高了代码的复用性。我还对变量名进行了重构，并重新组织了项目结构。此外，我还导出了软件包依赖列表，极大地提高了代码的可复用性，使得项目的移植和重构变得非常简单。

原项目地址传送：[项目源地址](https://github.com/duanshengliu/End-to-end-for-chinese-plate-recognition)

## 开发环境搭建：
开发环境要求：Python 3.7.6，软件包要求请参见项目根目录下的requirements.txt文件。

## 实现原理：
该软件的实现原理如下：
1. 使用U-Net进行图像分割，得到二值化图像。
2. 使用OpenCV进行边缘检测，获取车牌区域的坐标，并对车牌图像进行矫正。
3. 使用卷积神经网络（CNN）进行车牌的多标签端到端识别。

## 实现效果：
该软件能够较好地识别拍摄角度倾斜、强曝光或昏暗环境等条件下的车牌图像，甚至对于一些百度AI车牌识别无法识别的图片也能进行有效识别。

注意事项：
如果要直接识别类似下图所示的完整车牌图像（无需定位），请确保图像尺寸小于或等于240x80像素，否则会被误认为图像中包含其他区域而进行定位，从而导致识别效果不佳。

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
