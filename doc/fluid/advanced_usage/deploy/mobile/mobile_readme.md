# 项目简介

<!--[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Mobile.svg)](https://github.com/PaddlePaddle/Paddle-Mobile/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)-->


欢迎来到 Paddle-Mobile GitHub 项目。Paddle-Mobile是PaddlePaddle组织下的项目，是一个致力于嵌入式平台的深度学习的框架

## Features

- 高性能支持ARM CPU
- 支持Mali GPU
- 支持Andreno GPU
- 支持苹果设备的GPU Metal实现
- 支持ZU5、ZU9等FPGA开发板
- 支持树莓派等arm-linux开发板

## Demo
[ANDROID](https://github.com/xiebaiyuan/paddle-mobile-demo)

### 原Domo目录

请参考这里[这里](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo)

## 文档

### 设计文档

关于paddle-mobile设计文档请参考[这里](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md)，如果想了解更多内容，[Issue](https://github.com/PaddlePaddle/paddle-mobile/issues)中会有很多早期的设计和讨论过程


### 开发文档

开发文档主要是关于编译、运行等问题。作为开发者，它可以和贡献文档共同结合使用

* [iOS](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_ios.md)
* [Android_CPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android.md)
* [Android_GPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android_GPU.md)
* [FPGA](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_fpga.md)
* [ARM_LINUX](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_arm_linux.md)

### 贡献代码

- [贡献代码](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md)

- 上面文档中涵盖了主要的贡献代码流程，如果在实践中您还遇到了其他问题，可以发[Issue](https://github.com/PaddlePaddle/paddle-mobile/issues)。我们看到后会尽快处理


## 模型获得
目前Paddle-Mobile仅支持Paddle fluid训练的模型。如果你手中的模型是不同种类的模型，需要进行模型转换才可以运行

### 1. 直接使用Paddle Fluid训练

该方式最为可靠，推荐方式

### 2. Caffe转为Paddle Fluid模型

请参考这里[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/caffe2fluid)

### 3. ONNX

ONNX全称为“Open Neural Network Exchange”，即“开放的神经网络切换”，该项目的目的是让不同的神经网络开发框架做到互通互用

除直接使用PaddlePaddle训练fluid版本的模型外，还可以通过onnx转换得到个别Paddle Fluid模型

目前，百度也在做onnx支持工作。相关转换项目在[这里](https://github.com/PaddlePaddle/paddle-onnx)

### 4. 部分测试模型和测试图片下载

[下载链接](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip)

- 测试输入数据可由本仓库下的脚本`tools/python/imagetools`生成。

## 交流与反馈
- 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/Paddle/issues)来提交问题、报告与建议
- QQ群: 696965088 (Paddle-Mobile)
- [论坛](http://ai.baidu.com/forum/topic/list/168): 欢迎大家在PaddlePaddle论坛分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围

## Copyright and License
Paddle-Mobile 提供相对宽松的Apache-2.0开源协议 [Apache-2.0 license](LICENSE)


## 旧版 Mobile-Deep-Learning
原MDL(Mobile-Deep-Learning)工程被迁移到了这里 [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning)

