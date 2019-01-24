# Brief Introduction of Project

<!--[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Mobile.svg)](https://github.com/PaddlePaddle/Paddle-Mobile/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)-->


Welcome to Paddle-Mobile GitHub project.Paddle-Mobile,a project of PaddlePaddle Organization,is a deep learning framework for embeded platform.

## Features

- Support ARM CPU with hign performance 
- Support Mali GPU
- Support Andreno GPU
- Support GPU Metal of Apple devices
- Support FPGA demoboard like ZU5、ZU9
- Support arm-linux demoboard like raspberry pi

## Demo
[ANDROID](https://github.com/xiebaiyuan/paddle-mobile-demo).

### Primary Domo Directory

Please refer to [here](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo).

## Document

### Design Document

About design document of paddle-mobile,please refer to [here](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md).For more details,please refer to [Issue](https://github.com/PaddlePaddle/paddle-mobile/issues) to know more about previous design and discussion.


### Development Document

Development document is mainly about build,operation and other problems.As a developer,you can combine the development document with contribution document.

* [iOS](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_ios.md).
* [Android_CPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android.md).
* [Android_GPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android_GPU.md).
* [FPGA](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_fpga.md).
* [ARM_LINUX](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_arm_linux.md).

### Contribute Code

- [Contribute Code](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md).

- Document above  introduces main process to contribute code.If you come across other problems,you can send [Issue](https://github.com/PaddlePaddle/paddle-mobile/issues). We will deal with them as soon as possible once we receieve them.


## Get Model
At present Paddle-Mobile only supports models trained by Paddle fluid.If your model is other kinds of model,you need to transform the model to make it operated normally.

### 1. Directly Use Paddle Fluid to Train

It's the most reliable method(recomended).

### 2. Caffe Transformed as Paddle Fluid Model

Please refer to [here](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/caffe2fluid).

### 3. ONNX

The full name of is “Open Neural Network Exchange” which is “开放的神经网络切换” in Chinese. The aim of the project is to share developing frameworks of different neural networks.

Except for directly using PaddlePaddle to train model in fluid,we can also get certain Paddle Fluid models with transformation of onnx.

At present,Baidu is also supporting onnx.Related projects can be found [here](https://github.com/PaddlePaddle/paddle-onnx).

### 4. Download Part of Test Models and Pictures

[Download Link](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip).

## Q&A

Welcome to come up with or solve our problems.Please send [Issue](https://github.com/PaddlePaddle/paddle-mobile/issues) if you have any question.

## Copyright and License

Paddle-Mobile provides relatively unstrict Apache-2.0 opensource license [Apache-2.0 license](LICENSE).


## Old Mobile-Deep-Learning
Primary MDL(Mobile-Deep-Learning) project is transferred to [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning) .
