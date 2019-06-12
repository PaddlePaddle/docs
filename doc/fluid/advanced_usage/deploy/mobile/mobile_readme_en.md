# Brief Introduction to the Project

<!--[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle-Mobile.svg)](https://github.com/PaddlePaddle/Paddle-Mobile/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)-->


Welcome to Paddle-Mobile GitHub project. Paddle-Mobile is a project of PaddlePaddle as well as a deep learning framework for embedded platforms.

## Features

- high performance in support of ARM CPU
- support Mali GPU
- support Andreno GPU
- support the realization of GPU Metal on Apple devices
- support implementation on ZU5、ZU9 and other FPGA-based development boards
- support implementation on Raspberry Pi and other arm-linux development boards

## Demo
- [ANDROID](https://github.com/xiebaiyuan/paddle-mobile-demo)

### Catalog of original Demo

[https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo](https://github.com/PaddlePaddle/paddle-mobile/tree/develop/demo)

## Documentation

### Documentation of design

If you want to know more details about the documentation of paddle-mobile design, please refer to [documentation of design](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/design_doc.md) . There are many previous designs and discussions: [issue](https://github.com/PaddlePaddle/paddle-mobile/issues).



### Documentation of development

Documentation of development is mainly about building, running and other tasks. As a developer, you can use it with the help of contributed documents.

- [iOS](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_ios.md)
- [Android_CPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android.md)
- [Android_GPU](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_android_GPU.md)
- [FPGA](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_fpga.md)
- [ARM_LINUX](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/doc/development_arm_linux.md)

### How to contribute your documents
- [tutorial link to contribute documents](https://github.com/PaddlePaddle/paddle-mobile/blob/develop/CONTRIBUTING.md)
- Main procedure of contributing code is covered in the document above. If you have other problems during the procedure, please send them as [issue](https://github.com/PaddlePaddle/paddle-mobile/issues). We will deal with it as quickly as possible.


## Acquisition of Models
At present Paddle-Mobile only supports models trained by Paddle fluid. Models can only be operated regularly after transformation if you have models trained by other framworks.
### 1. Use Paddle Fluid directly to train
It is the most reliable method to be recommended
### 2. Transform Caffe to Paddle Fluid model
[https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/caffe2fluid)
### 3. ONNX
ONNX is the acronym of Open Neural Network Exchange. The project is aimed to make a full communication and usage among different neural network development frameworks.

Except for directly using fluid models trained by PaddlePaddle, you can also get certain Paddle fluid models through onnx transformation.

At present，work in support of onnx is also under operation in Baidu. Related transformation project can be referred to here：
[https://github.com/PaddlePaddle/paddle-onnx](https://github.com/PaddlePaddle/paddle-onnx)

### 4. Download parts of testing models and testing pictures
[http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip](http://mms-graph.bj.bcebos.com/paddle-mobile%2FmodelsAndImages.zip)

<!--## Online output of simple search

Gif as following is the application output of online main part detection of simple search app
![ezgif-1-050a733dfb](http://otkwwi4x8.bkt.clouddn.com/2018-07-05-ezgif-1-050a733dfb.gif)-->

## Ask Questions

Welcome to put forward or tackle with our problems. You can post your question in our issue module on github. [Github Issues](https://github.com/PaddlePaddle/paddle-mobile/issues).

## Copyright and License
Paddle-Mobile provides relatively unstrict Apache-2.0 Open source agreement [Apache-2.0 license](LICENSE).


## Old version Mobile-Deep-Learning
Original MDL(Mobile-Deep-Learning) project has been transferred to [Mobile-Deep-Learning](https://github.com/allonli/mobile-deep-learning)
