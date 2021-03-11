飞桨2.0.1版本支持昆仑XPU运行训练和预测。

## 训练支持

目前基于昆仑XPU和X86（Intel）CPU可实现12个模型单机单卡/单机多卡的训练，如下图所示：

| 模型               | 领域     | 模型readme                                                   | 编程范式      | 可用的CPU类型           | 单机多卡支持   |
| ------------------ | -------- | ------------------------------------------------------------ | ------------- | ----------------------- | -------------- |
| VGG16/19           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| ResNet50           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）ARM（飞腾） | 支持           |
| MobileNet_v3       | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| HRNet              | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| Yolov3-DarkNet53   | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| Yolov3-MobileNetv1 | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| Mask_RCNN          | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| Deeplab_v3         | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/legacy/docs/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| Unet               | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/legacy/docs/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| Bert-Base          | NLP      | [模型链接](https://github.com/PaddlePaddle/models/blob/6bb6834cf399254d59d67cf9f9d4c92b41eb6678/PaddleNLP/examples/language_model/bert/README.md) | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| Ernie-Base         | NLP      |                                                              | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| DQN                | 强化学习 | [模型链接](https://github.com/PaddlePaddle/PARL/blob/develop/examples/DQN/README.md) | 静态图        | X86（Intel）            | 支持           |

模型放置在飞桨模型套件中，各领域套件是 github.com/PaddlePaddle 下的独立repo，git clone下载即可获取所需的模型文件：

| 领域     | 套件名称        | 分支/版本        |
| -------- | --------------- | ---------------- |
| 图像分类 | PaddleClas      | release/2.0      |
| 目标检测 | PaddleDetection | release/2.0-beta |
| 图像分割 | PaddleSeg       | release/2.0-beta |
| NLP      | models          | develop          |
| 强化学习 | PARL            | r1.4             |

随着ARM架构的高性能、低功耗、低成本的优势日益突显，ARM CPU更多地进入PC和服务器领域，众多新锐国产CPU也纷纷采用ARM架构。在这一趋势下，我们开始尝试在飞腾CPU和昆仑XPU上运行飞桨，当前已验证ResNet50的训练效果。

## 预测支持

飞桨框架集成了基于python的常规原生预测功能，安装飞桨框架即可使用。
在框架之外，飞桨提供多个高性能预测库，支持云边端不同环境下的部署场景，适合相对应的多种硬件平台、操作系统、编程语言，同时提供服务化部署能力。当前预测库通过验证支持的模型包括：

| 模型       | 领域     | 模型下载                                                     | 编程范式 | 可用的CPU类型           |
| ---------- | -------- | ------------------------------------------------------------ | -------- | ----------------------- |
| VGG19      | 图像分类 | [模型链接](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/VGG19.tar.gz) | 静态图   | X86（Intel）            |
| ResNet50   | 图像分类 | [模型链接](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/resnet50.tar.gz) | 静态图   | X86（Intel）ARM（飞腾） |
| GoogleNet  | 图像分类 | [模型链接](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/GoogLeNet.tar.gz) | 静态图   | X86（Intel）            |
| Bert-Base  | NLP      | [模型链接](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/bert.tar.gz) | 静态图   | X86（Intel）            |
| Ernie-Base | NLP      | [模型链接](http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests/ernie.tar.gz) | 静态图   | X86（Intel）            |

更多的模型及动态图开发等能力将在后续版本增加。

