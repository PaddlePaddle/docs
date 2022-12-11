# 飞桨对昆仑 XPU 芯片的支持

飞桨自 2.0 版本起支持在昆仑 XPU 上运行，经验证的模型训练和预测的支持情况如下：

## 训练支持

可进行单机单卡/单机多卡训练的模型，如下所示：

| 模型               | 领域     | 模型 readme                                                   | 编程范式      | 可用的 CPU 类型           | 单机多卡支持   |
| ------------------ | -------- | ------------------------------------------------------------ | ------------- | ----------------------- | -------------- |
| VGG16/19           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| ResNet50           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）ARM（飞腾） | 支持           |
| MobileNet_v3       | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| HRNet              | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/extension/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| Yolov3-DarkNet53   | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）ARM（飞腾） | 支持           |
| Yolov3-MobileNetv1 | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| PPYOLO             | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| Mask_RCNN          | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| Deeplab_v3         | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/legacy/docs/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| Unet               | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/legacy/docs/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |
| LSTM               | NLP      | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.0/examples/text_classification/rnn) | 静态图/动态图 | X86（Intel）            | 支持           |
| Bert-Base          | NLP      | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.0/examples/language_model/bert/README.md) | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| Ernie-Base         | NLP      |                                                              | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| NAML               | 推荐     | [模型链接](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/naml/train_on_kunlun.md) | 静态图        | X86（Intel）            | 支持           |
| DQN                | 强化学习 | [模型链接](https://github.com/PaddlePaddle/PARL/blob/r1.4.3/examples/DQN/train_on_xpu.md) | 静态图        | X86（Intel）            | 支持           |

模型放置在飞桨模型套件中，作为 github.com/PaddlePaddle 下的独立 repo 存在，git clone 下载即可获取所需的模型文件：

| 领域     | 套件名称        | 分支/版本   |
| -------- | --------------- | ----------- |
| 图像分类 | PaddleClas      | release/2.1 |
| 目标检测 | PaddleDetection | release/2.1 |
| 图像分割 | PaddleSeg       | release/2.1 |
| NLP      | PaddleNLP       | release/2.0 |
| 推荐     | PaddleRec       | release/2.1 |
| 强化学习 | PARL            | >= r1.4     |



## 预测支持

飞桨框架集成了 python 原生预测功能，安装飞桨框架即可使用。
在框架之外，飞桨提供多个高性能预测库，包括 Paddle Inference、Paddle Serving、Paddle Lite 等，支持云边端不同环境下的部署场景，适合相对应的多种硬件平台、操作系统、编程语言，同时提供服务化部署能力。当前预测库验证支持的模型包括：

| 模型                     | 领域     | 编程范式 | 可用的 CPU 类型           |
| ------------------------ | -------- | -------- | ----------------------- |
| VGG16/19                 | 图像分类 | 静态图   | X86（Intel）            |
| ResNet50                 | 图像分类 | 静态图   | X86（Intel）ARM（飞腾） |
| GoogleNet                | 图像分类 | 静态图   | X86（Intel）            |
| yolov3-darknet53         | 目标检测 | 静态图   | X86（Intel）ARM（飞腾） |
| yolov3-mobilenetv1       | 目标检测 | 静态图   | X86（Intel）            |
| ch_ppocr_mobile_v2.0_det | OCR      | 动态图   | X86（Intel）            |
| ch_ppocr_mobile_v2.0_rec | OCR      | 动态图   | X86（Intel）            |
| ch_ppocr_server_v2.0_det | OCR      | 动态图   | X86（Intel）            |
| ch_ppocr_server_v2.0_rec | OCR      | 动态图   | X86（Intel）            |
| LSTM                     | NLP      | 静态图   | X86（Intel）            |
| Bert-Base                | NLP      | 静态图   | X86（Intel）            |
| Ernie-Base               | NLP      | 静态图   | X86（Intel）            |


随着 ARM 架构的高性能、低功耗、低成本的优势日益突显，ARM CPU 更多地进入 PC 和服务器领域，众多新锐国产 CPU 也采用 ARM 架构。在这一趋势下，我们开始尝试在 ARM CPU + 昆仑 XPU 的硬件环境上运行飞桨，当前已验证 ResNet50、YOLOv3 的训练和预测效果。后续版本将持续增加昆仑 XPU 在更多模型任务上的验证。
