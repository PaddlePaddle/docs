# 飞桨对昆仑XPU芯片的支持

自飞桨2.0版本起支持昆仑XPU，目前基于昆仑XPU和X86（Intel）CPU可实现12个模型单机单卡/单机多卡的训练，如下图所示：

| 模型               | 领域     | 模型readme | 编程范式      | 可用的CPU类型           | 单机多卡支持   |
| ------------------ | -------- | ---------- | ------------- | ----------------------- | -------------- |
| VGG16/19           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md)   | 静态图        | X86（Intel）            | 支持           |
| ResNet50           | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md)    | 静态图        | X86（Intel）ARM（飞腾） | 支持           |
| MobileNet_v3       | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md)    | 静态图        | X86（Intel）            | 支持           |
| HRNet              | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/docs/zh_CN/extension/train_on_xpu.md)    | 静态图        | X86（Intel）            | 支持           |
| Yolov3-DarkNet53   | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md)    | 静态图        | X86（Intel）            | 支持           |
| Yolov3-MobileNetv1 | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md)   | 静态图        | X86（Intel）            | 支持           |
| Mask_RCNN          | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/train_on_kunlun.md)   | 静态图        | X86（Intel）            | 支持           |
| Deeplab_v3         | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/legacy/docs/train_on_xpu.md)    | 静态图        | X86（Intel）            | 支持           |
| Unet               | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/legacy/docs/train_on_xpu.md)    | 静态图        | X86（Intel）            | 支持           |
| Bert-Base          | NLP      | [模型链接](https://github.com/PaddlePaddle/models/blob/6bb6834cf399254d59d67cf9f9d4c92b41eb6678/PaddleNLP/examples/language_model/bert/README.md)    | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| Ernie-Base         | NLP      |    | 静态图/动态图 | X86（Intel）            | 支持（静态图） |
| DQN                | 强化学习 | [模型链接](https://github.com/PaddlePaddle/PARL/blob/develop/examples/DQN/README.md)    | 静态图        | X86（Intel）            | 支持           |
