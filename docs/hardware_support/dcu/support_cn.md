# 海光 DCU 验证模型

飞桨框架在海光 DCU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/tutorials/base/devices_use_guidance.md)

| 模型库 | 模型类型 | 模型名称 | 训练 | 推理 |
| - | - | - | - | - |
| PaddleX | 图像分类 | [ResNet18](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet18.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet34](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet34.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet50.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet101.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet152](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet152.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R101.yaml) | √ | √ |
| PaddleNLP | 语义模型 | [BERT]() | √ | √ |
