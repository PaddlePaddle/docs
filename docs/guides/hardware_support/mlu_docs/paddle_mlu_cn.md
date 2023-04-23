
# 飞桨框架寒武纪 MLU 版支持模型

目前 Paddle MLU 版基于寒武纪 MLU370 系列板卡支持以下模型的单机单卡/单机多卡的训练。

## 图像分类

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| ResNet50  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/ResNet/ResNet50.yaml) |  动态图  | 支持 | 支持 | 支持 |
| VGG16/19 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/VGG/VGG16.yaml) |  动态图  | 支持 | 支持 | 支持 |
| InceptionV4 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/Inception/InceptionV4.yaml) |  动态图  | 支持 | 支持 | 支持 |
| MobileNetV3 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml) |  动态图  | 支持 | 支持 | 支持 |


## 目标检测

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| YOLOv3  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3) |  动态图  | 支持 | 支持 | 支持 |
| PP-YOLO | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo) |  动态图  | 支持 | 支持 | 支持 |
| SSD | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd) |  动态图  | 支持 | 支持 | 支持 |
| Mask R-CNN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Mask R-CNN + FPN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |


## 图像分割

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| DeepLabV3+ | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/deeplabv3p) |  动态图  | 支持 | 不支持 | 支持 |
| U-Net | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/unet) |  动态图  | 支持 | 不支持 | 支持 |

## 自然语言处理

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| BERT | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert) |  动态图  | 支持 | 支持 | 支持 |
| Transformer | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer) |  动态图  | 支持 | 支持 | 支持 |
| Bi-LSTM | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn) |  动态图  | 支持 | 支持 | 支持 |


## 字符识别

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| OCR-DB | 文本检测 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md) |  动态图  | 支持 | 支持 | 支持 |
| CRNN-CTC | 文本识别 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/recognition.md) |  动态图  | 支持 | 支持 | 支持 |
| OCR-Clas | 角度分类 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/angle_class.md) |  动态图  | 支持 | 支持 | 支持 |
| OCR-E2E | 字符识别 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/pgnet.md) |  动态图  | 支持 | 支持 | 支持 |


## 模型套件

模型放置在飞桨模型套件中，各领域套件是 github.com/PaddlePaddle 下的独立 repo，git clone 下载即可获取所需的模型文件：

| 领域        | 套件名称        | 分支/版本        |
| ----------- | --------------- | ---------------- |
| 图像分类     | PaddleClas      | develop          |
| 目标检测     | PaddleDetection | develop          |
| 图像分割     | PaddleSeg       | develop          |
| 自然语言处理  | PaddleNLP       | develop          |
| 字符识别     | PaddleOCR       | develop          |
