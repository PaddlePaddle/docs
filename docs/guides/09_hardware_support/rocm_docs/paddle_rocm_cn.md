
# 飞桨框架ROCm版支持模型

目前Paddle ROCm版基于海光CPU(X86)和DCU支持以下模型的单机单卡/单机多卡的训练与推理。

## 图像分类

| 模型               | 领域     | 模型readme                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| ResNet50  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/ResNet_and_vd.md) |  动态图  | 支持 | 支持 | 支持 |
| ResNet101  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/ResNet_and_vd.md) |  动态图  | 支持 | 支持 | 支持 |
| SE_ResNet50_vd  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/SEResNext_and_Res2Net.md) |  动态图  | 支持 | 支持 | 支持 |
| VGG16 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/Others.md) |  动态图  | 支持 | 支持 | 支持 |
| InceptionV4 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/Inception.md) |  动态图  | 支持 | 支持 | 支持 |
| GoogleNet | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/Inception.md) |  动态图  | 支持 | 支持 | 支持 |
| MobileNetV3 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/Mobile.md) |  动态图  | 支持 | 支持 | 支持 |
| AlexNet | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/Others.md) |  动态图  | 支持 | 支持 | 支持 |
| DenseNet121 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/DPN_DenseNet.md) |  动态图  | 支持 | 支持 | 支持 |


## 目标检测

| 模型               | 领域     | 模型readme                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| YOLOv3  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/yolov3) |  动态图  | 支持 | 支持 | 支持 |
| Faster R-CNN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/faster_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Faster R-CNN + FPN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/faster_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Mask R-CNN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Mask R-CNN + FPN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Cascade Faster R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/cascade_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Cascade Mask R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/cascade_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| SSD | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/ssd) |  动态图  | 支持 | 支持 | 支持 |
| BlazeFace | 人脸检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/configs/face_detection) |  动态图  | 支持 | 支持 | 支持 |
| FaceBoxes | 人脸检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/static/docs/featured_model/FACE_DETECTION.md) |  静态图  | 支持 | 支持 | 支持 |
| Libra R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/static/configs/libra_rcnn/README_cn.md) |  静态图  | 支持 | 支持 | 支持 |
| RetinaNet | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/static/configs/retinanet_r50_fpn_1x.yml) |  静态图  | 支持 | 支持 | 支持 |


## 更多分类

| 模型               | 领域     | 模型readme                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| DeepLabV3 | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/deeplabv3) |  动态图  | 支持 | 不支持 | 支持 |
| HRNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fcn) |  动态图  | 支持 | 不支持 | 支持 |
| BERT | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert) |  动态图  | 支持 | 不支持 | 支持 |
| Transformer | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer) |  动态图  | 支持 | 不支持 | 支持 |
| Seq2Seq | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/seq2seq) |  动态图  | 支持 | 不支持 | 支持 |
| Bi-LSTM | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn) |  动态图  | 支持 | 不支持 | 支持 |
| DeepFM | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/deepfm) |  动态图  | 支持 | 不支持 | 支持 |
| Wide&Deep | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep) |  动态图  | 支持 | 不支持 | 支持 |
| Word2Vec | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/recall/word2vec) |  动态图  | 支持 | 不支持 | 支持 |
| NCF | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/recall/ncf) |  动态图  | 支持 | 不支持 | 支持 |
| TSM | 视频分类 | [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/tsm.md) |  动态图  | 支持 | 支持 | 支持 |
| StNet | 视频分类 | [模型链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video/models/stnet) |  动态图  | 支持 | 支持 | 支持 |
| Attention-LSTM | 视频分类| [模型链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video/models/attention_lstm) |  动态图  | 支持 | 支持 | 支持 |
| Transformer TTS | 语音合成 | [模型链接](https://github.com/PaddlePaddle/Parakeet/tree/release/v0.2/examples/transformer_tts) |  动态图  | 支持 | 不支持 | 支持 |
| WaveFlow | 语音合成 | [模型链接](https://github.com/PaddlePaddle/Parakeet/tree/release/v0.2/examples/waveflow) |  动态图  | 支持 | 不支持 | 支持 |

## 模型套件

模型放置在飞桨模型套件中，各领域套件是 github.com/PaddlePaddle 下的独立repo，git clone下载即可获取所需的模型文件：

| 领域     | 套件名称        | 分支/版本        |
| -------- | --------------- | ---------------- |
| 图像分类 | PaddleClas      | release/2.1      |
| 目标检测 | PaddleDetection | release/2.0      |
| 图像分割 | PaddleSeg       | release/v2.0     |
| NLP     | PaddleNLP       | develop          |
| OCR     | PaddleOCR       | release/2.1      |
| 推荐系统 | PaddleRec       | master           |
| 视频分类 | PaddleVideo     | develop          |
| 语音合成 | Parakeet        | release/v0.2     |
