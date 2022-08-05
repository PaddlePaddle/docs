
# 飞桨框架 ROCm 版支持模型

目前 Paddle ROCm 版基于海光 CPU(X86)和 DCU 支持以下模型的单机单卡/单机多卡的训练与推理。

## 图像分类

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| ResNet50  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/ResNet/ResNet50.yaml) |  动态图  | 支持 | 支持 | 支持 |
| ResNet101  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/ResNet/ResNet101.yaml) |  动态图  | 支持 | 支持 | 支持 |
| SE_ResNet50_vd  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/SENet/SE_ResNet50_vd.yaml) |  动态图  | 支持 | 支持 | 支持 |
| VGG16 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/VGG/VGG16.yaml) |  动态图  | 支持 | 支持 | 支持 |
| InceptionV4 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/Inception/InceptionV4.yaml) |  动态图  | 支持 | 支持 | 支持 |
| GoogleNet | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/Inception/GoogLeNet.yaml) |  动态图  | 支持 | 支持 | 支持 |
| MobileNetV3 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml) |  动态图  | 支持 | 支持 | 支持 |
| AlexNet | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/AlexNet/AlexNet.yaml) |  动态图  | 支持 | 支持 | 支持 |
| DenseNet121 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml) |  动态图  | 支持 | 支持 | 支持 |


## 目标检测

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| YOLOv3  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/yolov3) |  动态图  | 支持 | 支持 | 支持 |
| PP-YOLO | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/ppyolo) |  动态图  | 支持 | 支持 | 支持 |
| SSD | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/ssd) |  动态图  | 支持 | 支持 | 支持 |
| RetinaNet | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/static/configs/retinanet_r50_fpn_1x.yml) |  静态图  | 支持 | 支持 | 支持 |
| Faster R-CNN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/faster_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Faster R-CNN + FPN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/faster_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Cascade Faster R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/cascade_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Libra R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/static/configs/libra_rcnn/README_cn.md) |  静态图  | 支持 | 支持 | 支持 |
| Mask R-CNN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Mask R-CNN + FPN  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/mask_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| Cascade Mask R-CNN | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/cascade_rcnn) |  动态图  | 支持 | 支持 | 支持 |
| SOLOv2 | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/solov2) |  动态图  | 支持 | 支持 | 支持 |
| FCOS | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/fcos) |  动态图  | 支持 | 支持 | 支持 |
| TTFNet | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/ttfnet) |  动态图  | 支持 | 支持 | 支持 |
| BlazeFace | 人脸检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/configs/face_detection) |  动态图  | 支持 | 支持 | 支持 |
| FaceBoxes | 人脸检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2/static/docs/featured_model/FACE_DETECTION.md) |  静态图  | 支持 | 支持 | 支持 |

## 图像分割

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| ANN | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/ann) |  动态图  | 支持 | 不支持 | 支持 |
| BiSeNetV2 | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/bisenet) |  动态图  | 支持 | 不支持 | 支持 |
| DANet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/danet) |  动态图  | 支持 | 不支持 | 支持 |
| DeepLabV3+ | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/deeplabv3p) |  动态图  | 支持 | 不支持 | 支持 |
| Fast-SCNN | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fastscnn) |  动态图  | 支持 | 不支持 | 支持 |
| FCN | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fcn) |  动态图  | 支持 | 不支持 | 支持 |
| GCNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/gcnet) |  动态图  | 支持 | 不支持 | 支持 |
| GSCNN | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/gscnn) |  动态图  | 支持 | 不支持 | 支持 |
| HarDNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/hardnet) |  动态图  | 支持 | 不支持 | 支持 |
| OCRNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/ocrnet) |  动态图  | 支持 | 不支持 | 支持 |
| U-Net | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/unet) |  动态图  | 支持 | 不支持 | 支持 |
| DecoupledSegNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/decoupled_segnet) |  动态图  | 支持 | 不支持 | 支持 |
| EMANet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/emanet) |  动态图  | 支持 | 不支持 | 支持 |
| ISANet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/isanet) |  动态图  | 支持 | 不支持 | 支持 |
| DNLNet | 图像分割 | [模型链接](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/dnlnet) |  动态图  | 支持 | 不支持 | 支持 |

## 自然语言处理

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| BERT | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert) |  动态图  | 支持 | 支持 | 支持 |
| XLNet | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/xlnet) |  动态图  | 支持 | 支持 | 支持 |
| ELECTRA | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/electra) |  动态图  | 支持 | 支持 | 支持 |
| Transformer | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer) |  动态图  | 支持 | 支持 | 支持 |
| Seq2Seq | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/seq2seq) |  动态图  | 支持 | 支持 | 支持 |
| TextCNN | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn) |  动态图  | 支持 | 支持 | 支持 |
| Bi-LSTM | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/rnn) |  动态图  | 支持 | 支持 | 支持 |
| LAC | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis) |  动态图  | 支持 | 支持 | 支持 |

## 字符识别

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| OCR-DB | 文本检测 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/detection.md) |  动态图  | 支持 | 支持 | 支持 |
| CRNN-CTC | 文本识别 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/recognition.md) |  动态图  | 支持 | 支持 | 支持 |
| OCR-Clas | 角度分类 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/angle_class.md) |  动态图  | 支持 | 支持 | 支持 |
| OCR-E2E | 字符识别 | [模型链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/pgnet.md) |  动态图  | 支持 | 支持 | 支持 |

## 推荐系统

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| DeepFM | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/deepfm) |  动态图  | 支持 | 不支持 | 支持 |
| Wide&Deep | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/wide_deep) |  动态图  | 支持 | 不支持 | 支持 |
| Word2Vec | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/recall/word2vec) |  动态图  | 支持 | 不支持 | 支持 |
| NCF | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/recall/ncf) |  动态图  | 支持 | 不支持 | 支持 |

## 视频分类

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| SlowFast | 视频分类 | [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md) |  动态图  | 支持 | 支持 | 支持 |
| TSM | 视频分类 | [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/tsm.md) |  动态图  | 支持 | 支持 | 支持 |
| TSN | 视频分类 | [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/tsn.md) |  动态图  | 支持 | 支持 | 支持 |
| Attention-LSTM | 视频分类| [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/attention_lstm.md) |  动态图  | 支持 | 支持 | 支持 |
| BMN | 视频分类 | [模型链接](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md) |  动态图  | 支持 | 支持 | 支持 |
| StNet | 视频分类 | [模型链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video/models/stnet) |  动态图  | 支持 | 支持 | 支持 |

## 语音合成

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| Transformer TTS | 语音合成 | [模型链接](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/transformer_tts) |  动态图  | 支持 | 支持 | 支持 |
| WaveFlow | 语音合成 | [模型链接](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/waveflow) |  动态图  | 支持 | 支持 | 支持 |

## 生成对抗网络

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  训练单机多卡支持  | 训练多机多卡支持  | 推理支持 |
| ----------------- | -------- | ------------------------------------------------------------ | ------------- | -------------- | -------------- | -------------- |
| Pix2Pix | 风格迁移 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/pix2pix_cyclegan.md) |  动态图  | 支持 | 支持 | 支持 |
| CycleGAN | 风格迁移 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/pix2pix_cyclegan.md) |  动态图  | 支持 | 支持 | 支持 |
| StyleGAN V2 | 人脸生成 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/styleganv2.md) |  动态图  | 支持 | 支持 | 支持 |
| Wav2Lip | 唇形合成 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/wav2lip.md) |  动态图  | 支持 | 支持 | 支持 |
| ESRGAN | 图像超分 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md) |  动态图  | 支持 | 支持 | 支持 |
| EDVR | 视频超分 | [模型链接](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md) |  动态图  | 支持 | 支持 | 支持 |

## 模型套件

模型放置在飞桨模型套件中，各领域套件是 github.com/PaddlePaddle 下的独立 repo，git clone 下载即可获取所需的模型文件：

| 领域        | 套件名称        | 分支/版本        |
| ----------- | --------------- | ---------------- |
| 图像分类     | PaddleClas      | release/2.3      |
| 目标检测     | PaddleDetection | release/2.2      |
| 图像分割     | PaddleSeg       | release/v2.0     |
| 自然语言处理  | PaddleNLP       | develop          |
| 字符识别     | PaddleOCR       | release/2.3      |
| 推荐系统     | PaddleRec       | release/2.1.0    |
| 视频分类     | PaddleVideo     | develop          |
| 语音合成     | Parakeet        | develop          |
| 生成对抗网络  | PaddleGAN       | develop          |
