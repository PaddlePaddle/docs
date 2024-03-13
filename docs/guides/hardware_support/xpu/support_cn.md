# 昆仑 XPU 支持模型

飞桨框架在昆仑芯 XPU 上经验证的模型的支持情况如下：

## 训练支持

可进行单机单卡/单机多卡训练的模型，如下所示：

| 模型  | 领域  | 编程范式 | 可用的 CPU 类型 | 单机单卡支持 | 单机多卡支持 |
| --- | --- | --- | --- | --- | --- |
| ResNet50 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| ResNet101 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| MobileNet_v3 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| MobileNetV2 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| VGG19 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| VGG16 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| PP-LCNet | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| PP-HGNet | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| InceptionV4 | 图像分类 | 动态图 | X86（Intel） | 支持  | -   |
| UNet | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| deeplabv3 | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| HRNet | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| PP-LiteSeq | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| PP-humansegv2 | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| PP-mating | 图像分割 | 动态图 | X86（Intel） | 支持  | -   |
| MaskRcnn | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| FasterRcnn | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| fairmot | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| Yolov3-DarkNet53 | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| SSD-ResNet34 | 目标检测 | 动态图 | X86（Intel） | 支持  | 支持  |
| Yolov3-mobileNetv1 | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| PPYoloE | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| deepsort | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| ssd-mv1 | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| ssd-vgg16 | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| PP-picoDet | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| PPYolov2 | 目标检测 | 动态图 | X86（Intel） | 支持  | -   |
| OCR-DB | 文字检测 | 动态图 | X86（Intel） | 支持  | -   |
| OCR-crnn | 文字检测 | 动态图 | X86（Intel） | 支持  | -   |
| PPOCR-v2 | 文字检测 | 动态图 | X86（Intel） | 支持  | -   |
| PPOCR-v3 | 文字检测 | 动态图 | X86（Intel） | 支持  | -   |
| Bert-Base | NLP | 静态图 | X86（Intel） | 支持  | 支持  |
| Transformer | NLP | 静态图 | X86（Intel） | 支持  | 支持  |
| GPT-2 | NLP | 动态图 | X86（Intel） | 支持  | -   |
| ernie-base | NLP | 动态图 | X86（Intel） | 支持  | -   |
| ernie 3.0 medium | NLP | 动态图 | X86（Intel） | 支持  | -   |
| lstm | NLP | 动态图 | X86（Intel） | 支持  | -   |
| seq2seq | NLP | 动态图 | X86（Intel） | 支持  | -   |
| DeepFM | 推荐  | 动态图 | X86（Intel） | 支持  | -   |
| Wide&Deep | 推荐  | 动态图 | X86（Intel） | 支持  | -   |
| dlrm | 推荐  | 动态图 | X86（Intel） | 支持  | -   |
| deepspeech2 | 语音识别 | 动态图 | X86（Intel） | 支持  | -   |
| speedyspeech | 语音合成 | 动态图 | X86（Intel） | 支持  | -   |
| dqn | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| ppo | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| ddpg | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| A2C | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| TD3 | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| SAC | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| MADDPG | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| CQL | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| ES  | 强化学习 | 动态图 | X86（Intel） | 支持  | -   |
| pp-tsm | 视频分类 | 动态图 | X86（Intel） | 支持  | -   |

模型放置在飞桨模型套件中，作为 github.com/PaddlePaddle 下的独立 repo 存在，git clone 下载即可获取所需的模型文件：

| 领域  | 套件名称 | 分支/版本 |
| --- | --- | --- |
| 图像分类 | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) | [develop](https://github.com/PaddlePaddle/PaddleClas/tree/develop) |
| 目标检测 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | [develop](https://github.com/PaddlePaddle/PaddleDetection/tree/develop) |
| 图像分割 | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) | [develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop) |
| NLP | [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) | [develop](https://github.com/PaddlePaddle/PaddleNLP/tree/develop) |
| OCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | [dygraph](https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph) |
| 推荐  | [PaddleREC](https://github.com/PaddlePaddle/PaddleRec) | [master](https://github.com/PaddlePaddle/PaddleRec/tree/master) |
