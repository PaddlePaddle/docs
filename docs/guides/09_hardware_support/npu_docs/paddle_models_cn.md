# 飞桨框架昇腾NPU版支持模型

目前Paddle昇腾NPU版基于华为鲲鹏CPU与昇腾NPU支持以下模型的单机单卡/单机多卡的训练。

| 模型               | 领域     | 模型链接                                                   | 编程范式      |  单卡训练支持  |
| ----------------- | -------- | -------------------------------------------------------- | ------------- | -------------- |
| ResNet50  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/ResNet/ResNet50.yaml) |  动态图  | 支持 |
| ResNet18  | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/ResNet/ResNet18.yaml) |  动态图  | 支持 |
| VGG16 | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/VGG/VGG16.yaml) |  动态图  | 支持 |
| AlexNet | 图像分类 | [模型链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/configs/ImageNet/AlexNet/AlexNet.yaml) |  动态图  | 支持 |
| YOLOv3  | 目标检测 | [模型链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/configs/yolov3/yolov3_darknet53_270e_voc.yml) |  动态图  | 支持 |
| BERT  | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert) |  动态图  | 支持 |
| XLNet  | NLP | [模型链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/xlnet) |  动态图  | 支持 |
| DeepFM  | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/deepfm) |  动态图  | 支持 |
| Wide&Deep | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/wide_deep) |  动态图  | 支持 |
| NCF | 推荐系统 | [模型链接](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/recall/ncf) |  动态图  | 支持 |
| DQN  | 强化学习 | [模型链接](https://github.com/PaddlePaddle/PARL/tree/develop/examples/DQN) |  动态图  | 支持 |

## 模型套件

模型放置在飞桨模型套件中，各领域套件是 github.com/PaddlePaddle 下的独立repo，git clone下载即可获取所需的模型文件：

| 领域        | 套件名称        | 分支/版本        |
| ----------- | --------------- | ---------------- |
| 图像分类     | PaddleClas      | release/2.3      |
| 目标检测     | PaddleDetection | release/2.2      |
| 自然语言处理  | PaddleNLP       | develop          |
| 推荐系统     | PaddleRec       | release/2.1.0    |
| 强化学习     | PARL            | develop          |


后续版本将持续增加昇腾NPU在更多模型任务上的验证。
