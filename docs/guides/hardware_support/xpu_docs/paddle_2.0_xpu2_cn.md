# 飞桨对昆仑 2 代芯片的支持

飞桨自 2.3rc 版本起支持在昆仑 2 代芯片上（R200，R300）运行，经验证的模型训练的支持情况如下：

## 训练支持

可进行单机单卡/单机多卡训练的模型，如下所示：

| 模型               | 领域     | 编程范式      | 可用的 CPU 类型           | 单机单卡支持   |  单机多卡支持   |
| ------------------ | -------- |------------- | ----------------------- | -------------- | -------------- |
| ResNet50           | 图像分类 | 动态图        | X86（Intel）            | 支持           |-           |
| MobileNet_v3       | 图像分类 | 动态图        | X86（Intel）            | 支持           |-           |
| UNet               | 图像分割 | 动态图        | X86（Intel）            | 支持           |-           |
| Yolov3-DarkNet53   | 目标检测 | 动态图        | X86（Intel）            | 支持           |-           |
| SSD-ResNet34       | 目标检测 | 动态图        | X86（Intel）            | 支持           |支持         |
| OCR-DB             | 文字检测 | 动态图        | X86（Intel）            | 支持           |-           |
| Bert-Base          | NLP     | 静态图        | X86（Intel）            | 支持           |支持         |
| Transformer        | NLP     | 静态图        | X86（Intel）            | 支持           |支持         |
| GPT-2              | NLP     | 动态图        | X86（Intel）            | 支持           |-           |
| DeepFM             | 推荐    | 动态图        | X86（Intel）             | 支持           |-           |
| Wide&Deep          | 推荐    | 动态图        | X86（Intel）             | 支持           |-           |



模型放置在飞桨模型套件中，作为 github.com/PaddlePaddle 下的独立 repo 存在，git clone 下载即可获取所需的模型文件：

| 领域     | 套件名称        | 分支/版本   |
| -------- | --------------- | ----------- |
| 图像分类 | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)      | [develop](https://github.com/PaddlePaddle/PaddleClas/tree/develop) |
| 目标检测 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | [develop](https://github.com/PaddlePaddle/PaddleDetection/tree/develop) |
| 图像分割 | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)       | [develop](https://github.com/PaddlePaddle/PaddleSeg/tree/develop) |
| NLP     | [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)       | [develop](https://github.com/PaddlePaddle/PaddleNLP/tree/develop) |
| OCR     | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)       | [dygraph](https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph) |
| 推荐     | [PaddleREC](https://github.com/PaddlePaddle/PaddleRec)       | [master](https://github.com/PaddlePaddle/PaddleRec/tree/master) |

* 注：支持基于 Kermel Primitive 算子的昆仑 2 代芯片支持，[点击这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/kernel_primitive_api/index_cn.html)。
