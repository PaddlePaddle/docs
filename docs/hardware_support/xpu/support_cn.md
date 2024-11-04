# 昆仑芯 XPU 支持模型

飞桨框架在昆仑芯 XPU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/other_devices_support/multi_devices_use_guide.md)
* PaddleNLP 大语言模型多硬件使用文档详见：[PaddleNLP XPU 大语言模型使用文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/xpu)
* 如果您适配/验证过更多模型，欢迎向飞桨开源社区贡献适配代码，然后邮件联系我们更新本列表 [ext_paddle_oss](ext_paddle_oss@baidu.com)

| 模型库 | 模型类型 | 模型名称 | 训练 | 推理 |
| - | - | - | - | - |
| PaddleX | 图像分类 | [ResNet18](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet18.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet34](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet34.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet50.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet101.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet152](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet152.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x0_25](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x0_25.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x0_35](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x0_35.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x0_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x0_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x0_75](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x0_75.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x1_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x1_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x1_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x2_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x2_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x2_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-LCNet_x2_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x0_35](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_small_x0_35.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x0_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_small_x0_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x0_75](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_small_x0_75.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_small_x1_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x1_25](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_small_x1_25.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_large_x0_35](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_large_x0_35.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_large_x0_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_large_x0_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_large_x0_75](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_large_x0_75.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_large_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_large_x1_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_large_x1_25](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV3_large_x1_25.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNet_small](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-HGNet_small.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-M.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-X.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-L.yaml) | √ | √ |
| PaddleX | 语义分割 | [PP-LiteSeg-T](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/PP-LiteSeg-T.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_server_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_server_det.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_mobile_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_server_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_mobile_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml) | √ | √ |
| PaddleX | 版面分析 | [PicoDet_layout_1x](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/structure_analysis/PicoDet_layout_1x.yaml) | √ | √ |
| PaddleX | 时序预测 | [DLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/DLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [RLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/RLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [NLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/NLinear.yaml) | √ | √ |
| PaddleNLP | 自然语言理解模型 | [BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/legacy/model_zoo/bert) | √ | √ |
| PaddleNLP | 自然语言理解模型 | [ERINE3.0](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/legacy/model_zoo/ernie-3.0/configs/default.yml) | √ | √ |
| PaddleNLP | 大语言模型 | [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/xpu/llama) | √ | √ |
