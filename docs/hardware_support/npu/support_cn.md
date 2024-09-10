# 昇腾 NPU 验证模型

飞桨框架在昇腾 NPU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/tutorials/base/devices_use_guidance.md)
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
| PaddleX | 图像分类 | [MobileNetV2_x0_25](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV2_x0_25.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV2_x0_5](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV2_x0_5.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV2_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/MobileNetV2_x1_0.yaml) | √ | √ |
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
| PaddleX | 图像分类 | [PP-HGNetV2-B0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-HGNetV2-B0.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNetV2-B4](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-HGNetV2-B4.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNetV2-B6](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/PP-HGNetV2-B6.yaml) | √ | √ |
| PaddleX | 图像分类 | [SwinTransformer_base_patch4_window7_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/SwinTransformer_base_patch4_window7_224.yaml) | √ | √ |
| PaddleX | 图像分类 | [ConvNeXt_tiny](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ConvNeXt_tiny.yaml) | √ | √ |
| PaddleX | 图像分类 | [CLIP_vit_base_patch16_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/CLIP_vit_base_patch16_224.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-M.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-X.yaml) | √ | √ |
| PaddleX | 目标检测 | [RT-DETR-R18](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-R18.yaml) | √ | √ |
| PaddleX | 目标检测 | [RT-DETR-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-R50.yaml) | √ | √ |
| PaddleX | 目标检测 | [RT-DETR-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [RT-DETR-H](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-H.yaml) | √ | √ |
| PaddleX | 目标检测 | [RT-DETR-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-X.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-L.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3-R50.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3-R101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3-R101.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R101.yaml) | √ | √ |
| PaddleX | 语义分割 | [PP-LiteSeg-T](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/PP-LiteSeg-T.yaml) | √ | √ |
| PaddleX | 语义分割 | [OCRNet_HRNet-W48](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/OCRNet_HRNet-W48.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_server_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_server_det.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_mobile_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_server_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_mobile_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml) | √ | √ |
| PaddleX | 版面分析 | [PicoDet_layout_1x](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/structure_analysis/PicoDet_layout_1x.yaml) | √ | √ |
| PaddleX | 时序预测 | [DLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/DLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [RLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/RLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [NLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/NLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [TimesNet](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/TimesNet.yaml) | √ | √ |
| PaddleX | 时序预测 | [Nonstationary](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/Nonstationary.yaml) | √ | √ |
| PaddleX | 时序预测 | [TiDE](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/ts_forecast/TiDE.yaml) | √ | √ |
| PaddleNLP | 语义模型 | [BERT]() | √ | √ |
| PaddleNLP | 大语言模型 | [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/npu/llama) | √ | √ |
