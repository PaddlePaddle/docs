# 昆仑芯 XPU 支持模型

飞桨框架在昆仑芯 XPU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/other_devices_support/multi_devices_use_guide.md)
* PaddleNLP 大语言模型多硬件使用文档详见：[PaddleNLP XPU 大语言模型使用文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/xpu)
* 如果您适配/验证过更多模型，欢迎向飞桨开源社区贡献适配代码，然后邮件联系我们更新本列表 [ext_paddle_oss](ext_paddle_oss@baidu.com)

| 模型库 | 模型类型 | 模型名称 | 训练 | 推理 |
| - | - | - | - | - |
| PaddleX | 图像分类 | [ResNet50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet50.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-LCNet_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [MobileNetV3_small_x1_0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x1_0.yaml) | √ | √ |
| PaddleX | 图像分类 | [SwinTransformer_small_patch4_window7_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_small_patch4_window7_224.yaml) | √ | √ |
| PaddleX | 图像分类 | [SwinTransformer_base_patch4_window7_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_base_patch4_window7_224.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNet_small](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_small.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNetV2-B0](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B0.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNetV2-B4](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B4.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-HGNetV2-B6](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B6.yaml) | √ | √ |
| PaddleX | 图像分类 | [CLIP_vit_base_patch16_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_base_patch16_224.yaml) | √ | √ |
| PaddleX | 图像分类 | [CLIP_vit_large_patch14_224](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_large_patch14_224.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-ShiTuV2_rec_CLIP_vit_base](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_base.yaml) | √ | √ |
| PaddleX | 图像分类 | [PP-ShiTuV2_rec_CLIP_vit_large](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_large.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-M.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-X.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [STFPM](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-L.yaml) | √ | √ |
| PaddleX | 语义分割 | [PP-LiteSeg-T](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_server_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_mobile_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_server_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_mobile_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml) | √ | √ |
| PaddleX | 文本识别 | [ch_SVTRv2_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_SVTRv2_rec.yaml) | √ | √ |
| PaddleX | 版面分析 | [PicoDet_layout_1x](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x.yaml) | √ | √ |
| PaddleX | 公式识别 | [LaTeX_OCR_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/LaTeX_OCR_rec.yaml) | √ | √ |
| PaddleX | 图像异常检测 | [STFPM](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/anomaly_detection/STFPM.yaml) | √ | √ |
| PaddleX | 时序预测 | [DLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/DLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [RLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/RLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [NLinear](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/NLinear.yaml) | √ | √ |
| PaddleX | 时序预测 | [PatchTST](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/PatchTST.yaml) | √ | √ |
| PaddleX | 时序预测 | [TimesNet](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TimesNet.yaml) | √ | √ |
| PaddleX | 时序预测 | [TiDE](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TiDE.yaml) | √ | √ |
| PaddleX | 时序异常检测 | [DLinear_ad](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/DLinear_ad.yaml) | √ | √ |
| PaddleX | 时序异常检测 | [PatchTST_ad](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml) | √ | √ |
| PaddleX | 时序异常检测 | [TimesNet_ad](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/TimesNet_ad.yaml) | √ | √ |
| PaddleX | 时序异常检测 | [Nonstationary_ad](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/Nonstationary_ad.yaml) | √ | √ |
| PaddleX | 时序异常检测 | [AutoEncoder_ad](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml) | √ | √ |
| PaddleX | 时序分类 | [TimesNet_cls](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_classification/TimesNet_cls.yaml) | √ | √ |
| PaddleNLP | 自然语言理解模型 | [BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/bert) | √ | √ |
