# 海光 DCU 验证模型

飞桨框架在海光 DCU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/other_devices_support/multi_devices_use_guide.md)
* 如果您适配/验证过更多模型，欢迎向飞桨开源社区贡献适配代码，然后邮件联系我们更新本列表 [ext_paddle_oss](ext_paddle_oss@baidu.com)

| 模型库 | 模型类型 | 模型名称 | 训练 | 推理 |
| - | - | - | - | - |
| PaddleX | 图像分类 | [ResNet18](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet18.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet34](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet34.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet50.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet101.yaml) | √ | √ |
| PaddleX | 图像分类 | [ResNet152](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet152.yaml) | √ | √ |
| PaddleX | 图像多标签分类 | [CLIP_vit_base_patch16_448_ML](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/multilabel_classification/CLIP_vit_base_patch16_448_ML.yaml) | √ | √ |
| PaddleX | 图像多标签分类 | [PP-HGNetV2-B0_ML](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/multilabel_classification/PP-HGNetV2-B0_ML.yaml) | √ | √ |
| PaddleX | 图像多标签分类 | [PP-HGNetV2-B4_ML](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/multilabel_classification/PP-HGNetV2-B4_ML.yaml) | √ | √ |
| PaddleX | 图像多标签分类 | [PP-HGNetV2-B6_ML](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/multilabel_classification/PP-HGNetV2-B6_ML.yaml) | √ | √ |
| PaddleX | 图像特征 | [PP-ShiTuV2_rec_CLIP_vit_base](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/general_recognition/PP-ShiTuV2_rec_CLIP_vit_base.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-XS](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-XS.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-M.yaml) | √ | √ |
| PaddleX | 目标检测 | [PicoDet-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PicoDet-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-M.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml) | √ | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-X.yaml) | √ | √ |
| PaddleX | 小目标检测 | [PP-YOLOE_plus_SOD-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml) | √ | √ |
| PaddleX | 小目标检测 | [PP-YOLOE_plus_SOD-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-L.yaml) | √ | √ |
| PaddleX | 小目标检测 | [PP-YOLOE_plus_SOD-largesize-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml) | √ | √ |
| PaddleX | 语义分割 | [Deeplabv3_Plus-R101](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R101.yaml) | √ | √ |
| PaddleX | 图像异常检测 | [STFPM](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/anomaly_detection/STFPM.yaml) | √ | √ |
| PaddleX | 人脸检测 | [PicoDet_LCNet_x2_5_face](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_server_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_server_det.yaml) | √ | √ |
| PaddleX | 文本检测 | [PP-OCRv4_mobile_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_server_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml) | √ | √ |
| PaddleX | 文本识别 | [PP-OCRv4_mobile_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml) | √ | √ |
| PaddleNLP | 自然语言理解模型 | [BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/legacy/model_zoo/bert) | √ | √ |
