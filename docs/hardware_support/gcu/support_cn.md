# 燧原 GCU 验证模型

飞桨框架在燧原 GCU 上通过精度验证的模型情况如下：

* PaddleX 使用文档详见：[PaddleX 多硬件使用](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/multi_devices_use_guide.md)
* PaddleNLP 大语言模型多硬件使用文档详见：[PaddleNLP GCU 大语言模型使用文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/gcu/llama/README.md)
* 如果您适配/验证过更多模型，欢迎按照此 [贡献教程](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/how_to_contribute_device.md) 向飞桨开源社区贡献适配结果，我们验证后会更新本模型验证列表

| 模型库 | 模型类型 | 模型名称 | 训练 | 推理 |
| - | - | - | - | - |
| PaddleX | 图像分类 | [ResNet50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/image_classification/ResNet50.yaml) |   | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-S](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml) |   | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-M](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-M.yaml) |   | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml) |   | √ |
| PaddleX | 目标检测 | [PP-YOLOE_plus-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/PP-YOLOE_plus-X.yaml) |   | √ |
| PaddleX | 目标检测 | [RT-DETR-R18](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-R18.yaml) |   | √ |
| PaddleX | 目标检测 | [RT-DETR-R50](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-R50.yaml) |   | √ |
| PaddleX | 目标检测 | [RT-DETR-L](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-L.yaml) |   | √ |
| PaddleX | 目标检测 | [RT-DETR-H](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-H.yaml) |   | √ |
| PaddleX | 目标检测 | [RT-DETR-X](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/object_detection/RT-DETR-X.yaml) |   | √ |
| PaddleX | 文本检测 | [PP-OCRv4_server_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_server_det.yaml) |   | √ |
| PaddleX | 文本检测 | [PP-OCRv4_mobile_det](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml) |   | √ |
| PaddleX | 文本识别 | [PP-OCRv4_server_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml) |   | √ |
| PaddleX | 文本识别 | [PP-OCRv4_mobile_rec](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml) |   | √ |
| PaddleNLP | 大语言模型 | [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/gcu/llama) |   | √ |
