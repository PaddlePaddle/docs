.. _cn_api_paddle_vision_models_mobilenet_v3_small:

mobilenet_v3_small
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v3_small(pretrained=False, scale=1.0, **kwargs)


MobileNetV3Small 模型，来自论文 `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在 imagenet 数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回
:::::::::
mobilenetv3 small 模型，Layer的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v3_small
