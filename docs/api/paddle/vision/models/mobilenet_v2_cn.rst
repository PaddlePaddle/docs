.. _cn_api_paddle_vision_models_mobilenet_v2:

mobilenet_v2
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v2(pretrained=False, scale=1.0, **kwargs)


MobileNetV2模型，来自论文 `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在 imagenet 数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回
:::::::::
mobilenetv2 模型，Layer 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v2
