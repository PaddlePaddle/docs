.. _cn_api_paddle_vision_models_mobilenet_v2:

mobilenet_v2
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v2(pretrained=False, scale=1.0, **kwargs)


MobileNetV2 模型，来自论文 `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值为 1.0。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`MobileNetV2 <cn_api_paddle_vision_models_MobileNetV2>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，MobileNetV2 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v2
