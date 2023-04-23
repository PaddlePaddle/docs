.. _cn_api_paddle_vision_models_MobileNetV2:

MobileNetV2
-------------------------------

.. py:class:: paddle.vision.models.MobileNetV2(scale=1.0, num_classes=1000, with_pool=True)


MobileNetV2 模型，来自论文 `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ 。

参数
:::::::::

  - **scale** (float，可选) - 模型通道数的缩放比例。默认值为 1.0。
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值为 True。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，MobileNetV2 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.MobileNetV2
