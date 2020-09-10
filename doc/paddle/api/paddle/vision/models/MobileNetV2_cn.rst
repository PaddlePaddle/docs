.. _cn_api_paddle_vision_models_MobileNetV2:

MobileNetV2
-------------------------------

.. py:class:: paddle.vision.models.MobileNetV2()

 MobileNetV2模型，来自论文`"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_。

参数：
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

**代码示例**：

.. code-block:: python

    from paddle.vision.models import MobileNetV2

    model = MobileNetV2()
