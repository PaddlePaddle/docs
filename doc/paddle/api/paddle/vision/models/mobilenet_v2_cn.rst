.. _cn_api_paddle_vision_models_mobilenet_v2:

mobilenet_v2
-------------------------------

.. py:class:: paddle.vision.models.mobilenet_v2()

 MobileNetV2模型，来自论文`"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_。

参数：
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

**代码示例**：

.. code-block:: python

    from paddle.vision.models import mobilenet_v2

    # build model
    model = mobilenet_v2()

    # build model and load imagenet pretrained weight
    # model = mobilenet_v2(pretrained=True)

    # build mobilenet v2 with scale=0.5
    model = mobilenet_v2(scale=0.5)
