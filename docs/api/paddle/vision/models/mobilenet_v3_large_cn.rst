.. _cn_api_paddle_vision_models_mobilenet_v3_large:

mobilenet_v3_large
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v3_large(pretrained=False, scale=1.0, **kwargs)


MobileNetV3 Large 架构模型，来自论文 `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值为 1.0。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`MobileNetV3Large <cn_api_paddle_vision_models_MobileNetV3Large>`。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，MobileNetV3 Large 架构模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v3_large
