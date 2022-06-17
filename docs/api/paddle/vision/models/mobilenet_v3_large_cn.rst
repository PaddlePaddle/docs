.. _cn_api_paddle_vision_models_mobilenet_v3_large:

mobilenet_v3_large
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v3_large(pretrained=False, scale=1.0, **kwargs)


MobileNetV3 Large 架构模型，来自论文 `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 ImageNet 数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回
:::::::::

MobileNetV3 Large 架构模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v3_large
