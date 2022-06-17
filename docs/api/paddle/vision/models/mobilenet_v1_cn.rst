.. _cn_api_paddle_vision_models_mobilenet_v1:

mobilenet_v1
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v1(pretrained=False, scale=1.0, **kwargs)


MobileNetV1 模型，来自论文 `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 imagenet 数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回
:::::::::

MobileNetV1 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.mobilenet_v1
