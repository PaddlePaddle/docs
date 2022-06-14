.. _cn_api_paddle_vision_models_resnet152:

resnet152
-------------------------------

.. py:function:: paddle.vision.models.resnet152(pretrained=False, **kwargs)


152 层的 ResNet 模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 ImageNet 数据集上的预训练权重。默认值：False。

返回
:::::::::

152 层的 ResNet 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.resnet152
