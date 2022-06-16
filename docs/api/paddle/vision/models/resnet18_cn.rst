.. _cn_api_paddle_vision_models_resnet18:

resnet18
-------------------------------

.. py:function:: paddle.vision.models.resnet18(pretrained=False, **kwargs)


18 层的 ResNet 模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值：False。

返回
:::::::::

18 层的 ResNet 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.resnet18
