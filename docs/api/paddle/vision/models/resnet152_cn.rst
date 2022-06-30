.. _cn_api_paddle_vision_models_resnet152:

resnet152
-------------------------------

.. py:function:: paddle.vision.models.resnet152(pretrained=False, **kwargs)


152 层的 ResNet 模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`ResNet <cn_api_paddle_vision_models_ResNet>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，152 层的 ResNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.resnet152
