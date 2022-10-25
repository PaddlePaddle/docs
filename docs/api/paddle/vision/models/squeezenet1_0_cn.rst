.. _cn_api_paddle_vision_models_squeezenet1_0:

squeezenet1_0
-------------------------------

.. py:function:: paddle.vision.models.squeezenet1_0(pretrained=False, **kwargs)


SqueezeNet v1.0 模型，来自论文 `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`SqueezeNet <cn_api_paddle_vision_models_SqueezeNet>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，SqueezeNet v1.0 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.squeezenet1_0
