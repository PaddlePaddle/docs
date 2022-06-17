.. _cn_api_paddle_vision_models_squeezenet1_0:

squeezenet1_0
-------------------------------

.. py:function:: paddle.vision.models.squeezenet1_0(pretrained=False, **kwargs)


SqueezeNet v1.0 模型，来自论文 `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值：False。

返回
:::::::::

SqueezeNet v1.0 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.squeezenet1_0
