.. _cn_api_paddle_vision_models_wide_resnet50_2:

wide_resnet50_2
-------------------------------

.. py:function:: paddle.vision.models.wide_resnet50_2(pretrained=False, **kwargs)


Wide ResNet-50-2 模型，来自论文 `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`ResNet <cn_api_paddle_vision_models_ResNet>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，Wide ResNet-50-2 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.wide_resnet50_2
