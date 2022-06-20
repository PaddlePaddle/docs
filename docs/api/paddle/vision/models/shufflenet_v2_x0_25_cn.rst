.. _cn_api_paddle_vision_models_shufflenet_v2_x0_25:

shufflenet_v2_x0_25
-------------------------------

.. py:function:: paddle.vision.models.shufflenet_v2_x0_25(pretrained=False, **kwargs)


输出通道缩放比例为 0.25 的 ShuffleNetV2 模型，来自论文 `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`ShuffleNetV2 <cn_api_paddle_vision_models_ShuffleNetV2>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，输出通道缩放比例为 0.25 的 ShuffleNetV2 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.shufflenet_v2_x0_25
