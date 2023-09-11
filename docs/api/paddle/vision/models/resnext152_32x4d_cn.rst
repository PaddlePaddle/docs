.. _cn_api_paddle_vision_models_resnext152_32x4d:

resnext152_32x4d
-------------------------------

.. py:function:: paddle.vision.models.resnext152_32x4d(pretrained=False, **kwargs)


ResNeXt-152 32x4d 模型，来自论文 `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`ResNet <cn_api_paddle_vision_models_ResNet>`。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，ResNeXt-152 32x4d 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.resnext152_32x4d
