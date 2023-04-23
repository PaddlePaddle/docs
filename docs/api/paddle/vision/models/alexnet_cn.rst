.. _cn_api_paddle_vision_models_alexnet:

alexnet
-------------------------------

.. py:function:: paddle.vision.models.alexnet(pretrained=False, **kwargs)


AlexNet 模型，来自论文 `"ImageNet Classification with Deep Convolutional Neural Networks" <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`AlexNet <cn_api_paddle_vision_models_AlexNet>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，AlexNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.alexnet
