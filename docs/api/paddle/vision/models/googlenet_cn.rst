.. _cn_api_paddle_vision_models_googlenet:

googlenet
-------------------------------

.. py:function:: paddle.vision.models.googlenet(pretrained=False, **kwargs)


GoogLeNet（Inception v1）模型，来自论文 `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值：False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`GoogLeNet <cn_api_paddle_vision_models_GoogLeNet>`。

返回
:::::::::

GoogLeNet（Inception v1）模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.googlenet
