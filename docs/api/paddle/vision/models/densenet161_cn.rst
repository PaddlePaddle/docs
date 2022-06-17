.. _cn_api_paddle_vision_models_densenet161:

densenet161
-------------------------------

.. py:function:: paddle.vision.models.densenet161(pretrained=False, **kwargs)


161 层的 DenseNet 模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值：False。

返回
:::::::::

161 层的 DenseNet 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.densenet161
