.. _cn_api_paddle_vision_models_densenet121:

densenet121
-------------------------------

.. py:function:: paddle.vision.models.densenet121(pretrained=False, **kwargs)


121 层的 DenseNet 模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 ImageNet 数据集上的预训练权重。默认值：False。

返回
:::::::::

121 层的 DenseNet 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.densenet121
