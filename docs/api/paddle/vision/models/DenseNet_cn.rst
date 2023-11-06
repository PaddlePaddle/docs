.. _cn_api_paddle_vision_models_DenseNet:

DenseNet
-------------------------------

.. py:class:: paddle.vision.models.DenseNet(layers=121, bn_size=4, dropout=0., num_classes=1000, with_pool=True)


DenseNet 模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::

  - **layers** (int，可选) - DenseNet 的层数。默认值为 121。
  - **bn_size** (int，可选) - 中间层 growth rate 的拓展倍数。默认值为 4。
  - **dropout** (float，可选) - dropout rate。默认值为 :math:`0.0`。
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值为 True。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，DenseNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.DenseNet
