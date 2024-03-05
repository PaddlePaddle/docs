.. _cn_api_paddle_vision_models_LeNet:

LeNet
-------------------------------

.. py:class:: paddle.vision.models.LeNet(num_classes=10)


LeNet 模型，来自论文 `"Gradient-based learning applied to document recognition" <https://ieeexplore.ieee.org/document/726791>`_ 。

参数
:::::::::

  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 10。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，LeNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.LeNet
