.. _cn_api_paddle_vision_models_AlexNet:

AlexNet
-------------------------------

.. py:function:: paddle.vision.models.AlexNet(num_classes=1000)


AlexNet 模型，来自论文 `"ImageNet Classification with Deep Convolutional Neural Networks" <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ 。

参数
:::::::::

  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，AlexNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.AlexNet
