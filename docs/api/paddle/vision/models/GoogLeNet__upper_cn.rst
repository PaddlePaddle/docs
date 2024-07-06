.. _cn_api_paddle_vision_models_GoogLeNet__upper:

GoogLeNet
-------------------------------

.. py:class:: paddle.vision.models.GoogLeNet(num_classes=1000, with_pool=True)


GoogLeNet（Inception v1）模型，来自论文 `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_ 。

参数
:::::::::

  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值为 True。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，GoogLeNet（Inception v1）模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.GoogLeNet
