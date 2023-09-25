.. _cn_api_paddle_vision_models_InceptionV3:

InceptionV3
-------------------------------

.. py:class:: paddle.vision.models.InceptionV3(num_classes=1000, with_pool=True)


Inception v3 模型，来自论文 `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_ 。

参数
:::::::::

  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值为 True。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，Inception v3 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.InceptionV3
