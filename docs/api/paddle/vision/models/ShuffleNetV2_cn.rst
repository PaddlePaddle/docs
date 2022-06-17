.. _cn_api_paddle_vision_models_ShuffleNetV2:

ShuffleNetV2
-------------------------------

.. py:class:: paddle.vision.models.ShuffleNetV2(scale=1.0, act="relu", num_classes=1000, with_pool=True)


ShuffleNetV2 模型，来自论文 `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

参数
:::::::::
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。
  - **act** (str，可选) - 网络中使用的激活函数。默认值："relu"。
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于 0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回
:::::::::
ShuffleNetV2 模型，Layer 的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import ShuffleNetV2

    shufflenet_v2_swish = ShuffleNetV2(scale=1.0, act="swish")

    x = paddle.rand([1, 3, 224, 224])
    out = shufflenet_v2_swish(x)

    print(out.shape)
