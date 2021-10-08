.. _cn_api_paddle_vision_models_InceptionV3:

InceptionV3
-------------------------------

.. py:class:: paddle.vision.models.InceptionV3(num_classes=1000, with_pool=True)

 InceptionV3模型，来自论文 `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_ 。

参数
:::::::::
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回
:::::::::
InceptionV3模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import InceptionV3

    inception_v3 = InceptionV3()

    x = paddle.rand([1, 3, 224, 224])
    out = inception_v3(x)

    print(out.shape)
