.. _cn_api_paddle_vision_models_GoogLeNet:

GoogLeNet
-------------------------------

.. py:function:: paddle.vision.models.GoogLeNet(num_classes=1000, with_pool=True)

 GoogLeNet（Inception v1）模型，来自论文 `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_ 。

参数
:::::::::
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回
:::::::::
GoogLeNet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import GoogLeNet

    # build model
    model = GoogLeNet()

    x = paddle.rand([1, 3, 224, 224])
    out, out1, out2 = model(x)

    print(out.shape)
