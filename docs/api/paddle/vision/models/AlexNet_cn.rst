.. _cn_api_paddle_vision_models_AlexNet

AlexNet
-------------------------------

.. py:function:: paddle.vision.models.AlexNet(num_classes=1000)

 AlexNet模型，来自论文 `"ImageNet Classification with Deep Convolutional Neural Networks" <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ 。

参数
:::::::::
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。默认值：1000。

返回
:::::::::
alexnet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import AlexNet

    # build model
    model = AlexNet()

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
