.. _cn_api_paddle_vision_models_LeNet:

LeNet
-------------------------------

.. py:class:: paddle.vision.models.LeNet(num_classes=10)

 LeNet模型，来自论文 `"Gradient-based learning applied to document recognition" <https://ieeexplore.ieee.org/document/726791>`_ 。

参数
:::::::::
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。默认值：10。


代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import LeNet

    model = LeNet()

    x = paddle.rand([1, 1, 28, 28])
    out = model(x)

    print(out.shape)