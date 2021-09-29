.. _cn_api_paddle_vision_models_SqueezeNet

SqueezeNet
-------------------------------

.. py:function:: paddle.vision.models.SqueezeNet(version, num_classes=1000)

 SqueezeNet模型，来自论文 `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ 。

参数
:::::::::
  - **version** (str) - SqueezeNet的版本，有"1.0"和"1.1"可选。默认值："1.1"。
  - **num_classes** (int，可选) - 分类的类别数目。默认值：1000。

返回
:::::::::
SqueezeNet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import SqueezeNet

    # build v1.0 model
    model = SqueezeNet(version='1.0')

    # build v1.1 model
    # model = SqueezeNet(version='1.1')

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
