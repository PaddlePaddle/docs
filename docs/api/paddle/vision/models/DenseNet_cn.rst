.. _cn_api_paddle_vision_models_DenseNet:

DenseNet
-------------------------------

.. py:class:: paddle.vision.models.DenseNet(layers=121, bn_size=4, dropout=0., class_num=1000)

 DenseNet模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::
  - **layers** (int, 可选) - densenet的层数。默认值：121。
  - **bn_size** (int，可选) - 中间层growth rate的拓展倍数。默认值：4。
  - **dropout** (float, 可选) - dropout rate。默认值：0.。
  - **class_num** (int，可选) - 类别数目，即最后一个全连接层输出的维度。默认值：1000。

返回
:::::::::
DenseNet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import DenseNet

    densenet121 = DenseNet()

    x = paddle.rand([1, 3, 224, 224])
    out = resnet18(x)

    print(out.shape)
