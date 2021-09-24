.. _cn_api_paddle_vision_models_googlenet:

GoogLeNet
-------------------------------

.. py:function:: paddle.vision.models.GoogLeNet(pretrained=False, **kwargs)

 GoogLeNet模型，来自论文 `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

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

    # build model and load imagenet pretrained weight
    # model = GoogLeNet(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
