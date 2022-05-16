.. _cn_api_paddle_vision_models_googlenet:

googlenet
-------------------------------

.. py:function:: paddle.vision.models.googlenet(pretrained=False, **kwargs)

 GoogLeNet（Inception v1）模型，来自论文 `"Going Deeper with Convolutions" <https://arxiv.org/pdf/1409.4842.pdf>`_ 。

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
    from paddle.vision.models import googlenet

    # build model
    model = googlenet()

    # build model and load imagenet pretrained weight
    # model = googlenet(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out, out1, out2 = model(x)

    print(out.shape)
