.. _cn_api_paddle_vision_models_densenet121:

densenet121
-------------------------------

.. py:function:: paddle.vision.models.densenet121(pretrained=False, **kwargs)

 121层的densenet模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
densenet121模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import densenet121

    # build model
    model = densenet121()

    # build model and load imagenet pretrained weight
    # model = densenet121(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
