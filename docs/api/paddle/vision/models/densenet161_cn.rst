.. _cn_api_paddle_vision_models_densenet161:

densenet161
-------------------------------

.. py:function:: paddle.vision.models.densenet161(pretrained=False, **kwargs)


161 层的 densenet 模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在 imagenet 数据集上的预训练权重。默认值：False。

返回
:::::::::
densenet161 模型，Layer 的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import densenet161

    # build model
    model = densenet161()

    # build model and load imagenet pretrained weight
    # model = densenet161(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
