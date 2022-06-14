.. _cn_api_paddle_vision_models_resnext101_64x4d:

resnext101_64x4d
-------------------------------

.. py:function:: paddle.vision.models.resnext101_64x4d(pretrained=False, **kwargs)

 ResNeXt-101 64x4d模型，来自论文 `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
resnext101_64x4d模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import resnext101_64x4d

    # build model
    model = resnext101_64x4d()

    # build model and load imagenet pretrained weight
    # model = resnext101_64x4d(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
