.. _cn_api_paddle_vision_models_wideresnet

wideresnet
-------------------------------

.. py:function:: paddle.vision.models.wideresnet(pretrained=False, **kwargs)

 WideResNet模型，来自论文 `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在cifar-10数据集上的预训练权重。默认值：False。

返回
:::::::::
wideresnet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import wideresnet

    # build model
    model = wideresnet()

    # build model and load imagenet pretrained weight
    # model = wideresnet(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)