.. _cn_api_paddle_vision_models_squeezenet1_0:

squeezenet1_0
-------------------------------

.. py:function:: paddle.vision.models.squeezenet1_0(pretrained=False, **kwargs)

 squeezenet1_0模型，来自论文 `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
squeezenet1_0模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import squeezenet1_0

    # build model
    model = squeezenet1_0()

    # build model and load imagenet pretrained weight
    # model = squeezenet1_0(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
