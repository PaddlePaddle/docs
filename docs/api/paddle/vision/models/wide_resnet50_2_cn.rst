.. _cn_api_paddle_vision_models_wide_resnet50_2:

wide_resnet50_2
-------------------------------

.. py:function:: paddle.vision.models.wide_resnet50_2(pretrained=False, **kwargs)

 50层的wide_resnet模型，来自论文 `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
wide_resnet50_2模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import wide_resnet50_2

    # build model
    model = wide_resnet50_2()

    # build model and load imagenet pretrained weight
    # model = wide_resnet50_2(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
