.. _cn_api_paddle_vision_models_resnet152:

resnet152
-------------------------------

.. py:function:: paddle.vision.models.resnet152(pretrained=False, **kwargs)

 152层的resnet模型，来自论文`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_。

参数：
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回：resnet152模型，Layer的实例。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.vision.models import resnet152

    # build model
    model = resnet152()

    # build model and load imagenet pretrained weight
    # model = resnet152(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
