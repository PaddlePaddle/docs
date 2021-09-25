.. _cn_api_paddle_vision_models_shufflenet_v2_swish:

shufflenet_v2_swish
-------------------------------

.. py:function:: paddle.vision.models.shufflenet_v2_swish(pretrained=False, **kwargs)

 使用 swish 进行激活的 ShuffleNetV2 模型，来自论文 `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
shufflenet_v2_swish模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import shufflenet_v2_swish

    # build model
    model = shufflenet_v2_swish()

    # build model and load imagenet pretrained weight
    # model = shufflenet_v2_swish(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
