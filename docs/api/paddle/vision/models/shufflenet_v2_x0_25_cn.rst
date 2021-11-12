.. _cn_api_paddle_vision_models_shufflenet_v2_x0_25:

shufflenet_v2_x0_25
-------------------------------

.. py:function:: paddle.vision.models.shufflenet_v2_x0_25(pretrained=False, **kwargs)

 输出通道缩放比例为 0.25 的 ShuffleNetV2 模型，来自论文 `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
shufflenet_v2_x0_25模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import shufflenet_v2_x0_25

    # build model
    model = shufflenet_v2_x0_25()

    # build model and load imagenet pretrained weight
    # model = shufflenet_v2_x0_25(pretrained=True)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
