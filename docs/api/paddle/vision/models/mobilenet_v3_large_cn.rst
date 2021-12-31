.. _cn_api_paddle_vision_models_mobilenet_v3_large:

mobilenet_v3_large
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v3_large(pretrained=False, scale=1.0, last_channel=1280, **kwargs)

 MobileNetV3Large模型，来自论文 `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。
  - **last_channel** (int, 可选) - 倒数第二层的通道数。默认值：1280。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回
:::::::::
mobilenetv3 large模型，Layer的实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.vision.models import mobilenet_v3_large

    # build model
    model = mobilenet_v3_large()

    # build model and load imagenet pretrained weight
    # model = mobilenet_v3_large(pretrained=True)

    # build mobilenet v3 large model with scale=0.5
    model = mobilenet_v3_large(scale=0.5)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
