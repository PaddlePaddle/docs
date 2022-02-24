.. _cn_api_paddle_vision_models_MobileNetV3Large:

MobileNetV3Large
-------------------------------

.. py:class:: paddle.vision.models.MobileNetV3Large(scale=1.0, last_channel=1280, num_classes=1000, with_pool=True)

 MobileNetV3Large模型，来自论文 `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_ 。

参数
:::::::::
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回
:::::::::
mobilenetv3 large模型，Layer的实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.vision.models import MobileNetV3Large

    # build model
    model = MobileNetV3Large(scale=1.0)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
