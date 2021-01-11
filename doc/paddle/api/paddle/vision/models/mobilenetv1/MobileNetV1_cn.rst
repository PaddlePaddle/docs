.. _cn_api_paddle_vision_models_MobileNetV1:

MobileNetV1
-------------------------------

.. py:class:: paddle.vision.models.MobileNetV1(scale=1.0, num_classes=1000, with_pool=True)

 MobileNetV1模型，来自论文 `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_ 。

参数：
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回：mobilenetv1模型，Layer的实例。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.vision.models import MobileNetV1

    model = MobileNetV1()

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
