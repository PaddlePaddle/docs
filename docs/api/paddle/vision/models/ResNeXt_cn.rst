.. _cn_api_paddle_vision_models_ResNeXt:

ResNeXt
-------------------------------

.. py:class:: paddle.vision.models.ResNeXt(layers=layers, cardinality=cardinality, num_classes=1000, with_pool=True)

 ResNeXt模型，来自论文 `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_ 。

参数
:::::::::
  - **layers** (int，可选) - ResNeXt 模型的深度。默认值：50
  - **cardinality** (int，可选) - 模型基数，也即划分组的数量。默认值：32
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **input_image_channel**（int, 可选）- 输入图片的通道数，默认值：3
  - **data_format**（str, 可选）- 数据的格式，默认值："NCHW"

返回
:::::::::
ResNeXt模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import ResNeXt

    resnext50_32x4d = ResNeXt(layers=50, cardinality=32)

    x = paddle.rand([1, 3, 224, 224])
    out = resnext50_32x4d(x)

    print(out.shape)