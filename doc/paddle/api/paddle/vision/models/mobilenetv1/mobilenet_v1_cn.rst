.. _cn_api_paddle_vision_models_mobilenet_v1:

mobilenet_v1
-------------------------------

.. py:function:: paddle.vision.models.mobilenet_v1(pretrained=False, scale=1.0, **kwargs)

 MobileNetV1模型，来自论文`"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_。

参数：
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。
  - **scale** (float，可选) - 模型通道数的缩放比例。默认值：1.0。

返回：mobilenetv1模型，Layer的实例。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.vision.models import mobilenet_v1

    # build model
    model = mobilenet_v1()

    # build model and load imagenet pretrained weight
    # model = mobilenet_v1(pretrained=True)

    # build mobilenet v1 with scale=0.5
    model_scale = mobilenet_v1(scale=0.5)

    x = paddle.rand([1, 3, 224, 224])
    out = model(x)

    print(out.shape)
