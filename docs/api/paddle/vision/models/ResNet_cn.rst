.. _cn_api_paddle_vision_models_ResNet:

ResNet
-------------------------------

.. py:class:: paddle.vision.models.ResNet(Block, depth=50, width=64, num_classes=1000, with_pool=True, groups=1)

 ResNet模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::
  - **Block** (BasicBlock|BottleneckBlock) - 模型的残差模块。
  - **depth** (int，可选) - resnet模型的深度。默认值：50。
  - **width** (int，可选) - 各个卷积块的每个卷积组基础宽度。默认值：64。
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。
  - **groups** (int，可选) - 各个卷积块的分组数。默认值：1。

返回
:::::::::
ResNet模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    import paddle
    from paddle.vision.models import ResNet
    from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

    # build ResNet with 18 layers
    resnet18 = ResNet(BasicBlock, 18)

    # build ResNet with 50 layers
    resnet50 = ResNet(BottleneckBlock, 50)

    # build Wide ResNet model
    wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

    # build ResNeXt model
    resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)

    x = paddle.rand([1, 3, 224, 224])
    out = resnet18(x)

    print(out.shape)
    # [1, 1000]
