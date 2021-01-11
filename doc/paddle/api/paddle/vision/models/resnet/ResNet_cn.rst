.. _cn_api_paddle_vision_models_ResNet:

ResNet
-------------------------------

.. py:class:: paddle.vision.models.ResNet(Block, depth=50, num_classes=1000, with_pool=True)

 ResNet模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数：
  - **Block** (BasicBlock|BottleneckBlock) - 模型的残差模块。
  - **depth** (int，可选) - resnet模型的深度。默认值：50
  - **num_classes** (int, 可选) - 最后一个全连接层输出的维度。如果该值小于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回：ResNet模型，Layer的实例。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.vision.models import ResNet
    from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

    resnet50 = ResNet(BottleneckBlock, 50)

    resnet18 = ResNet(BasicBlock, 18)

    x = paddle.rand([1, 3, 224, 224])
    out = resnet18(x)

    print(out.shape)
