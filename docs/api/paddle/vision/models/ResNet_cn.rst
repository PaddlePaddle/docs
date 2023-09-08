.. _cn_api_paddle_vision_models_ResNet:

ResNet
-------------------------------

.. py:class:: paddle.vision.models.ResNet(Block, depth=50, width=64, num_classes=1000, with_pool=True, groups=1)


ResNet 模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::

  - **Block** (BasicBlock|BottleneckBlock) - 模型的残差模块。
  - **depth** (int，可选) - ResNet 模型的深度。默认值为 50。
  - **width** (int，可选) - 各个卷积块的每个卷积组基础宽度。默认值为 64。
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于 0，则不定义最后一个全连接层。默认值为 1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值为 True。
  - **groups** (int，可选) - 各个卷积块的分组数。默认值为 1。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，ResNet 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.ResNet
