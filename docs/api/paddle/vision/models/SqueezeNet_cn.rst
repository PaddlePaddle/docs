.. _cn_api_paddle_vision_models_SqueezeNet:

SqueezeNet
-------------------------------

.. py:function:: paddle.vision.models.SqueezeNet(version, num_classes=1000)


SqueezeNet 模型，来自论文 `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ 。

参数
:::::::::
  - **version** (str) - SqueezeNet 的版本，有 "1.0" 和 "1.1" 可选。默认值："1.1"。
  - **num_classes** (int，可选) - 分类的类别数目。默认值：1000。
  - **with_pool** (bool，可选) - 是否定义最后一个全连接层之前的池化层。默认值：True。

返回
:::::::::
SqueezeNet 模型，Layer 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.SqueezeNet
