.. _cn_api_paddle_vision_models_VGG:

VGG
-------------------------------

.. py:class:: paddle.vision.models.VGG(features, num_classes=1000, with_pool=True)



 VGG模型，来自论文 `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_ 。

参数
:::::::::
  - **features** (Layer) - vgg模型的特征层。由函数make_layers产生。
  - **num_classes** (int，可选) - 最后一个全连接层输出的维度。如果该值小于等于0，则不定义最后一个全连接层。默认值：1000。
  - **with_pool** (bool，可选) - 是否在最后三个全连接层前使用池化。默认值：True。
  
返回
:::::::::
vgg模型，Layer的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.VGG:code-example
