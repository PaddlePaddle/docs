.. _cn_api_paddle_vision_models_vgg13:

vgg13
-------------------------------

.. py:function:: paddle.vision.models.vgg13(pretrained=False, batch_norm=False, **kwargs)


13 层的 VGG 模型，来自论文 `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 imagenet 数据集上的预训练权重。默认值：False。
  - **batch_norm** (bool，可选) - 是否在每个卷积层后添加批归一化层。默认值：False。

返回
:::::::::

13 层的 VGG 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.vgg13

