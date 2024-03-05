.. _cn_api_paddle_vision_models_vgg16:

vgg16
-------------------------------

.. py:function:: paddle.vision.models.vgg16(pretrained=False, batch_norm=False, **kwargs)


16 层的 VGG 模型，来自论文 `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **batch_norm** (bool，可选) - 是否在每个卷积层后添加批归一化层。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`VGG <cn_api_paddle_vision_models_VGG>`。

返回
:::::::::

:ref:`cn_api_paddle_nn_Layer`，16 层的 VGG 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.vgg16
