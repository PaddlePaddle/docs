.. _cn_api_paddle_vision_models_densenet161:

densenet161
-------------------------------

.. py:function:: paddle.vision.models.densenet161(pretrained=False, **kwargs)

 161层的densenet模型，来自论文 `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
densenet161模型，Layer的实例。

代码示例
:::::::::
COPY-FROM: paddle.vision.models.densenet161