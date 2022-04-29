.. _cn_api_paddle_vision_models_resnet101:

resnet101
-------------------------------

.. py:function:: paddle.vision.models.resnet101(pretrained=False, **kwargs)


101层的resnet模型，来自论文 `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ 。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
resnet101模型，Layer的实例。

代码示例
:::::::::
COPY-FROM: paddle.vision.models.resnet101
