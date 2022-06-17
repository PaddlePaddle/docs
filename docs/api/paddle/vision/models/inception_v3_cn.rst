.. _cn_api_paddle_vision_models_inception_v3:

inception_v3
-------------------------------

.. py:function:: paddle.vision.models.inception_v3(pretrained=False, **kwargs)


Inception v3 模型，来自论文 `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载在 ImageNet 数据集上的预训练权重。默认值：False。

返回
:::::::::

Inception v3 模型，:ref:`cn_api_fluid_dygraph_Layer` 的实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.inception_v3
