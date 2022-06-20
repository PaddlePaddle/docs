.. _cn_api_paddle_vision_models_inception_v3:

inception_v3
-------------------------------

.. py:function:: paddle.vision.models.inception_v3(pretrained=False, **kwargs)


Inception v3 模型，来自论文 `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_ 。

参数
:::::::::

  - **pretrained** (bool，可选) - 是否加载预训练权重。如果为 True，则返回在 ImageNet 上预训练的模型。默认值为 False。
  - **\*\*kwargs** (可选) - 附加的关键字参数，具体可选参数请参见 :ref:`InceptionV3 <cn_api_paddle_vision_models_InceptionV3>`。

返回
:::::::::

:ref:`cn_api_fluid_dygraph_Layer`，Inception v3 模型实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.models.inception_v3
