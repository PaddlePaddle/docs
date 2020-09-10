.. _cn_api_paddle_vision_models_vgg11:

vgg11
-------------------------------

.. py:class:: paddle.vision.models.vgg11()

 vgg11模型，来自论文`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_。

参数：
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。
  - **batch_norm** (bool, 可选) - 是否在每个卷积层后添加批归一化层。默认值：False。


**代码示例**：

.. code-block:: python

    from paddle.vision.models import vgg11

    # build model
    model = vgg11()

    # build vgg11 model with batch_norm
    model = vgg11(batch_norm=True)
