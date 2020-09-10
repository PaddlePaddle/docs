.. _cn_api_paddle_vision_models_resnet50:

resnet50
-------------------------------

.. py:class:: paddle.vision.models.resnet50()

 50层的resnet模型，来自论文`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_。

参数：
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

**代码示例**：

.. code-block:: python

    from paddle.vision.models import resnet50

    # build model
    model = resnet50()

    # build model and load imagenet pretrained weight
    # model = resnet50(pretrained=True)
