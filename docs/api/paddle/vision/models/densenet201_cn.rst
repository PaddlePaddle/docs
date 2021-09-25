.. _cn_api_paddle_vision_models_densenet201:

densenet201
-------------------------------

.. py:function:: paddle.vision.models.densenet201(pretrained=False, **kwargs)

 densenet201模型，来自论文`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ 。


参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
densenet201模型，Layer的实例。

代码示例
:::::::::
.. code-block:: python

    from paddle.vision.models import densenet161

    # build model
    model = densenet201()
