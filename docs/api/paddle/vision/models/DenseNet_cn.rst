.. _cn_api_paddle_vision_models_DenseNet:

DenseNet
-------------------------------

.. py:class:: paddle.vision.models.DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000)

 DenseNet模型，来自论文`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ 。

参数
:::::::::
  - **growth_rate** (int， 可选) - 每层新增多少滤波器。默认值：32。
  - **block_config** (list of 4 ints， 可选) - 每个池化块的层数。默认值：(6, 12, 24, 16)。
  - **num_init_features** (int， 可选) - 第一层卷积层需要学习的滤波器数量。默认值：64。
  - **bn_size** (int， 可选) - 瓶颈层数的乘数。默认值：4。
  - **drop_rate** (float， 可选) - 通过每层dense层后的dropout率。默认值：0。
  - **num_classes** (int， 可选) - 分类种类数。默认值：1000。
  - **with_pool** (bool，可选) - 是否在最后一层全连接层之前加上池化层。默认值：True。
  
返回
:::::::::
DenseNet模型，Layer的实例。

代码示例
:::::::::

.. code-block:: python

    from paddle.vision.models import DenseNet

    config = (6,12,32,32)

    densenet = DenseNet(block_config=config, num_classes=10)
