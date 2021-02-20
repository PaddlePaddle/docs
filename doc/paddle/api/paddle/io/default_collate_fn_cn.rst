.. _cn_api_io_cn_default_collate_fn:

default_collate_fn
-------------------------------

.. py:function:: paddle.io.default_collate_fn

``paddle.io.DataLoader`` 中默认组批次函数，输入 ``batch`` 为样本列表，格式如下：

  ``[[filed1，filed2，...]，[filed1，filed2，...]，...]``

此函数将其各按字段组装整合为如下批次数据：

  ``[batch_filed1，batch_filed2，...]``


参数:
    - **batch** (list of numpy array|paddle.Tensor) - 批次数据，格式为样本列表，每个样本包含多个字段，字段为numpy array或Tensor

返回：转换为Tensor的批次数据

返回类型: list of paddle.Tensor

用法见 :ref:`DataLoader <cn_api_fluid_io_DataLoader>`
