.. _cn_api_io_cn_default_convert_fn:

default_convert_fn
-------------------------------

.. py:function:: paddle.io.default_convert_fn
``paddle.io.DataLoader`` 中默认批次转换函数，输入 ``batch`` 为批次数据，将批次数据中的 ``numpy array`` 数据转换为 ``paddle.Tensor`` 。

一般用于 ``paddle.io.DataLoader`` 中 ``batch_size=None`` 及不需要 ``paddle.io.DataLoader`` 中组批次的情况


参数:
    - **batch** (list of numpy array|paddle.Tensor) - 批次数据，包含numpy array或Tensor的列表

返回：转换为Tensor的批次数据

返回类型: list of paddle.Tensor

用法见 :ref:`DataLoader <cn_api_fluid_io_DataLoader>`
