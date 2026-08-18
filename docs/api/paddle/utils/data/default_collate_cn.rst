.. _cn_api_paddle_utils_data_default_collate:

default_collate
-------------------------------

.. py:function:: paddle.utils.data.default_collate(batch)

``paddle.io.dataloader.collate.default_collate_fn`` 的别名。该函数接收样本数据组成的列表，递归处理其中的列表、字典、字符串、数值、NumPy 数组和 Tensor，并沿第 0 维堆叠数值、NumPy 数组和 Tensor。请参考 :ref:`cn_api_paddle_io_DataLoader`。

参数
:::::::::

    - **batch** (list) - 样本数据组成的列表。

返回
:::::::::

批处理后的数据；输入中的数值、NumPy 数组和 paddle.Tensor 会被堆叠为 batch 数据。
