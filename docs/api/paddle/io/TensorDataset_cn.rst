.. _cn_api_paddle_io_TensorDataset:

TensorDataset
-------------------------------

.. py:class:: paddle.io.TensorDataset

由 Tensor 列表定义的数据集。

每个 Tensor 的形状应为[N，...]，而 N 是样本数，每个 Tensor 表示样本中一个字段，TensorDataset 中通过在第一维索引 Tensor 来获取每个样本。

参数
::::::::::::

    - **tensors** (list of Tensors) - Tensor 列表，这些 Tensor 的第一维形状相同

返回
::::::::::::
Dataset，由 Tensor 列表定义的数据集

代码示例
::::::::::::

COPY-FROM: paddle.io.TensorDataset
