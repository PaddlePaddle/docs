.. _cn_api_paddle_utils_data_TensorDataset:

TensorDataset
-------------------------------

.. py:class:: paddle.utils.data.TensorDataset(tensors)

``paddle.io.TensorDataset`` 的别名，同时支持可变位置参数形式 ``paddle.utils.data.TensorDataset(*tensors)``。请参考 :ref:`cn_api_paddle_io_TensorDataset`。

参数
:::::::::

    - **tensors** (Sequence[Tensor]|Tensor) - 第一维长度相同的一组 Tensor，可作为一个序列传入，也可作为可变位置参数逐个传入。
