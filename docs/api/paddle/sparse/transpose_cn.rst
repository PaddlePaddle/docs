.. _cn_api_paddle_sparse_transpose:

transpose
-------------------------------

.. py:function:: paddle.sparse.transpose(x, perm, name=None)


根据 :attr:`perm` 对输入的 :attr:`x` 维度进行重排，但不改变数据，
:attr:`x` 必须是多维 SparseTensor 或 COO 格式的 2 维或 3 维 SparseTensor。

.. math::
    out = transpose(x, perm)

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
    - **perm** (list|tuple) - :attr:`perm` 长度必须和 :attr:`x` 的维度相同，并依照 :attr:`perm` 中数据进行重排。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
转置后的稀疏 Tensor, 数据类型和压缩格式与 :attr:`x` 相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.transpose
