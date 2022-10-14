.. _cn_api_paddle_incubate_sparse_transpose:

transpose
-------------------------------

.. py:function:: paddle.incubate.sparse.transpose(x, perm, name=None)

根据 perm 对输入的稀疏 Tensor 进行数据重排，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

数学公式：

.. math::
    out = transpose(x, perm)

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64。
    - **perm** (list|tuple) - perm 长度必须和 x 的维度相同，并依照 perm 中数据进行重排。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.incubate.sparse.transpose
