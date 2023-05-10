.. _cn_api_paddle_sparse_sum:

sum
-------------------------------

.. py:function:: paddle.sparse.sum(x, axis=None, dtype=None, keepdim=False, name=None):

计算给定维度 :attr:`axis` 上稀疏张量 :attr:`x` 元素的和。
输入 :attr:`x` 必须为稀疏压缩格式（ `SparseCooTensor` 或 `SparseCsrTensor`）。

等式为：

.. math::
        out = sum(x, axis, dtype, keepdim)

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为 bool、float16、float32、float64、int32 或 int64。
    - **axis** (int|list|tuple，可选) - 沿着哪些维度进行求和操作。如果为 :attr:`None`，则对 :attr:`x` 的所有元素进行求和并返回一个只有一个元素的 Tensor；否则必须在 :math:`[-rank(x), rank(x))` 范围内。如果 :math:`axis[i] < 0`，则要减少的维度是 :math:`rank + axis[i]`。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型。默认值为 None，表示与输入 Tensor `x` 数据类型一致。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留减少的维度。如果为 True，则结果 Tensor 的维数比 :attr:`x` 少一维，否则与 :attr:`x` 维数一致。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

  ``Tensor``，在指定维度上进行求和运算的 Tensor。如果 `x.dtype='bool'` 或 `x.dtype='int32'`，则其数据类型为 `'int64'`，否则数据类型与 `x` 一致。


代码示例
:::::::::

COPY-FROM: paddle.sparse.sum
