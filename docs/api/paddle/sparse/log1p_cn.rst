.. _cn_api_paddle_sparse_log1p:

log1p
-------------------------------

.. py:function:: paddle.sparse.log1p(x, name=None)

逐元素计算 :attr:`x+1` 的自然对数，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

数学公式：

.. math::
    out = ln(1+x)

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64、complex64 或 complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.sparse.log1p
