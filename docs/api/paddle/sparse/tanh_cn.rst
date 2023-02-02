.. _cn_api_paddle_sparse_tanh:

tanh
-------------------------------

.. py:function:: paddle.sparse.tanh(x, name=None)


逐元素计算 :attr:`x` 的双曲正切，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

数学公式：

.. math::
    out = tanh(x)

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.sparse.tanh
