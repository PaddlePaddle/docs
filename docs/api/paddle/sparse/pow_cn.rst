.. _cn_api_paddle_sparse_pow:

pow
-------------------------------

.. py:function:: paddle.sparse.pow(x, factor, name=None)


逐元素计算 :attr:`x` 的幂函数，幂的系数为 `factor`，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。


数学公式：

.. math::
    out = x^{factor}

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64。
    - **factor** (float|int) - 幂函数的系数，可以为 float 或 int。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.sparse.pow
