.. _cn_api_paddle_sparse_deg2rad:

deg2rad
-------------------------------

.. py:function:: paddle.sparse.deg2rad(x, name=None)


逐元素将输入 :attr:`x` 从度转换为弧度，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

数学公式：

.. math::
    deg2rad(x) = \pi * x / 180

参数
:::::::::
    - **x** (SparseTensor) - 输入的稀疏 Tensor，可以为 Coo 或 Csr 格式，数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.sparse.deg2rad
