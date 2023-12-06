.. _cn_api_paddle_sparse_mv:

mv
-------------------------------

.. py:function:: paddle.sparse.mv(x, vec, name=None)

.. note::
    该 API 从 `CUDA 11.0` 开始支持。

输入 :attr:`x` 为稀疏矩阵，输入 :attr:`vec` 为稠密向量，对 `x` 与 `vec` 计算矩阵与向量相乘。

输入、输出的格式对应关系如下：

.. note::

     x[SparseCsrTensor] @ vec[DenseTensor] -> out[DenseTensor]

     x[SparseCooTensor] @ vec[DenseTensor] -> out[DenseTensor]

该 API 支持反向传播。输入 `x` 的 shape 应该为 `[M, N]` ，输入 `vec` 的 shape 应该为 `[N]` ，输出 `out`
的 shape 为 `[M]` 。

参数
:::::::::
    - **x** (SparseTensor) - 输入的 2D 稀疏 Tensor，可以为 SparseCooTensor|SparseCsrTensor。数据类型为 float32、float64。
    - **vec** (DenseTensor) - 输入 1D 稠密 Tensor，表示一个向量。数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
DenseTensor: 维度为 1，表示一个向量，数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.mv
