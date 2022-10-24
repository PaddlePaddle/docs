.. _cn_api_paddle_sparse_matmul:

matmul
-------------------------------

.. py:function:: paddle.sparse.matmul(x, y, name=None)

.. note::
    该 API 从 `CUDA 11.0` 开始支持。

对输入 :attr:`x` 与输入 :attr:`y` 求稀疏矩阵乘法，`x` 为稀疏 Tensor， `y` 可为稀疏 Tensor 或稠密 Tensor。

输入、输出的格式对应关系如下：

.. note::

     x[SparseCsrTensor] @ y[SparseCsrTensor] -> out[SparseCsrTensor]

     x[SparseCsrTensor] @ y[DenseTensor] -> out[DenseTensor]

     x[SparseCooTensor] @ y[SparseCooTensor] -> out[SparseCooTensor]

     x[SparseCooTensor] @ y[DenseTensor] -> out[DenseTensor]

该 API 支持反向传播，`x` 和 `y` 必须 >= 2D，不支持自动广播。 `x` 的 shape 应该为 `[*, M, K]` ， `y` 的 shape 应该为
`[*, K, N]` ，其中 `*` 为 0 或者批维度。

参数
:::::::::
    - **x** (SparseTensor) - 输入的 Tensor，可以为 Coo 或 Csr 格式。数据类型为 float32、float64。
    - **y** (SparseTensor|DenseTensor) - 输入 Tensor，可以为 Coo 或 Csr 格式 或 DenseTensor。数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
SparseTensor|DenseTensor: 其 Tensor 类型由 `x` 和 `y` 共同决定，数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.matmul
