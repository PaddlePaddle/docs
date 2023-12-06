.. _cn_api_paddle_sparse_masked_matmul:

masked_matmul
-------------------------------

.. py:function:: paddle.sparse.masked_matmul(x, y, mask, name=None)

.. note::
    该 API 从 `CUDA 11.3` 开始支持。

对输入 :attr:`x` 与输入 :attr:`y` 两个 DenseTensor 求矩阵乘法，同时根据稀疏 Tensor `mask` 进行压缩存储，
返回一个与 `mask` 布局一致的稀疏 Tensor。

输入、输出的格式对应关系如下：

.. note::

     x[DenseTensor] @ y[DenseTensor] * mask[SparseCooTensor] -> out[SparseCooTensor]

     x[DenseTensor] @ y[DenseTensor] * mask[SparseCsrTensor] -> out[SparseCsrTensor]

该 API 支持反向传播，`x` 和 `y` 必须 >= 2D，不支持自动广播。 `x` 的 shape 应该为 `[*, M, K]` ， `y` 的 shape 应该为
`[*, K, N]` ， `mask` 的 shape 应该为 `[*, M, N]` 。其中 `*` 为 0 或者批维度。

参数
:::::::::
    - **x** (DenseTensor) - 输入的 DenseTensor。数据类型为 float32、float64。
    - **y** (DenseTensor) - 输入的 DenseTensor。数据类型为 float32、float64。
    - **mask** (SparseTensor) - 输入的稀疏掩码，是一个稀疏 Tensor，可以为 Coo 或 Csr 格式。数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
SparseTensor: 其 Tensor 类型、dtype、shape 均与 `mask` 相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.masked_matmul
