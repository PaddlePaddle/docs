.. _cn_api_paddle_sparse_addmm:

addmm
-------------------------------

.. py:function:: paddle.sparse.addmm(input, x, y, beta=1.0, alpha=1.0, name=None)

.. note::
    该 API 从 `CUDA 11.0` 开始支持。

对输入 :attr:`x` 与输入 :attr:`y` 求稀疏矩阵乘法，并将 `input` 加到计算结果上。

数学公式：

..  math::
    out = alpha * x * y + beta * input

输入、输出的格式对应关系如下：

.. note::

     input[SparseCsrTensor] + x[SparseCsrTensor] @ y[SparseCsrTensor] -> out[SparseCsrTensor]

     input[DenseTensor] + x[SparseCsrTensor] @ y[DenseTensor] -> out[DenseTensor]

     input[SparseCooTensor] + x[SparseCooTensor] @ y[SparseCooTensor] -> out[SparseCooTensor]

     input[DenseTensor] + x[SparseCooTensor] @ y[DenseTensor] -> out[DenseTensor]

该 API 支持反向传播，`input` 、 `x` 、 `y` 的维度相同且>=2D，不支持自动广播。

参数
:::::::::
    - **input** (SparseTensor|DenseTensor) - 输入 Tensor，可以为 Coo 或 Csr 格式 或 DenseTensor。数据类型为 float32、float64。
    - **x** (SparseTensor) - 输入 Tensor，可以为 Coo 或 Csr 格式。数据类型为 float32、float64。
    - **y** (SparseTensor|DenseTensor) - 输入 Tensor，可以为 Coo 或 Csr 格式 或 DenseTensor。数据类型为 float32、float64。
    - **beta** (float, 可选) - `input` 的系数。默认：1.0。
    - **alpha** (float, 可选) - `x * y` 的系数。默认：1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
SparseTensor|DenseTensor: 其 Tensor 类型、dtype、shape 与 `input` 相同。


代码示例
:::::::::

COPY-FROM: paddle.sparse.addmm
