.. _cn_api_paddle_sparse_nn_functional_softmax:

softmax
-------------------------------
.. py:function:: paddle.sparse.nn.functional.softmax(x, axis=-1, name=None)

稀疏 softmax 激活函数，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

当输入 `x` 为 `SparseCsrTensor` 时，仅支持 axis=-1，是由于 Csr 稀疏存储格式，更适合按行读取数据。

如果将 `x` 从稀疏矩阵转换为稠密矩阵， :math:`i`  代表行数， :math:`j` 代表列数，且 axis=-1 时有如下公式：

.. math::
    softmax_ij = \frac{\exp(x_ij - max_j(x_ij))}{\sum_j(exp(x_ij - max_j(x_ij))}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **x** (Tensor) - 输入的稀疏 Tensor，可以是 SparseCooTensor 或 SparseCsrTensor，数据类型为 float32、float64。
    - **axis** (int, 可选) - 指定对输入 SparseTensor 计算 softmax 的轴。对于 SparseCsrTensor，仅支持 axis=-1。默认值：-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同。

代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.functional.softmax
