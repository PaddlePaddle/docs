.. _cn_api_paddle_sparse_nn_Softmax:

Softmax
-------------------------------
.. py:class:: paddle.sparse.nn.Softmax(axis=-1, name=None)

稀疏 Softmax 激活层，创建一个可调用对象以计算输入 `x` 的 `Softmax` 。

当输入 `x` 为 `SparseCsrTensor` 时，仅支持 axis=-1，是由于 Csr 稀疏存储格式，更适合按行读取数据。

如果将 `x` 从稀疏矩阵转换为稠密矩阵， :math:`i`  代表行数， :math:`j` 代表列数，且 axis=-1 时有如下公式：

.. math::
    softmax_ij = \frac{\exp(x_ij - max_j(x_ij))}{\sum_j(exp(x_ij - max_j(x_ij))}

参数
::::::::::
    - **axis** (int, 可选) - 指定对输入 SparseTensor 计算 softmax 的轴。对于 SparseCsrTensor，仅支持 axis=-1。默认值：-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - input：任意形状的 SparseTensor。
    - output：和 input 具有相同形状和数据类型的 SparseTensor。

代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.Softmax
