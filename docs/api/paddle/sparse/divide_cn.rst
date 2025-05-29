.. _cn_api_paddle_sparse_divide:

divide
-------------------------------

.. py:function:: paddle.sparse.divide(x, y, name=None)

输入 :attr:`x` 与输入 :attr:`y` 逐元素相除，并将各个位置的输出元素保存到返回结果中。

输入 :attr:`x` 与输入 :attr:`y` 必须为相同形状且为相同稀疏压缩格式（同为 `SparseCooTensor` 或同为 `SparseCsrTensor`），如果同为 `SparseCooTensor` 则 `sparse_dim` 也需要相同。

等式为：

.. math::
        out = x / y

- :math:`x`：多维稀疏 Tensor。
- :math:`y`：多维稀疏 Tensor。

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32、int64、complex64 或 complex128。
    - **y** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32、int64、complex64 或 complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和压缩格式与 :attr:`x` 相同，如果是整数相除则返回数据类型为 `float32`。


代码示例
:::::::::

COPY-FROM: paddle.sparse.divide
