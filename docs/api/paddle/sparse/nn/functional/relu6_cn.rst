.. _cn_api_paddle_sparse_nn_functional_relu6:

relu6
-------------------------------
.. py:function:: paddle.sparse.nn.functional.relu6(x, name=None)

稀疏 relu6 激活函数，要求 输入 :attr:`x` 为 `SparseCooTensor` 或 `SparseCsrTensor` 。

.. math::
    relu6(x) = min(max(0, x), 6)

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **x** (Tensor) - 输入的稀疏 Tensor，可以是 SparseCooTensor 或 SparseCsrTensor，数据类型为 float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维稀疏 Tensor, 数据类型和稀疏格式与 :attr:`x` 相同。

代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.functional.relu6
