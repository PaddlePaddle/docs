.. _cn_api_paddle_sparse_nn_ReLU:

ReLU
-------------------------------
.. py:class:: paddle.sparse.nn.ReLU(name=None)

稀疏 ReLU 激活层，创建一个可调用对象以计算输入 `x` 的 `ReLU` 。

.. math::
    ReLU(x) = max(x, 0)

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - input：任意形状的 SparseTensor。
    - output：和 input 具有相同形状和数据类型的 SparseTensor。

代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.ReLU
