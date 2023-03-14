.. _cn_api_paddle_sparse_nn_LeakyReLU:

LeakyReLU
-------------------------------
.. py:class:: paddle.sparse.nn.LeakyReLU(negative_slope=0.01, name=None)

稀疏 LeakyReLU 激活层，创建一个可调用对象以计算输入 `x` 的 `LeakReLU` 。

.. math::
    LeakyReLU(x)=
        \left\{
            \begin{array}{rcl}
                x, & & if \ x >= 0 \\
                negative\_slope * x, & & otherwise \\
            \end{array}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **negative_slope** (float，可选) - :math:`x < 0` 时的斜率。默认值为 0.01。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - input：任意形状的 SparseTensor。
    - output：和 input 具有相同形状和数据类型的 SparseTensor。

代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.LeakyReLU
