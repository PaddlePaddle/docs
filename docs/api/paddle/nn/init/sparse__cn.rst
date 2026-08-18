.. _cn_api_paddle_nn_init_sparse_:

sparse_
-------------------------------

.. py:function:: paddle.nn.init.sparse_(tensor, sparsity, std=0.01)

将二维输入 Tensor 原地初始化为稀疏矩阵。每列指定比例的元素会被置零，其余元素从均值为 0、标准差为 ``std`` 的正态分布采样。

参数
:::::::::

    - **tensor** (Tensor) - 要初始化的二维 Tensor。
    - **sparsity** (float) - 每列中被置零元素的比例。
    - **std** (float，可选) - 非零元素正态分布的标准差。默认值：0.01。

返回
:::::::::

Tensor，原地初始化后的输入 Tensor。

