.. _cn_api_paddle_incubate_nn_functional_fused_linear_activation:

fused_linear_activation
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_linear_activation(x, y, bias, trans_x=False, trans_y=False, activation=None)

全连接的线性和激活变换操作符。此方法要求 CUDA 版本 >= 11.6。

参数
:::::::::
    - **x** (Tensor) - 要相乘的输入张量。
    - **y** (Tensor) - 要相乘的权重张量。其秩必须为 2。
    - **bias** (Tensor) - 输入偏置张量，该偏置将添加到矩阵乘法的结果中。
    - **trans_x** (bool，可选) - 是否在乘法之前转置 x。
    - **trans_y** (bool，可选) - 是否在乘法之前转置 y。
    - **activation** (bool，可选) - 激活函数类型，目前仅支持 ``gelu`` , ``relu``。

返回
:::::::::
    - Tensor，输出 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_linear_activation
