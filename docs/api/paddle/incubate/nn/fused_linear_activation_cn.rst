.. _cn_api_paddle_incubate_nn_functional_fused_linear_activation:

fused_linear_activation
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_linear_activation(x, y, bias, trans_x=False, trans_y=False, activation=None)

全连接线性和激活变换操作符。该方法要求 CUDA 版本大于等于 11.6。


参数
:::::::::

  - **x** (Tensor) – 需要进行乘法运算的输入 Tensor 。
  - **y** (Tensor) – 需要进行乘法运算的权重 Tensor 。它的阶数必须为2。
  - **bias** (Tensor) – 输入的偏置 Tensor，该偏置会加到矩阵乘法的结果上。
  - **trans_x** (bool, 可选) - 是否在乘法之前对 x 进行矩阵转置。
  - **trans_y** (bool, 可选) - 是否在乘法之前对 y 进行矩阵转置。
  - **activation** (str, 可选) - 目前，可用的激活函数仅限于“GELU”（高斯误差线性单元）和“ReLU”（修正线性单元）。这些激活函数应用于添加偏置之后的输出上。默认值：None。

返回
:::::::::

**Tensor**，变换之后的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_linear_activation
