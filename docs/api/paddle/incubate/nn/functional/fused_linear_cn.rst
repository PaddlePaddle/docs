.. _cn_api_paddle_incubate_nn_functional_fused_linear:

fused_linear
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_linear(x, weight, bias=None, trans_x=False, transpose_weight=False, name=None)
全连接线性变换算子。该方法要求 CUDA 版本大于等于 11.6 。

参数
:::::::::

  - **x** (Tensor) – 需要进行乘法运算的输入 Tensor。
  - **weight** (Tensor) – 需要进行乘法运算的权重 Tensor，它的阶数必须为 2。
  - **bias** (Tensor, 可选) – 输入的偏置 Tensor。如果为 None ，则不执行偏置加法。否则，将偏置加到矩阵乘法的结果上。默认值为 None。
  - **transpose_weight** (bool, 可选) - 是否在乘法之前转置权重。默认值：False。
  - **name** (str, 可选) - 如需详细信息，请参阅 :ref:`api_guide_Name` 。一般无需设置，默认值为 None。

返回
:::::::::

**Tensor**，变换之后的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_linear
