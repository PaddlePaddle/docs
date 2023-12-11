.. _cn_api_paddle_incubate_nn_functional_fused_matmul_bias:

fused_matmul_bias
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_matmul_bias(x, y, bias=None, transpose_x=False, transpose_y=False, name=None)

应用两个张量的矩阵乘法，然后如果提供了偏置，进行偏置加法。

此方法要求 CUDA 版本不低于 11.6。

参数
::::::::::::
    - **x** (Tensor) - 第一个输入 ``Tensor``，被乘 ``Tensor``。
    - **y** (Tensor) - 第二个输入 ``Tensor``，被乘 ``Tensor``。其秩必须为 2。
    - **bias** (Tensor，可选) - 输入的偏置。如果为 None，则不执行偏置加法。否则，偏置将被加到矩阵乘法结果上。默认：None。
    - **transpose_x** (bool，可选) - 是否在乘法前转置 :math:`x`。默认：False。
    - **transpose_y** (bool，可选) - 是否在乘法前转置 :math:`y`。默认：False。
    - **name** (str，可选) - 具体信息请参阅 :ref:`api_guide_Name`。通常无需设置名称，默认为 None。

返回
::::::::::::
输出 ``Tensor``

代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_matmul_bias
