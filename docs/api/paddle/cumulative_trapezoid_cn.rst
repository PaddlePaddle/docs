.. _cn_api_paddle_tensor_cumulative_trapezoid:

cumulative_trapezoid
--------------------------------

.. py:function:: paddle.cumulative_trapezoid(y, x=None, dx=None, axis=-1, name=None)

在指定维度上对输入实现 trapezoid rule 算法。与 paddle.trapezoid 的区别是，所用的求和函数为 cumsum。

参数
:::::::::

    - **y** (Tensor) - 输入多维 Tensor，可选的数据类型为 float16、float32、float64。
    - **x** (Tensor，可选) - ``y`` 中数值对应的浮点数所组成的 Tensor，类型与 ``y`` 相同，形状与 ``y`` 的形状相匹配；若 ``x`` 有输入，已知 ``y`` 的尺寸为 `[d_1, d_2, ... , d_n]` 且 `axis=k`，则 ``x`` 的尺寸只能为 `[d_k]` 或 `[d_1, d_2, ... , d_n]`；如果 ``x`` 为 None，则假定采样点均匀分布 ``dx``。
    - **dx** (float，可选) - 相邻采样点之间的常数间隔；当 ``x`` 和 ``dx`` 均未指定时，``dx`` 默认为 1.0。
    - **axis** (int，可选) - 计算 trapezoid rule 时 ``y`` 的维度。默认值 -1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
返回
:::::::::
Tensor，形状与 ``y`` 的形状与用于计算 trapezoidal rule 时维度有关，类型与 ``y`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.cumulative_trapezoid
