.. _cn_api_paddle_copysign:

copysign
-------------------------------

.. py:function:: paddle.copysign(x, y, name=None)


按照元素计算两个输入 Tensor 的 copysign 大小，由数值和符号组成，其数值部分来自于第一个 Tensor 中的元素，符号部分来自于第二个 Tensor 中的元素。

.. math::

    copysign(x_{i},y_{i})=\left\{\begin{matrix}
    & -|x_{i}| & if \space y_{i} <= -0.0\\
    & |x_{i}| & if \space y_{i} >= 0.0
    \end{matrix}\right.


参数
::::::::::::

    - **x** (Tensor) - 输入的复数值的 Tensor，数据类型为：bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64。
    - **y** (Tensor) - 输入的复数值的 Tensor，数据类型为：bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，输出数据类型与输入数据类型相同。


代码示例 1
::::::::::::

COPY-FROM: paddle.copysign:example1


代码示例 2
::::::::::::

支持广播机制

COPY-FROM: paddle.copysign:example2

代码示例 3
::::::::::::

y 为 +0.0 时

COPY-FROM: paddle.copysign:example_zero1

代码示例 4
::::::::::::

y 为 -0.0 时

COPY-FROM: paddle.copysign:example_zero2
