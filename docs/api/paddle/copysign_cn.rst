.. _cn_api_paddle_copysign:

copysign
-------------------------------

.. py:function:: paddle.copysign(x, y, name=None)


按照元素计算两个输入 Tensor 的 copysign 大小，由数值和符号组成，其数值部分来自于第一个 Tensor 中的元素，符号部分来自于第二个 Tensor 中的元素。
 
.. math::
    out_{i}=
    \left\{
        \begin{array}{lcl}
        -|input_{i}|&ifother_{i}\le-0.0 \\
        |input_{i}|&ifother_{i}\ge0.0\
        \end{array}
    \right.



参数
::::::::::::

    - **x** (Tensor) - 输入的复数值的 Tensor，数据类型为：bfloat16、float16、float32、float64、uint8、int16、int32、int64。
    - **x** (Tensor) - 输入的复数值的 Tensor，数据类型为：bfloat16、float16、float32、float64、uint8、int16、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，若输入 Tensor 的 dtype 不为 float16 或 bfloat16，则输出为 double 的 Tensor。若输入 Tensor 的 dtype 为 bfloat16 或 float16 ，则输出类型与输入类型相同。


代码示例
::::::::::::

COPY-FROM: paddle.copysign
