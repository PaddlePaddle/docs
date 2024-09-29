.. _cn_api_paddle_reduce_as:

reduce_as
-------------------------------

.. py:function:: paddle.reduce_as(x, target, name=None)

对 x 在某些维度上求和，使其结果与 target 的 shape 一致。


参数
::::::::::::

  - **x** (Tensor) - 输入变量为多维 Tensor，支持数据类型为 bool、float16、 float32、float64、int8、uint8、int16、uint16、int32、int64、complex64 以及 complex128。
  - **target** (Tensor) - 输入变量为多维 Tensor，x 的 shape 长度必须大于或等于 target 的 shape 长度。支持数据类型为 bool、float16、 float32、float64、int8、uint8、int16、uint16、int32、int64、complex64 以及 complex128。
  - **name** (str, 可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::::

Tensor，输入 x 沿某些轴进行求和，使其结果与 target 的 shape 相同，如果 x 的 type 为 bool 或者为 int32，返回值的 type 将变为 int64，否则返回值的 type 与 x 相同。

代码示例
::::::::::::

COPY-FROM: paddle.reduce_as
