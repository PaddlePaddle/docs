.. _cn_api_paddle_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.cumsum(x, axis=None, dtype=None, name=None)



沿给定 ``axis`` 计算 Tensor ``x`` 的累加和。

**注意**：结果的第一个元素和输入的第一个元素相同。

参数
:::::::::
    - **x** (Tensor) - 累加的输入，需要进行累加操作的 Tensor。
    - **axis** (int，可选) - 指明需要累加的维度。-1 代表最后一维。默认：None，将输入展开为一维变量再进行累加计算。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float16、float32、float64。如果指定了，那么在执行操作之前，输入 Tensor 将被转换为 dtype。这对于防止数据类型溢出非常有用。默认为：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，累加的结果。

代码示例
::::::::::

COPY-FROM: paddle.cumsum
