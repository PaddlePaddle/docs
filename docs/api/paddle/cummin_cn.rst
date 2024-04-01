.. _cn_api_paddle_cummin:

cummin
-------------------------------

.. py:function:: paddle.cummin(x, axis=None, dtype='int64', name=None)

沿给定 ``axis`` 计算 Tensor ``x`` 的累积最小值。

.. note::
    结果的第一个元素和输入的第一个元素相同。

参数
::::::::::
    - **x** (Tensor) - 需要进行累积最小值操作的 Tensor。
    - **axis** (int，可选) - 指明需要累积最小值的维度。-1 代表最后一维。默认：None，将输入展开为一维变量再进行累积最小值计算。
    - **dtype** (str，可选) - 输出 Indices 的数据类型，可以是 int32、int64，默认值为 int64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。
返回
::::::::::
    - ``out`` (Tensor)：返回累积最小值操作的结果，累积最小值结果类型和输入 x 相同。
    - ``indices`` (Tensor)：返回对应累积最小值操作的的索引结果。

代码示例
::::::::::

COPY-FROM: paddle.cummin
