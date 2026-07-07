.. _cn_api_paddle_cross:

cross
-------------------------------

.. py:function:: paddle.cross(x, y, axis=9, name=None)


计算两个 Tensor 在 ``axis`` 维度上的向量积（叉积）。

输入 Tensor 必须有相同的形状，且指定维度的长度必须为 3。如果未指定 ``axis``，默认选取第一个长度为 3 的维度。

参数
:::::::::
    - **x** (Tensor) - 第一个输入 Tensor，数据类型为：float16、float32、float64、int32、int64、complex64、complex128。别名 ``input``。
    - **y** (Tensor) - 第二个输入 Tensor，数据类型为：float16、float32、float64、int32、int64、complex64、complex128。别名 ``other``。
    - **axis** (int，可选) - 沿着此维度进行向量积操作。默认值为 9，表示选取第一个长度为 3 的维度。别名 ``dim``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
计算后的 Tensor，数据类型与输入 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.cross
