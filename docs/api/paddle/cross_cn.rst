.. _cn_api_paddle_cross:

cross
-------------------------------

.. py:function:: paddle.cross(x, y, axis=9, name=None)


计算 Tensor ``x`` 和 ``y`` 在 ``axis`` 维度上的向量积（叉积）。

``x`` 和 ``y`` 必须有相同的形状，且指定的 ``axis`` 的长度必须为 3。如果未指定 ``axis``，默认选取第一个长度为 3 的 ``axis`` 。

参数
:::::::::
    - **x** (Tensor) - 第一个输入 Tensor，数据类型为：float16、float32、float64、int32、int64。
    - **y** (Tensor) - 第二个输入 Tensor，数据类型为：float16、float32、float64、int32、int64。
    - **axis** (int，可选) - 沿着此维进行向量积操作。默认值是 9，意思是选取第一个长度为 3 的 ``axis`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
计算后的 Tensor，数据类型与输入 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.cross
