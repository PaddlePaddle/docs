.. _cn_api_tensor_cn_median:

median
-------------------------------

.. py:function:: paddle.median(x, axis=None, keepdim=False, name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的中位数。

参数
::::::::::
   - **x** (Tensor) - 输入的 Tensor，数据类型为：bool、float16、float32、float64、int32、int64。
   - **axis** (int，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int。``axis`` 值应该在范围 [-D, D) 内，D 是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D`。如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算中位数。默认值为 None。
   - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行中位数计算的结果。如果 ``x`` 的数据类型为 float64，则返回值的数据类型为 float64，反之返回值数据类型为 float32。

代码示例
::::::::::

COPY-FROM: paddle.median
