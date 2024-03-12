.. _cn_api_paddle_median:

median
-------------------------------

.. py:function:: paddle.median(x, axis=None, keepdim=False, mode='avg', name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的中位数。

参数
::::::::::
   - **x** (Tensor) - 输入的 Tensor，数据类型为：bool、float16、float32、float64、int32、int64。
   - **axis** (int，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int。``axis`` 值应该在范围 [-D, D) 内，D 是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D`。如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算中位数。默认值为 None。
   - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
   - **mode** (str，可选) - 当输入 Tensor ``x`` 在 ``axis`` 轴上有偶数个元素时，可选择按照中间两个数的平均值或最小值确定中位数。可选的值是 'avg' 或 'min'。默认为 'avg'。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor 或 (Tensor, Tensor)。 
    若 ``mode == 'avg'``，返回值是一个中位数 Tensor；
    若 ``mode == 'min'`` 且 ``axis`` 是 None，返回值是一个中位数 Tensor；
    若 ``mode == 'min'`` 且 ``axis`` 不是 None，返回值是两个 Tensor，第一个是中位数，第二个是中位数对应的下标；

    当 ``mode == 'avg'`` 时，若 ``x`` 的数据类型是 float64，返回值的数据类型则是 float64，其他输入数据类型情况下返回值的数据类型均是 float32；
    当 ``mode == 'min'`` 时，返回值中，中位数的数据类型与 ``x`` 的数据类型一致，下标的数据类型均为 int64。

代码示例
::::::::::

COPY-FROM: paddle.median
