.. _cn_api_tensor_cn_quantile:

quantile
-------------------------------

.. py:function:: paddle.quantile(x, q, axis=None, keepdim=False, name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的分位数。

参数
::::::::::
   - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
   - q (int|float|list) - 待计算的分位数，需要在符合取值范围[0, 1]。如果 ``q`` 是List，其中的每一个q分位数都会被计算，并且输出的首维大小与列表中元素的数量相同。
   - axis (int|list，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是int或内部元素为int类型的list。``axis`` 值应该在范围[-D, D)内，D是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于0，则等价于 :math:`axis + D`。如果 ``axis`` 是list，对给定的轴上的所有元素计算分位数。如果 ``axis`` 是None，则对 ``x`` 的全部元素计算分位数。默认值为None。
   - keepdim (bool，可选) - 是否在输出Tensor中保留减小的维度。如果 ``keepdim`` 为True，则输出Tensor和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为1)。否则，输出Tensor的形状会在 ``axis`` 上进行squeeze操作。默认值为False。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行分位数计算的结果。如果 ``x`` 的数据类型为float64，则返回值的数据类型为float64，反之返回值数据类型为float32。

代码示例
::::::::::

COPY-FROM: paddle.quantile